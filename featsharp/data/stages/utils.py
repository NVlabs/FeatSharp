# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import braceexpand
import hashlib
from logging import Logger
from multiprocessing import Value
from typing import Any, Callable, Iterator, Iterable, List, Union

from torch.utils.data import get_worker_info

import webdataset as wds

from ..filters import get_json_key_extractor

# From OpenCLIP
# https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py#L238
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 512 * 30
_SAMPLE_SHUFFLE_INITIAL = 512 * 10


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def increment(self, ct: int = 1):
        with self.shared_epoch.get_lock():
            self.shared_epoch.value += ct

    def get_value(self):
        return self.shared_epoch.value

def ignore_and_log(logger: Logger):
    def _ignore_exceptions(exn):
        logger.warning(exn)
        return True
    return _ignore_exceptions


def identity(x):
    return x


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def iterator_exhauster(gen_iter_fn: Callable[[], Iterator]):
    while True:
        curr_iter = gen_iter_fn()
        if curr_iter is None:
            break

        while True:
            try:
                v = next(curr_iter)
                yield v
            except StopIteration:
                break


def compute_md5_hash(value: str):
    if value is None:
        print('Warning: Encountered null url')
        value = ''

    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Convert the value to bytes (assuming it is a string)
    if isinstance(value, str):
        value = value.encode('utf-8')

    # Update the hash object with the bytes of the value
    md5_hash.update(value)

    return md5_hash


def extract_caption(gen: Iterable):
    for data in gen:
        if 'txt' not in data:
            json = data['json']
            txt = json['text'] if 'text' in json else json['caption']
            data['txt'] = txt
        # Ignore empty captions
        if data['txt']:
            yield data


def _coerce_ints(items: List[str]):
    ret = []
    for item in items:
        try:
            item = int(item)
        except ValueError:
            pass
        ret.append(item)
    return ret


def get_item_recursive(obj, path: List[Union[str, int]]):
    if not path:
        return obj

    next_p = path[0]

    if not isinstance(next_p, int):
        try:
            next_obj = getattr(obj, next_p)
        except AttributeError:
            next_obj = obj[next_p]
    else:
        next_obj = obj[next_p]


    return get_item_recursive(next_obj, path[1:])


def set_item_recursive(obj, path: List[Union[str, int]], value: Any):
    obj = get_item_recursive(obj, path[:-1])

    k = path[-1]

    if hasattr(obj, k):
        setattr(obj, k, value)
    else:
        obj[k] = value


def get_attribute_path(attr_path: str):
    return _coerce_ints(attr_path.split('.'))


def extract_dict_field(src_path: str, dest_path: str):
    src_path = get_attribute_path(src_path)
    dest_path = get_attribute_path(dest_path)

    def run(gen: Iterable):
        for data in gen:
            src_val = get_item_recursive(data, src_path)
            set_item_recursive(data, dest_path, src_val)

            yield data
    return run

def md5_str_to_bytes():
    key_extractor = get_json_key_extractor([b'md5', b'sha256'])
    def to_bytes(sample):
        key = key_extractor(sample)
        ret = bytes.fromhex(key)
        return ret
    return to_bytes
