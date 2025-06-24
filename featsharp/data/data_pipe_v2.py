# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
import io
import os
import random
from typing import (
    Any, Callable, Iterable, List, Optional, Union, Tuple,
)
from logging import getLogger, Logger
import cv2

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

import webdataset as wds
from data.transforms.default import get_pipeline

from data.transforms.defer_image import DeferImage
from data.transforms.quad import Quad

from data.transforms.equivariant_collate import equivariant_collate
from .filters import (
    UniformColorFilter,
    get_hash_database,
)
from .long_dataset_adaptor import LongDatasetAdaptor
from .prefetcher import IterablePrefetcher
from .stages import (
    ignore_and_log,
    identity,
    SharedEpoch,
    DetShuffle,
    _SAMPLE_SHUFFLE_SIZE,
    _SAMPLE_SHUFFLE_INITIAL,
    WeightedDetShuffle,
    md5_str_to_bytes,
    MultiStreamShuffle,
    MultiPipeSampler,
    extract_caption,
)


def _normalize_listing(data_listing: Union[str, Iterable[Union[str, Tuple[str, str]]]]):
    data_listing = deepcopy(data_listing)

    if isinstance(data_listing, str):
        data_listing = data_listing.split(',')

    for i in range(len(data_listing)):
        listing = data_listing[i]
        if isinstance(listing, str):
            parts = listing.split(':')
            if len(parts) == 1:
                data_path = parts[0]
                rate = 1
            else:
                data_path = parts[0]
                rate = float(parts[1])
            data_listing[i] = [data_path, rate]
        else:
            data_path = listing[0]
            if len(listing) > 1:
                rate = float(listing[1])
            data_listing[i] = [data_path, rate]

    return data_listing


def _simplify_listing(data_listing: Iterable[Tuple[str, float]]):
    uq_listings = dict()
    total_rate = 0
    for data_path, rate in data_listing:
        if data_path in uq_listings:
            uq_listings[data_path] += rate
        else:
            uq_listings[data_path] = rate
        total_rate += rate
    return [(k, v / total_rate) for k, v in uq_listings.items()]


def _get_dataset_pipeline(data_listing: Union[str, Iterable[str]],
                          batch_size: int,
                          is_train: bool,
                          data_weight_mode: str,
                          seed: int,
                          shared_epoch: SharedEpoch,
                          logger: Logger,
                          split_by_node: bool = True,
):
    data_listing = _normalize_listing(data_listing)
    data_listing = _simplify_listing(data_listing)

    rank = dist.get_rank() if dist.is_initialized() else 0
    log_msg = f'Rank {rank}\n'
    for ds_path, rate in data_listing:
        log_msg += f'\tRate {rate:.3f} - Path: {ds_path}\n'

    if rank == 0:
        logger.info('Datasets')
    if dist.is_initialized():
        for i in range(dist.get_world_size()):
            if i == rank:
                logger.info(log_msg)
            dist.barrier()
    else:
        logger.info(log_msg)

    pipes, rates = [], []
    for data_path, rate in data_listing:
        if os.path.isfile(data_path):
            urls = [data_path]
        else:
            urls = list(glob(f'{data_path}/**/*.tar', recursive=True))

        hash_count_db = get_hash_database(data_path)

        if is_train:
            bufsize = min(batch_size * 100, _SAMPLE_SHUFFLE_SIZE)
            if hash_count_db is not None and data_weight_mode is not None:
                shuffler = WeightedDetShuffle(
                    database=hash_count_db,
                    key_extractor=md5_str_to_bytes(),
                    weight_mode=data_weight_mode,
                    bufsize=bufsize,
                    batch_size=batch_size,
                    seed=seed,
                    epoch=shared_epoch,
                    logger=logger,
                )
            else:
                shuffler = DetShuffle(
                    bufsize=bufsize,
                    initial=bufsize // 10,
                    seed=seed,
                    epoch=shared_epoch,
                    logger=logger,
                )

            pipeline = [
                MultiStreamShuffle(
                    urls=urls,
                    epoch=shared_epoch,
                    deterministic=True,
                    reduce_urls_fn=None,
                ),
                # md5_filter,
                shuffler,
            ]
        else:
            pipeline = [
                wds.SimpleShardList(urls),
                wds.split_by_node if split_by_node else identity,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=ignore_and_log(logger)),
                # md5_filter,
            ]

        pipes.append(wds.DataPipeline(*pipeline))
        rates.append(rate)

    if len(pipes) > 1:
        pipe = MultiPipeSampler(pipes, rates, epoch=shared_epoch)
        pipe = wds.DataPipeline(pipe)
    else:
        pipe = pipes[0]

    return pipe


def _prepare_image(num_replicas: int):
    def _stage(data):
        for sample in data:
            img = to_tensor(sample[0])
            valid_mask = torch.ones_like(img)
            ret = tuple(
                (DeferImage(img), Quad(img))
                for _ in range(num_replicas)
            ) + sample[1:]
            yield ret
    return _stage


def _sample_hook():
    def _stage(data):
        ctr = 0
        for sample in data:
            # print(f'Sample {ctr} - "{sample["__key__"]}"\n'
            #     f'\tFile: {sample["__url__"]}\n'
            #     f'\tText: {sample["txt"]}\n'
            #     f'\tUrl: {sample["json"]["url"]}\n'
            #     f'\tNSFW: {sample["json"]["NSFW"]}\n'
            #     '\n'
            # )
            # ctr = (ctr + 1) % 32
            yield sample
    return _stage


_SUPPORTED_IMAGE_FORMATS = 'jpg;jpeg;png;gif;webp;bmp;tiff;img'
def _image_filter(formats: str = _SUPPORTED_IMAGE_FORMATS):
    keys = frozenset(formats.split(';'))
    def _stage(data):
        for sample in data:
            if not keys.isdisjoint(sample.keys()):
                yield sample
    return _stage


def _img_format_handler(img_idx: int = 0, mode='RGB', error_handler = None):
    mode = mode.upper()
    def _stage(data):
        for sample in data:
            img = sample[img_idx]
            if isinstance(img, bytes):
                try:
                    with io.BytesIO(img) as stream:
                        img = Image.open(stream)
                        img.load()
                        img = img.convert(mode)
                except UnidentifiedImageError as e:
                    if error_handler is not None:
                        error_handler(e)
                    continue
            ret = sample[:img_idx] + (img,) + sample[img_idx+1:]
            yield ret
    return _stage


@dataclass
class PipelineConfig:
    steps_per_epoch: int = 2000
    workers: int = 8


def get_data_pipeline(args: PipelineConfig,
                      ds_listing: str,
                      input_sizes: List[Union[int, Tuple[int, int]]],
                      patch_sizes: List[int],
                      batch_size: int,
                      is_train: Union[bool, List[bool]],
                      is_teacher: Optional[Union[bool, List[bool]]],
                      epoch: Union[int, SharedEpoch] = None,
                      seed: int = None,
                      data_weight_mode: str = 'inv_frequency',
                      prefetch: bool = True,
                      label_extractor: Callable = None,
                      label_key: str = None,
                      split_by_node: bool = True,
                      full_equivariance: bool = False,
                      shift_equivariance: bool = False,
):
    if isinstance(is_train, bool):
        is_train = [is_train] + ([False] * (len(input_sizes) - 1))

    max_input_size = max(
        t if isinstance(t, int) else max(t)
        for t in input_sizes
    )

    def listify(o):
        if not isinstance(o, (list, tuple)):
            o = [o] * len(input_sizes)
        return o

    rng = np.random.default_rng(seed)

    steps_per_epoch = args.steps_per_epoch if is_train[0] else None
    l_name = 'train' if is_train else 'val'
    logger = getLogger(f'{l_name}_data_pipeline')

    if is_teacher is None:
        is_teacher = [i > 0 for i in range(is_train)]
    elif isinstance(is_teacher, bool):
        is_teacher = listify(is_teacher)

    unified_seed = rng.bit_generator.random_raw()

    shift_equivariance = listify(shift_equivariance)
    full_equivariance = listify(full_equivariance)

    transforms = [
        get_pipeline(
            input_sizes[0], input_sizes[i], patch_sizes[i],
            is_train[i], is_teacher[i], max_img_size=max_input_size,
            shift_equivariance=shift_equivariance[i], full_equivariance=full_equivariance[i],
            rng=rng, unified_seed=unified_seed,
        )
        for i in range(len(input_sizes))
    ]

    if epoch is None:
        epoch = -1
    if isinstance(epoch, int):
        epoch = SharedEpoch(epoch)

    ds_pipe = _get_dataset_pipeline(
        data_listing=ds_listing,
        batch_size=batch_size,
        is_train=is_train[0],
        data_weight_mode=data_weight_mode,
        seed=seed,
        shared_epoch=epoch,
        logger=logger,
        split_by_node=split_by_node,
    )

    ds_pipe.append(wds.decode('pilrgb', handler=ignore_and_log(logger)))

    if label_extractor is not None and label_key is not None:
        ds_pipe.append(label_extractor)
        addl_tuple = (label_key,)
    else:
        addl_tuple = tuple()

    ds_pipe.pipeline.extend([
        _sample_hook(),
        _image_filter(),
        wds.to_tuple(_SUPPORTED_IMAGE_FORMATS, *addl_tuple, handler=ignore_and_log(logger)),
        _img_format_handler(error_handler=ignore_and_log(logger)),
        UniformColorFilter(),
        _prepare_image(num_replicas=len(input_sizes)),
        wds.map_tuple(*transforms, identity),
        wds.batched(batch_size, partial=not is_train[0], collation_fn=equivariant_collate(len(input_sizes), patch_sizes=patch_sizes)),
    ])

    loader = wds.WebLoader(
        ds_pipe,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )

    if prefetch:
        loader = IterablePrefetcher(loader)

    if steps_per_epoch is not None:
        loader = LongDatasetAdaptor(loader, steps_per_epoch, reset_each_epoch=not is_train)

    return loader, epoch
