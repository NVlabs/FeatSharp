# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import os
from typing import Dict
import sys

from .sample_filter import SampleFilterStage, get_json_key_extractor

_IMAGE_TYPES = ['jpg', 'jpeg', 'gif', 'webp', 'tiff', 'png']

def _save_filtered_image(key: str, occ: int, sample: Dict[str, bytes]):
    if occ != 1:
        return

    base_path = 'hash_filtered_images'
    os.makedirs(base_path, exist_ok=True)

    for img_type in _IMAGE_TYPES:
        if img_type in sample:
            break
    else:
        print(f'Warning: Couldn\'t extract image data. Keys: {sample.keys()}', file=sys.stderr)
        return

    fname = f'{base_path}/{key}.{img_type}'
    if not os.path.exists(fname):
        with open(fname, 'wb') as fd:
            fd.write(sample[img_type])
        pass


def get_hash_filter_stage(dataset, save_filtered_images: bool = False):
    config_fn = getattr(dataset, 'get_filter_config', None)

    if config_fn is not None:
        database, hash_set, hash_key = config_fn()
    else:
        database = None
        hash_set = set()
        hash_key = 'md5'

    return database, SampleFilterStage(
        blacklist=hash_set,
        key_extractor=get_json_key_extractor(hash_key),
        filter_callback=_save_filtered_image if save_filtered_images else None,
    )
