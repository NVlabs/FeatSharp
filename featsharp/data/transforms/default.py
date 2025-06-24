# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from argparse import Namespace
import random
from typing import Any, Dict, List, Tuple, Union, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from timm.layers import to_2tuple

from . import transforms as tx


def _get_zoom(scale: Union[float, Tuple[float, float]] = 1.0, random_seed: int = None):
    if isinstance(scale, float) and scale != 1.0:
        return tx.RandomZoomTransform(min_ratio=scale, max_ratio=1/scale, retain_dims=False, random_seed=random_seed)
    if isinstance(scale, tuple):
        return tx.RandomZoomTransform(min_ratio=scale[0], max_ratio=scale[1], retain_dims=False, random_seed=random_seed)
    return tx.Identity()


def _get_deformations(img_size,
                      rng: np.random.Generator,
                      unified_seed: int,
                      unified_scale: Union[float, Tuple[float, float]] = 1.0,
                      unified_stretch: float = 1.0,
                      flip_prob: float = 0.0,
                      rot_max: float = 0.0,
                      perspective_scale: Tuple[float, float] = (0, 0),
                      individual_scale: Union[float, Tuple[float, float]] = 1.0,
                      patch_size: int = None,
                      jitter: bool = False,
                      full_equivariance: bool = False
):
    ret = []

    jitter = jitter and not full_equivariance

    def get_seed():
        return rng.bit_generator.random_raw()

    # Unified Region
    ret.append(_get_zoom(unified_scale, random_seed=unified_seed + 41))

    if unified_stretch != 1.0:
        ret.extend([
            # These will stretch the image a bit along each dimension
            tx.RandomZoomTransform(min_ratio=unified_stretch, max_ratio=1/unified_stretch, fixed_width=True, retain_dims=False, random_seed=unified_seed + 42),
            tx.RandomZoomTransform(min_ratio=unified_stretch, max_ratio=1/unified_stretch, fixed_height=True, retain_dims=False, random_seed=unified_seed + 43),
        ])

    ret.append(tx.RandomCropTransform(height=img_size[0], width=img_size[1], random_seed=unified_seed + 44,
                                      quant_shift=patch_size if jitter else None, jitter=jitter, jit_seed=get_seed()))

    # Individual Region
    if full_equivariance:
        if flip_prob > 0:
            ret.append(tx.RandomFlipTransform(prob_apply=flip_prob, random_seed=get_seed()))
        ret.append(_get_zoom(individual_scale, random_seed=get_seed()))

        ret.append(tx.CenterCropTransform(height=img_size[0], width=img_size[1]))

        if max(perspective_scale) > 0:
            ret.append(tx.RandomPerspectiveTransform(scale=perspective_scale, prob_apply=1.0, random_seed=get_seed()))

        if rot_max > 0:
            ret.append(tx.RandomRotationTransform(abs_max=rot_max, random_seed=get_seed()))

    return ret


def get_pipeline(student_size: Union[int, Tuple[int, int]],
                 img_size: Union[int, Tuple[int, int]],
                 patch_size: int,
                 is_train: bool, is_teacher: bool,
                 max_img_size: int,
                 rng: np.random.Generator,
                 unified_seed: int,
                 full_equivariance: bool = False,
                 shift_equivariance: bool = False):
    student_size = to_2tuple(student_size)
    img_size = to_2tuple(img_size)

    is_hi_res = max_img_size > 512

    jitter_size = patch_size #* (img_size[0] / student_size[0])

    # TODO: Only enable this for SAM

    if True or not is_hi_res:
        # Ensure that the smallest image dimension is large enough
        transforms = [
            tx.MaxSizeTransform(height=img_size[0], width=img_size[1], smallest=True),
        ]

        if is_train:
            transforms.extend(_get_deformations(img_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(1.0, 2.0),
                # unified_stretch=0.98,
                flip_prob=0.0,
                individual_scale=.95,
                rot_max=5.0,
                perspective_scale=(0.01, 0.05),
                patch_size=patch_size,
                jitter=False,
                full_equivariance=full_equivariance,
            ))
        elif is_teacher:
            transforms.extend(_get_deformations(img_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(1.0, 2.0),
                # stretch=0.98,
                patch_size=jitter_size,
                jitter=shift_equivariance,
            ))
    else:
        # The hi-res pipeline is more tailored to the spatial features, and doing fine-grained
        # tasks. So in this case, it's better to downsample and pad the image so that
        # we see a lot more of it
        transforms = [
            # So the large dimension will be set to img_size, and we'll pad the rest
            tx.MaxSizeTransform(height=img_size[0], width=img_size[1], smallest=True),
        ]

        if is_train:
            transforms.extend(_get_deformations(img_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(0.9, 1.4),
                stretch=0.98,
                # flip_prob=0.01,
                rot_max=1.0,
                perspective_scale=(0.01, 0.05),
                patch_size=patch_size,
                jitter=False,
                full_equivariance=full_equivariance
            ))
        elif is_teacher:
            transforms.extend(_get_deformations(img_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(0.9, 1.4),
                # stretch=0.98,
                patch_size=jitter_size,
                jitter=shift_equivariance,
            ))

    rand_pad = shift_equivariance and is_train and not is_teacher

    def get_seed():
        return rng.bit_generator.random_raw()

    if is_train and not is_teacher:
        crop = tx.RandomCropTransform(height=img_size[0], width=img_size[1], quant_shift=patch_size, random_seed=get_seed())
    elif is_teacher:
        crop = tx.CropTransform(0, 0, height=img_size[0], width=img_size[1])
    else:
        crop = tx.CenterCropTransform(height=img_size[0], width=img_size[1])


    transforms.extend([
        # Finally, add padding to make sure the image fits
        tx.PadToTransform(height=img_size[0], width=img_size[1], quant_pad=patch_size, rand_pad=rand_pad),
        crop,
    ])

    transforms = tx.CompositeTransform(transforms)

    return transforms
