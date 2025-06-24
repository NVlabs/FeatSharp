# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import time
from typing import List, Optional
import torch
from torch.utils import cpp_extension
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_EXT = cpp_extension.load(
    name='spatial_transform_cpp',
    sources=[os.path.join(_current_dir, 'spatial_transform.cpp')],
    extra_cflags=['-O3'],
    verbose=True,
)


def spatial_transform(inputs: torch.Tensor, stms: torch.Tensor, output_width: int, output_height: int,
                      method: str = 'bilinear', background: float = 0.0, verbose: bool = False) -> torch.Tensor:
    assert not inputs.is_cuda, "Inputs must be on CPU"
    assert not stms.is_cuda, "STMs must be on CPU"

    ret = _EXT.spatial_transform_cpu(inputs, stms, output_width, output_height, method, background, verbose)
    return ret
