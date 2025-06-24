# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import torch
from torch import nn

class CarafeUpsampler(nn.Module):

    def __init__(self, dim, kernel_size, depth: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from mmcv.ops import CARAFEPack
        self.ups = nn.ModuleList([
            CARAFEPack(dim, up_kernel=kernel_size, up_group=1, scale_factor=2)
            for _ in range(depth)
        ])

    def forward(self, source, guidance):
        x = source
        for up in self.ups:
            x = up(source)
        return x
