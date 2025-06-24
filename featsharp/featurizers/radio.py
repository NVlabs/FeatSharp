# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import Optional
import torch
from torch import nn

from einops import rearrange
from torchvision.transforms import Compose, Normalize

from ..util import rank_gate


class RADIOFeaturizer(nn.Module):
    def __init__(self, arch: str = 'radio_v2.5-l', base_res: int = 432, vitdet_window_size: Optional[int] = None):
        super().__init__()
        self.arch = arch
        self.input_size = base_res

        with rank_gate():
            self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=arch, vitdet_window_size=vitdet_window_size, skip_validation=True)
        preprocess = self.model.make_preprocessor_external()

        self.input_conditioner = Compose([
            Normalize(preprocess.norm_mean.flatten().tolist(), preprocess.norm_std.flatten().tolist()),
        ])

        self.model.eval().cuda()

    @property
    def embed_dim(self):
        return self.model.embed_dim

    @property
    def patch_size(self):
        return self.model.patch_size

    @property
    def detail(self):
        return f'radio_{self.arch}'

    @property
    def output_size(self):
        return self.input_size // self.patch_size

    @property
    def upsample_factor(self):
        return 1

    @torch.no_grad()
    def forward(self, img: torch.Tensor, return_summary: bool = False):
        h = img.shape[-2] // self.patch_size
        w = img.shape[-1] // self.patch_size

        summary, features = self.model(img)
        features = rearrange(features, 'b (h w) c -> b c h w', h=h, w=w)
        if return_summary:
            return summary, features
        return features
