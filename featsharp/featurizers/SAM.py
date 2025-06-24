# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import Optional
import timm
from timm.models.vision_transformer_sam import VisionTransformerSAM, Attention
import torch
from torch import nn

from einops import rearrange
from torchvision.transforms import Compose, Normalize

from ..util import rank_gate


class SAMFeaturizer(nn.Module):
    def __init__(self, size: str = 'huge'):
        super().__init__()
        self.size = size

        with rank_gate():
            self.model = timm.create_model(f'samvit_{size}_patch16.sa1b', pretrained=True)
            data_config = timm.data.resolve_model_data_config(self.model)
            self.input_conditioner = timm.data.create_transform(**data_config, is_training=False)

        # We want to match prior to the neck
        self.model.neck = nn.Identity()
        self.model.head = nn.Identity()

        self.model.eval().cuda()

    @property
    def input_size(self):
        return 1024

    @property
    def embed_dim(self):
        return self.model.embed_dim

    @property
    def patch_size(self):
        return 16

    @property
    def detail(self):
        return f'sam_{self.size}'

    @property
    def allow_variable_resolution(self):
        return False

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        parts = []
        # slice_size = max(1, 4 // (img.shape[-2] // 1024))
        slice_size = 8
        if img.shape[-2] > 1024:
            slice_size = 1

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            for off in range(0, img.shape[0], slice_size):
                end = min(img.shape[0], off + slice_size)
                parts.append(self.model.forward_features(img[off:end]).clone())

        output = torch.cat(parts)

        return output.float()
