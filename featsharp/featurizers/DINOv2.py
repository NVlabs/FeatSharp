# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import math
from types import MethodType
import warnings
from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose, Normalize

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from einops import rearrange

# from featsharp.featurizers.dinov2.layers.attention import Attention


def dv2_sdpa(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]
    x = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=False,
        dropout_p=self.attn_drop.p if self.training else 0.,
        scale=self.scale,
    )
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def dv2_test_forward(self, x: torch.Tensor) -> torch.Tensor:
    old_way = self.old_forward(x)
    new_way = dv2_sdpa(self, x)

    assert torch.allclose(old_way, new_way), "Didn't work!"
    return new_way


class DINOv2Featurizer(nn.Module):

    def __init__(self, arch: str, base_res: int, **kwargs):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', arch)

        for n, m in self.model.named_modules():
            if n.endswith('.attn'):
                m.old_forward = m.forward
                m.forward = MethodType(dv2_sdpa, m)

        self.input_size = base_res
        self.input_conditioner = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        pass

    @property
    def patch_size(self):
        return self.model.patch_size

    @property
    def embed_dim(self):
        return self.model.num_features

    def get_cls_token(self, img):
        return self.model.forward(img)

    @torch.no_grad()
    def forward(self, img, return_summary: bool = False):
        h = img.shape[2] // self.patch_size
        w = img.shape[3] // self.patch_size

        with torch.autocast('cuda', dtype=torch.float16, enabled=True):
            op = self.model.forward_features(img)
            summary = op['x_norm_clstoken']
            features = op["x_norm_patchtokens"]

        features = rearrange(features, 'b (h w) c -> b c h w', h=h, w=w)

        if return_summary:
            return summary, features
        return features.float()
