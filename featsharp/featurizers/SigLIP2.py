# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Compose, Normalize

from transformers import AutoModel, AutoProcessor, AutoTokenizer


class SigLIP2Wrapper(nn.Module):
    def __init__(self, clip_model, input_conditioner: Normalize,
                 patch_size: int = 16, is_dynamic: bool = True,
                 version: str = 'so400m', input_size: int = None):
        super().__init__()
        self.inner = clip_model
        self.input_conditioner = input_conditioner
        self.version = version
        self._input_size = input_size

        self._patch_size = patch_size
        self._is_dynamic = is_dynamic

        self.register_buffer('mask', torch.ones(1, 1, dtype=torch.int32))

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def embed_dim(self):
        return self.inner.vision_model.config.hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def detail(self):
        return f'siglip2_{self.version}'

    @property
    def allow_variable_resolution(self):
        return self._is_dynamic

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *args, return_summary: bool = False, **kwargs):
        out_h = x.shape[-2] // self._patch_size
        out_w = x.shape[-1] // self._patch_size

        extra = dict()

        if self._is_dynamic:
            pixel_values = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=self._patch_size, p2=self._patch_size,
                                     h=out_h, w=out_w)
            mask = self.mask.expand(*pixel_values.shape[:2])
            shapes = torch.tensor([(out_h, out_w)] * pixel_values.shape[0], dtype=torch.int64, device=x.device)

            extra = dict(attention_mask=mask, spatial_shapes=shapes)
        else:
            pixel_values = x

        output = self.inner.vision_model(pixel_values=pixel_values, return_dict=True, **extra)

        summary = output.pooler_output
        features = output.last_hidden_state

        features = rearrange(features, 'b (h w) c -> b c h w', h=out_h, w=out_w)

        if return_summary:
            return summary, features
        return features


def get_siglip2_model(version: str, input_size: int = None):
    version_map = {
        'siglip2-so400m-512': ('google/siglip2-so400m-patch16-512', False, 16, 512),
        'siglip2-so400m': ('google/siglip2-so400m-patch16-naflex', True, 16, None),
        'siglip2-g-384': ('google/siglip2-giant-opt-patch16-384', False, 16, 384),
    }
    version_map['siglip2'] = version_map['siglip2-so400m']
    version_map['siglip2-g'] = version_map['siglip2-g-384']

    version, is_dynamic, patch_size, mod_input_size = version_map[version]

    input_size = input_size or mod_input_size
    if input_size is None:
        raise ValueError("Input size must be specified for this model")

    model = AutoModel.from_pretrained(version, trust_remote_code=True)
    # proc = AutoProcessor.from_pretrained(version, trust_remote_code=True)

    # img_proc = proc.image_processor
    # preprocessor = Normalize(
    #     img_proc.image_mean,
    #     img_proc.image_std,
    # )
    preprocessor = Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    )

    model = SigLIP2Wrapper(model, preprocessor, is_dynamic=is_dynamic, patch_size=patch_size, input_size=input_size)

    return model
