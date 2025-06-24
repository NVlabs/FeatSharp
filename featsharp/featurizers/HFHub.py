# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import Union, Optional

import torch
from torch import nn
import torchvision.transforms as T

from einops import rearrange
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, PaliGemmaProcessor

from ..util import rank_gate


class PaliGemmaFeaturizer(nn.Module):
    def __init__(self, repo: str, dtype: Optional[Union[str, torch.dtype]] = None,
                 interpolate_pos_encoding: Optional[bool] = None, **extra_args):
        super().__init__()

        self.repo = repo
        self.interpolate_pos_encoding = interpolate_pos_encoding

        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype = torch.float32
            elif dtype == 'float16':
                dtype = torch.float16
            elif dtype == 'bfloat16':
                dtype = torch.bfloat16
            else:
                raise ValueError(f'Unsupported dtype: {dtype}')

        self.dtype = dtype or torch.float32

        if dtype is not None:
            extra_args['torch_dtype'] = dtype
            rev = str(dtype).split('.')[-1]
            extra_args['revision'] = rev

        with rank_gate():
            config = PaliGemmaConfig.from_pretrained(repo)
            model = PaliGemmaForConditionalGeneration.from_pretrained(repo, config=config, **extra_args)
            processor = PaliGemmaProcessor.from_pretrained(repo, **extra_args)
            processor = processor.image_processor

        self.input_conditioner = T.Normalize(processor.image_mean, processor.image_std)

        self.model = model.vision_tower.vision_model
        self.model.eval().cuda()

    @property
    def embed_dim(self):
        return self.model.config.hidden_size

    @property
    def input_size(self):
        return self.model.config.image_size

    @property
    def patch_size(self):
        return self.model.config.patch_size

    @torch.no_grad()
    def forward(self, img: torch.Tensor, interpolate_pos_encoding: Optional[bool] = None, return_summary: bool = False):
        if interpolate_pos_encoding is None:
            interpolate_pos_encoding = self.interpolate_pos_encoding
        if interpolate_pos_encoding is None:
            interpolate_pos_encoding = False

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            outputs = self.model(img, return_dict=False, interpolate_pos_encoding=interpolate_pos_encoding)

        features = outputs[0].to(torch.float32)
        h = img.shape[-2] // self.patch_size
        w = img.shape[-1] // self.patch_size

        if return_summary:
            summary = torch.mean(features, dim=1)

        features = rearrange(features, 'b (h w) c -> b c h w', h=h, w=w)

        if return_summary:
            return summary, features
        return features


def get_model(repo: str, num_classes: int = None, **extra_args):
    if 'paligemma' in repo:
        return PaliGemmaFeaturizer(repo, **extra_args)

    raise ValueError(f'Unsupported model: {repo}')
