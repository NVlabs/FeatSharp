# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import os
from typing import List

import open_clip
import torch
from torch import nn
from torch import distributed as dist
import torch.nn.functional as F

from einops import rearrange
from timm.layers import to_2tuple

from ..util import rank_gate


class OpenCLIPFeaturizer(nn.Module):
    def __init__(self, model_name: str, checkpoint: str, ac_dtype: str = None, **kwargs):
        super().__init__()

        self.model_name = model_name
        self.checkpoint = checkpoint

        with rank_gate():
            model, self.input_conditioner = open_clip.create_model_from_pretrained(
                model_name,
                pretrained=checkpoint,
            )

        if model_name == 'ViT-SO400M-14-SigLIP-384':
            model = SigLIPWrapper(model.visual)
            self.embed_dim = model.embed_dim
            self.patch_size = model.patch_size
            self.input_size = model.input_size
            self.inter_enabled = False
        else:
            model = model.visual
            model.output_tokens = True
            model_config = open_clip.get_model_config(model_name)
            vis_cfg = model_config['vision_cfg']
            self.embed_dim = vis_cfg['width']
            self.patch_size = vis_cfg['patch_size']
            self.input_size = vis_cfg['image_size']
            self.inter_enabled = True
        self.model = model

        if ac_dtype == 'bfloat16':
            self.ac_dtype = torch.bfloat16
        elif ac_dtype == 'float16':
            self.ac_dtype = torch.float16
        elif not ac_dtype:
            self.ac_dtype = None
        else:
            raise ValueError(f'Unrecognized autocast dtype: {ac_dtype}')

        self.model.eval().cuda()

        pass

    @property
    def detail(self):
        return f'{self.model_name}_{self.checkpoint}'

    @property
    def output_size(self):
        return self.input_size // self.patch_size

    @property
    def upsample_factor(self):
        return 1

    @torch.no_grad()
    def forward(self, img: torch.Tensor, return_summary: bool = False):
        with torch.autocast('cuda', dtype=self.ac_dtype or torch.float32, enabled=self.ac_dtype is not None), \
             interpolate_pos_embed(self.model, self.input_size, img.shape[-2:], self.patch_size, enabled=self.inter_enabled):
            summary, features = self.model(img)
        h = img.shape[-2] // self.patch_size
        w = img.shape[-1] // self.patch_size

        features = rearrange(features, 'b (h w) c -> b c h w', h=h, w=w).float()

        if return_summary:
            return summary, features
        return features


class interpolate_pos_embed:
    def __init__(self, model: nn.Module, input_size: int, target_size: int, patch_size: int, enabled: bool = True):
        self.model = model
        self.input_size = to_2tuple(input_size)
        self.target_size = to_2tuple(target_size)
        self.patch_size = patch_size
        self.enabled = enabled

        self.prev_embed = None

    def __enter__(self, *args, **kwargs):
        if self.input_size == self.target_size or not self.enabled:
            return

        self.prev_embed = self.model.positional_embedding
        ds_res = tuple(sz // self.patch_size for sz in self.input_size)

        cls_token = self.prev_embed[:1]
        pos_embed_2d = rearrange(self.prev_embed[1:], '(b h w) c -> b c h w',
                                 b=1, h=ds_res[0], w=ds_res[1])

        ds_targ_size = tuple(sz // self.patch_size for sz in self.target_size)

        pos_embed_2d = F.interpolate(pos_embed_2d, ds_targ_size, mode='bilinear', align_corners=False)

        pos_emb_2d = rearrange(pos_embed_2d, 'b c h w -> (b h w) c')
        pos_emb = torch.cat([cls_token, pos_emb_2d])
        self.model.positional_embedding = nn.Parameter(pos_emb, requires_grad=False)

    def __exit__(self, *args, **kwargs):
        if self.prev_embed is not None:
            self.model.positional_embedding = self.prev_embed

class SigLIPWrapper(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner.trunk
        self.inner.patch_embed.img_size = (378, 378)

    @property
    def embed_dim(self):
        return self.inner.embed_dim

    @property
    def patch_size(self):
        return self.inner.patch_embed.patch_size[0]

    @property
    def input_size(self):
        return self.inner.patch_embed.img_size[0]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mod = self.inner

        dynamic = False
        if x.shape[-2:] != (378, 378):
            dynamic = True
            mod.patch_embed.strict_img_size = False
            mod.dynamic_img_size = True

        ph, pw = tuple(v // self.patch_size for v in x.shape[-2:])

        x = mod.patch_embed(x)
        if dynamic:
            x = rearrange(x, 'b (h w) c -> b h w c', h=ph, w=pw)
        x = mod._pos_embed(x)
        x = mod.patch_drop(x)
        x = mod.norm_pre(x)
        for block in mod.blocks:
            x = block(x)
        x = mod.norm(x)
        summary = mod.attn_pool(x)

        if dynamic:
            mod.patch_embed.strict_img_size = True
            mod.dynamic_img_size = False

        return summary, x
