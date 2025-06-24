# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import math
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from featsharp.upsampler_modules.featsharp import _DEFAULT_FEATSHARP_INPUTS

from .upsampler_modules import *


class IdentityUpsampler(nn.Module):
    def forward(self, feats, img):
        return feats


class BilinearUpsampler(torch.nn.Module):

    def __init__(self, upsample_factor: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample_factor = upsample_factor

    def forward(self, feats, img):
        if self.upsample_factor is not None:
            h, w = feats.shape[-2:]
            h *= self.upsample_factor
            w *= self.upsample_factor
        else:
            h, w = img.shape[-2:]

        return F.interpolate(feats, (h, w), mode="bilinear", align_corners=False)


class TileUpsampler(nn.Module):
    def __init__(self, upsample_factor: int, model_resolution: int):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.model_resolution = model_resolution
        self.ups_size = upsample_factor * model_resolution

    @property
    def hi_res_input(self):
        return True

    @property
    def input_upsample_factor(self) -> int:
        return self.upsample_factor

    def forward(self, guidance: torch.Tensor, tile_generator: nn.Module,
                return_summary: bool = False,
    ) -> torch.Tensor:
        lr_guidance = F.interpolate(guidance, size=(self.model_resolution, self.model_resolution), mode='bilinear', align_corners=False)

        lr_features = tile_generator(lr_guidance, return_summary=return_summary)
        if return_summary:
            summary, lr_features = lr_features

        guidance = F.interpolate(guidance, size=(self.ups_size, self.ups_size), mode='bilinear', align_corners=False)

        tiled_guidance = rearrange(guidance, 'b c (th h) (tw w) -> (b th tw) c h w',
                                   th=self.upsample_factor, tw=self.upsample_factor,
                                   h=self.model_resolution, w=self.model_resolution)

        tiled_features = tile_generator(tiled_guidance)

        hr_features = rearrange(tiled_features, '(b th tw) c h w -> b c (th h) (tw w)',
                                b=guidance.shape[0],
                                th=self.upsample_factor, tw=self.upsample_factor)

        if return_summary:
            return summary, lr_features, hr_features
        return lr_features, hr_features


class CombinedUpsampler(nn.Module):
    def __init__(self, upsamplers: List[nn.Module], dim: int):
        super().__init__()
        self.upsamplers = nn.ModuleList(upsamplers)

        self.mixer = nn.Sequential(
            nn.Conv2d(3 + len(upsamplers) * dim, 2 * dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )

    def forward(self, source, guidance):
        ups = []
        for upper in self.upsamplers:
            ups.append(upper(source, guidance))

        guidance = F.interpolate(guidance, (source.shape[2] * 16, source.shape[3] * 16), mode='bilinear')
        ups = torch.cat([guidance] + ups, dim=1)

        ret = self.mixer(ups)
        return ret


def _get_upsampler(model_resolution, patch_size, upsampler: str, dim: int, upsample_factor: int):
    multi = upsampler.split('+')

    if len(multi) > 1:
        return CombinedUpsampler([get_upsampler(model_resolution, patch_size, part, dim, upsample_factor) for part in multi], dim)

    if upsampler == 'bilinear':
        return BilinearUpsampler(upsample_factor=upsample_factor)
    elif upsampler == 'tile':
        return TileUpsampler(upsample_factor, model_resolution)
    elif upsampler == 'jbu_stack':
        return JBUStack(dim, upsample_factor)
    elif upsampler.startswith('featsharp'):
        args = upsampler.split('.')
        allow_proj = not 'no-proj' in args
        xpos_qk = 'xpos-qk' in args
        rope_qk = 'rope-qk' in args
        no_attn = 'no-attn' in args
        no_ffn = 'no-ffn' in args
        no_ln = 'no-ln' in args
        progressive = 'prog' in args
        window_size = 7
        inputs = _DEFAULT_FEATSHARP_INPUTS
        for arg in args[1:]:
            if arg.startswith('wnd-'):
                window_size = int(arg.split('-')[1])
            elif arg.startswith('inputs-'):
                inputs = arg.split('-')[1:]

        return FeatSharp(model_resolution, patch_size, dim, upsample_factor, allow_proj=allow_proj,
                              xpos_qk=xpos_qk, rope_qk=rope_qk, window_size=window_size,
                              no_attn=no_attn, no_ffn=no_ffn, inputs=inputs, progressive=progressive,
                              no_ln=no_ln)
    elif upsampler.startswith('loftup'):
        args = upsampler.split('.')
        extra = dict()
        for arg in args[1:]:
            if arg.startswith('n_freqs'):
                extra['n_freqs'] = int(arg.split('-')[1])
            elif arg.startswith('num_layers'):
                extra['num_layers'] = int(arg.split('-')[1])
            elif arg.startswith('lr_pe_type'):
                extra['lr_pe_type'] = arg.split('-')[1]
        return LoftUp(model_resolution=model_resolution, patch_size=patch_size, dim=dim, upsample_factor=upsample_factor, **extra)
    elif upsampler == 'carafe':
        ups_factor = int(round(math.log2(upsample_factor)))
        return CarafeUpsampler(dim, 1, depth=ups_factor)
    elif upsampler.startswith('sapa'):
        ups_factor = int(round(math.log2(upsample_factor)))
        return SAPAUpsampler(dim_x=dim, depth=ups_factor)
    elif upsampler == 'resfu':
        return ReSFUUpsampler(model_resolution, dim, upsample_factor)
    elif upsampler == 'resfu_tiled':
        return ReSFUTiledUpsampler(model_resolution, dim, upsample_factor)
    elif upsampler == 'jafar':
        return JAFAR_Upsampler(model_resolution, dim, upsample_factor)
    else:
        raise ValueError(f"Unknown upsampler {upsampler}")


class HiResWrapper(nn.Module):
    def __init__(self, inner: nn.Module, model_resolution: int):
        super().__init__()
        self.inner = inner
        self.model_resolution = model_resolution

    @property
    def upsample_factor(self) -> int:
        return self.inner.upsample_factor

    @property
    def input_upsample_factor(self) -> int:
        return 1

    def forward(self, guidance: torch.Tensor, tile_generator: nn.Module, return_summary: bool = False) -> torch.Tensor:
        img = F.interpolate(guidance, (self.model_resolution, self.model_resolution),
                            mode='bilinear', align_corners=False)

        op = tile_generator(img, return_summary=return_summary)
        if return_summary:
            summary, source = op
        else:
            source = op

        if return_summary:
            return summary, source, self.inner(source, guidance)
        return source, self.inner(source, guidance)


def get_upsampler(model_resolution, patch_size, *args, **kwargs):
    ret = _get_upsampler(model_resolution, patch_size, *args, **kwargs)

    hi_res_input = getattr(ret, 'hi_res_input', False)

    if not hi_res_input:
        ret = HiResWrapper(ret, model_resolution)

    return ret
