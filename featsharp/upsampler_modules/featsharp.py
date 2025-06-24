# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
"""
Implements FeatSharp (https://www.arxiv.org/abs/2502.16025)
"""

from enum import Enum
from functools import partial
import math
import os
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from einops import rearrange

from ..featurizers.dinov2.layers.block import Block
from ..featurizers.dinov2.layers import SwiGLUFFNFused
from .featup import JBUStack
from .resfu import ReSFU
from .jafar import JAFAR_Upsampler
from .util import create_tiles, get_prime_factors, cumprod

_AF_DEFAULT_WND_SIZE = 7

# [1,2]    [1,1,2,2]
# [3,4] -> [3,3,4,4]
# [5,6]    [5,5,6,6]
def duplicate_interleave(m):
    return m.view(-1, 1).repeat(1, 2).view(m.shape[0], -1)


# 0,1,2,3,4,5,6,7 -> -1,0,-3,2,-5,4,-7,6
def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


class XPosEmbedding2D(torch.nn.Module):
    """Implementation of xPos based on RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(
        self,
        head_dim: int,
        base=50000,
        scale_base=512,
        rope: bool = False,
    ):
        super().__init__()
        half_dim = head_dim // 2
        self.half_dim = half_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.token_shape_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None
        self.scale_cached: torch.Tensor | None = None
        self.scale_base = scale_base
        self.register_buffer("scale",
                             (torch.arange(0, half_dim, 2) + 0.4 * half_dim) / (1.4 * half_dim))
        if rope:
            self.scale.fill_(1)
            self.scale_base = 1

    def cos_sin(
        self,
        token_shape: Tuple[int, int],
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if token_shape != self.token_shape_cached:
            self.token_shape_cached = token_shape
            y = torch.arange(token_shape[0], device=device, dtype=self.inv_freq.dtype)
            x = torch.arange(token_shape[1], device=device, dtype=self.inv_freq.dtype)
            x, y = torch.meshgrid(x, y, indexing='xy')

            y_freqs = torch.einsum("i,j->ij", y.flatten(), self.inv_freq)
            x_freqs = torch.einsum("i,j->ij", x.flatten(), self.inv_freq)

            y_scales = self.scale ** y.flatten().div(self.scale_base)[:, None]
            x_scales = self.scale ** x.flatten().div(self.scale_base)[:, None]

            freqs = torch.cat([y_freqs, x_freqs], dim=-1)
            emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

            scales = torch.cat([y_scales, x_scales], dim=-1)
            scales = torch.repeat_interleave(scales, repeats=2, dim=-1)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]
            self.scale_cached = scales[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)
            self.scale_cached = self.scale_cached.type(dtype)

        return self.cos_cached, self.sin_cached, self.scale_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, token_shape: Tuple[int, int]):
        batch, seq_len, head_dim = q.shape
        cos, sin, scale = self.cos_sin(token_shape, q.device, q.dtype)
        # scale = self.scale**torch.arange(seq_len).to(self.scale).div(self.scale_base)[:, None]
        # scale = torch.repeat_interleave(scale, 2, dim=-1).to(q.device)
        # scale = torch.cat([scale, scale], dim=-1)
        # scale = 1
        def pad_to_fit(t: torch.Tensor, value=0):
            if t.shape[1] < seq_len:
                t = F.pad(t, (0, 0, 0, seq_len - t.shape[1], 0, 0), value=value)
            return t

        cos, sin = map(pad_to_fit, (cos, sin))
        scale = pad_to_fit(scale, value=1)

        return (
            (q * cos * scale) + (rotate_every_two(q) * sin * scale),
            (k * cos * (1 / scale)) + (rotate_every_two(k) * sin * (1 / scale)),
        )


def get_alibi_kernel(nhead: int, width: int, device: str = 'cuda'):
    alibi_bias = torch.arange(2, nhead + 2, step=1, dtype=torch.float32, device=device).floor_divide_(2)
    alibi_bias = 1 / (2 ** alibi_bias)

    def alibi(score, b, h, q_idx, kv_idx):
        qr = q_idx // width
        qc = q_idx % width

        kvr = kv_idx // width
        kvc = kv_idx % width

        rdelta = qr - kvr
        cdelta = qc - kvc

        delta = rdelta * ((h + 1) % 2) + cdelta * (h % 2)

        bias = alibi_bias[h] * -torch.abs(delta)
        return score + bias

    return alibi


def get_vanilla_kernel(nhead: int, width: int, device: str = 'cuda'):
    def kernel(score, b, h, q_idx, kv_idx):
        return score
    return kernel


def local_mask_kernel(width: int, window_size: int = 7):
    radius = window_size // 2
    numel = width * width
    def kernel(b, h, q_idx, kv_idx):
        qr = q_idx // width
        qc = q_idx % width

        kvr = kv_idx // width
        kvc = kv_idx % width

        rdelta = qr - kvr
        cdelta = qc - kvc

        valid_pos = torch.logical_or(torch.logical_and(q_idx < numel, kv_idx < numel), q_idx == kv_idx)
        within_window = torch.logical_and(
            torch.logical_and(rdelta >= -radius, rdelta <= radius),
            torch.logical_and(cdelta >= -radius, cdelta <= radius)
        )

        return torch.logical_and(valid_pos, within_window)
    return kernel


class MultiIdentity(nn.Module):
    def forward(self, *args):
        return args


class AlibiAttention(nn.Module):
    unique_instances = dict()

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 proj_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 width: int = None, xpos_qk: bool = False, rope_qk: bool = False,
                 window_size: int = _AF_DEFAULT_WND_SIZE,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.window_size = window_size

        key = (dim, num_heads, width, window_size)
        if key not in AlibiAttention.unique_instances:
            AlibiAttention.unique_instances[key] = len(AlibiAttention.unique_instances)
        self.instance_id = AlibiAttention.unique_instances[key]

        np2 = 2 ** int(math.ceil(math.log2(head_dim)))
        self.head_dim = np2
        self.real_head_dim = head_dim

        self.qkv = nn.Linear(dim, num_heads * np2 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.alibi_kernel = get_alibi_kernel(num_heads, width) if not (xpos_qk or rope_qk) else get_vanilla_kernel(num_heads, width)
        self.kernel_block_mask = None
        self.width = width
        torch._functorch.config.donated_buffer=False
        # self.flex_attention = torch.compile(flex_attention, fullgraph=True) if self.instance_id == 0 else flex_attention
        self.flex_attention = torch.compile(flex_attention, fullgraph=True)

        self._initial_bs = None

        if xpos_qk or rope_qk:
            self.xpos = XPosEmbedding2D(self.num_heads * self.head_dim, rope=rope_qk)
        else:
            self.xpos = MultiIdentity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.all(torch.isfinite(x))

        real_bs = x.shape[0]
        if self._initial_bs is not None and self._initial_bs != x.shape[0]:
            x = torch.cat([
                x,
                torch.zeros(self._initial_bs - x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device)
            ], dim=0)
        else:
            self._initial_bs = x.shape[0]

        B, N, C = x.shape

        pad = ((N + 127) // 128) * 128 - N

        if pad:
            x = torch.cat([
                x,
                torch.zeros(B, pad, C, dtype=x.dtype, device=x.device)
            ], dim=1)

        _, N2, _ = x.shape

        if self.kernel_block_mask is None:
            self.kernel_block_mask = create_block_mask(
                local_mask_kernel(self.width, window_size=self.window_size),
                B=B, H=self.num_heads,
                Q_LEN=N2,
                KV_LEN=N2,
                _compile=True,
                device=x.device,
            )

        # qkv = self.qkv(x).reshape(B, N2, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N2, 3, self.num_heads * self.head_dim).permute(2, 0, 1, 3)
        assert torch.all(torch.isfinite(qkv))
        q, k, v = qkv
        q, k = self.xpos(q, k, (self.width, self.width))[:2]

        def head_reshape(t: torch.Tensor):
            return t.reshape(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k, v = map(head_reshape, (q, k, v))

        x = self.flex_attention(q, k, v, self.alibi_kernel, scale=self.scale, block_mask=self.kernel_block_mask)

        x = x[..., :self.real_head_dim]

        assert torch.all(torch.isfinite(x))

        x = x.transpose(1, 2).reshape(B, N2, C)

        if pad > 0:
            x = x[:, :N]

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x[:real_bs]

        return x


class ResidualSingle(nn.Module):
    def __init__(self, dim: int, inner: nn.Module, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inner(self.norm(x))
        minc = min(x.shape[-1], y.shape[-1])
        x = x[..., :minc]
        y = y[..., :minc]
        x = x + y
        return x


_DEFAULT_FEATSHARP_INPUTS = ['jbu', 'tiles']

class AttentionLevel(nn.Module):
    def __init__(self, model_resolution: int, patch_size: int, feat_dim: int, upsample_factor: int,
                 xpos_qk: bool = False, rope_qk: bool = False, window_size: int = _AF_DEFAULT_WND_SIZE,
                 no_attn: bool = False, no_ffn: bool = False, inputs = _DEFAULT_FEATSHARP_INPUTS,
                 no_ln: bool = False, *args, **kwargs):
        super().__init__()
        self.model_resolution = model_resolution
        self.patch_size = patch_size
        self.upsample_factor = upsample_factor
        self.feature_dim = feat_dim

        guide_sizes = dict(bilinear=feat_dim, jbu=feat_dim, tiles=feat_dim, resfu=feat_dim, rgb=16, jafar=feat_dim)

        self.lr_num_patches = model_resolution // patch_size
        self.num_patches = self.lr_num_patches * upsample_factor
        guide_dim = sum(guide_sizes[i] for i in inputs)

        self.inputs = inputs
        if 'tiles' in inputs:
            self.tile_pos = nn.Parameter(torch.zeros(1, feat_dim, upsample_factor, upsample_factor))
        if 'jbu' in inputs:
            self.jbu = JBUStack(feat_dim, upsample_factor)
        if 'rgb' in inputs:
            self.rgb_proj = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        if 'resfu' in inputs:
            self.resfu = ReSFU(dim_y=3, dim_x=feat_dim, scale_factor=upsample_factor, normx=False, normy=False)
        if 'jafar' in inputs:
            self.jafar = JAFAR_Upsampler(model_resolution, feat_dim, upsample_factor)

        self._input_fetchers = {
            'bilinear': self._get_bilinear,
            'jbu': self._get_jbu,
            'tiles': self._get_tiles,
            'rgb': self._get_rgb,
            'resfu': self._get_resfu,
            'jafar': self._get_jafar,
        }

        norm_layer = nn.LayerNorm if not no_ln else nn.Identity

        if not no_attn and not no_ffn:
            self.block = Block(
                guide_dim,
                num_heads=16,
                attn_class=partial(AlibiAttention, width=self.num_patches, xpos_qk=xpos_qk, rope_qk=rope_qk, window_size=window_size),
                mlp_ratio=4.0,
                ffn_layer=partial(SwiGLUFFNFused, hidden_features=feat_dim * 4, out_features=feat_dim),
                init_values=1e-6,
                norm_layer=norm_layer,
            )
        elif no_attn:
            self.block = ResidualSingle(
                guide_dim,
                SwiGLUFFNFused(in_features=guide_dim, hidden_features=feat_dim * 4, out_features=feat_dim,
                               act_layer=nn.GELU),
                norm_layer=norm_layer,
            )
        elif no_ffn:
            self.block = ResidualSingle(
                guide_dim,
                AlibiAttention(dim=guide_dim, num_heads=16, width=self.num_patches, xpos_qk=xpos_qk, rope_qk=rope_qk, window_size=window_size),
                norm_layer=norm_layer,
            )
        else:
            self.block = ResidualSingle(
                guide_dim,
                nn.Linear(in_features=guide_dim, out_features=feat_dim),
                norm_layer=norm_layer,
            )

    def forward(self, guidance: torch.Tensor, lr_source: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        feat_buff = torch.cat([
            self._input_fetchers[i](guidance, lr_source, tiles)
            for i in self.inputs
        ], dim=1)

        h, w = feat_buff.shape[-2:]
        feat_buff = rearrange(feat_buff, 'b c h w -> b (h w) c')
        assert torch.all(torch.isfinite(feat_buff))

        feat_buff = self.block(feat_buff)

        feat_buff = rearrange(feat_buff[..., :self.feature_dim], 'b (h w) c -> b c h w', h=h, w=w)

        return feat_buff

    def _get_jbu(self, guidance: torch.Tensor, lr_source: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        guidance_ups = F.interpolate(guidance, (self.num_patches, self.num_patches), mode='bilinear', align_corners=False)
        lr_ups = self.jbu(lr_source, guidance_ups)
        return lr_ups

    def _get_bilinear(self, guidance: torch.Tensor, lr_source: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr_source, (self.num_patches, self.num_patches), mode='bilinear', align_corners=False)

    def _get_tiles(self, guidance: torch.Tensor, lr_source: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        tile_pos = self.tile_pos.repeat_interleave(self.lr_num_patches, dim=2).repeat_interleave(self.lr_num_patches, dim=3)
        tiles = tiles + tile_pos
        return tiles

    def _get_rgb(self, guidance: torch.Tensor, lr_source: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        ret = F.interpolate(guidance, (self.num_patches, self.num_patches), mode='bilinear', align_corners=False)
        ret = self.rgb_proj(ret)
        return ret

    def _get_resfu(self, guidance: torch.Tensor, lr_source: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        small_guidance = F.adaptive_avg_pool2d(guidance, (self.num_patches, self.num_patches))
        ups = self.resfu(small_guidance, lr_source)
        return ups

    def _get_jafar(self, guidance: torch.Tensor, lr_source: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        return self.jafar(lr_source, guidance)



class FeatSharp(nn.Module):
    def __init__(self, model_resolution: int, patch_size: int, feat_dim: int, upsample_factor: int,
                 allow_proj: bool = True, xpos_qk: bool = False, rope_qk: bool = False,
                 window_size: int = _AF_DEFAULT_WND_SIZE,
                 no_attn: bool = False, no_ffn: bool = False, inputs = _DEFAULT_FEATSHARP_INPUTS, progressive: bool = False,
                 no_ln: bool = False, *args, **kwargs):
        super().__init__()
        self.model_resolution = model_resolution
        self.patch_size = patch_size
        self.upsample_factors = [upsample_factor] if not progressive else get_prime_factors(upsample_factor)
        self.num_levels = len(self.upsample_factors)
        self.feature_dim = feat_dim
        self.upsample_factor = upsample_factor

        ress = cumprod(self.upsample_factors, inclusive=False)

        self.ups = nn.ModuleList([
            AttentionLevel(model_resolution * resf, patch_size, feat_dim, uf,
                           allow_proj=allow_proj, xpos_qk=xpos_qk, rope_qk=rope_qk, window_size=window_size,
                           no_attn=no_attn, no_ffn=no_ffn, inputs=inputs, no_ln=no_ln)
            for resf, uf in zip(ress, self.upsample_factors)
        ])

    @property
    def hi_res_input(self):
        return True

    @property
    def input_upsample_factor(self) -> int:
        return self.upsample_factor

    def forward(self, guidance: torch.Tensor, tile_generator: Callable[[torch.Tensor], torch.Tensor],
                return_summary: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tlop = create_tiles(guidance, tile_generator, self.model_resolution, upsample_factors=self.upsample_factors,
                                      return_summary=return_summary,
        )

        if return_summary:
            summary, tile_levels, _ = tlop
        else:
            tile_levels, _ = tlop

        lr_source = source = tile_levels[0]

        for i in range(1, len(tile_levels)):
            tiles = tile_levels[i]
            assert torch.all(torch.isfinite(source))
            assert torch.all(torch.isfinite(tiles))
            source = self.ups[i - 1](guidance, source, tiles)

        if return_summary:
            return summary, lr_source, source
        return lr_source, source
