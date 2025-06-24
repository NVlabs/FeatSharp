# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import Dict, NamedTuple, Optional
import torch
from torch import nn

from featsharp.featurizers.util import get_featurizer
from featsharp.layers import BiasBuffer, get_normalizer
from featsharp.upsamplers import BilinearUpsampler, HiResWrapper, IdentityUpsampler, TileUpsampler, get_upsampler
from featsharp.enable_spectral_reparam import enable_spectral_reparam, disable_parametrizations
from timm.models.vision_transformer import Block as TxBlock


class UpsampleFeatures(NamedTuple):
    low_res: torch.Tensor
    high_res: torch.Tensor
    summary: Optional[torch.Tensor] = None


class CondBiasWrapper(nn.Module):
    def __init__(self, cond_bias: nn.Module):
        super().__init__()
        self.cond_bias = cond_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_shape = x.shape
        xf = x.flatten(2).permute(0, 2, 1)
        xd = self.cond_bias(xf)
        xd = xd.permute(0, 2, 1).reshape(prev_shape)
        x = xd
        return x



class Upsampler(nn.Module):
    def __init__(self, featurizer: nn.Module, upsampler: nn.Module,
                 normalizer: nn.Module = None,
                 bias_buffer: nn.Module = None,
                 cond_bias: nn.Module = None,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.upsampler = upsampler
        self.normalizer = normalizer if normalizer is not None else nn.Identity()
        self.bias_buffer = bias_buffer if bias_buffer is not None else nn.Identity()
        self.cond_bias = CondBiasWrapper(cond_bias) if cond_bias is not None else nn.Identity()

    @property
    def upsample_factor(self) -> int:
        return self.upsampler.upsample_factor

    @property
    def input_upsample_factor(self) -> int:
        return self.upsampler.input_upsample_factor

    @property
    def input_conditioner(self):
        return self.featurizer.input_conditioner

    @property
    def patch_size(self):
        return self.featurizer.patch_size

    @property
    def input_size(self):
        return self.featurizer.input_size * self.input_upsample_factor

    @property
    def output_size(self):
        return self.featurizer.input_size // self.patch_size * self.upsample_factor

    @property
    def embed_dim(self):
        return self.featurizer.embed_dim

    @input_conditioner.setter
    def input_conditioner(self, v):
        self.featurizer.input_conditioner = v

    def get_lr_features(self, x: torch.Tensor, return_summary: bool = False) -> torch.Tensor:
        feats = self.featurizer(x, return_summary=return_summary)
        if return_summary:
            summary, feats = feats
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        feats = self.normalizer(feats)
        feats = self.bias_buffer(feats)
        feats = self.cond_bias(feats)
        if return_summary:
            return summary, feats
        return feats

    def forward(self, x: torch.Tensor, denormalize: bool = False, return_summary: bool = False) -> UpsampleFeatures:
        ups = self.upsampler(x, self.get_lr_features, return_summary=return_summary)
        if return_summary:
            summary, lr_y, hr_y = ups
        else:
            lr_y, hr_y = ups

        if denormalize:
            lr_y = self.denormalize(lr_y)
            hr_y = self.denormalize(hr_y)

        if return_summary:
            return UpsampleFeatures(lr_y, hr_y, summary=summary)
        return UpsampleFeatures(lr_y, hr_y)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.normalizer, 'denormalize'):
            x = self.normalizer.denormalize(x)
        return x


def _is_spectral_reparam(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any('_sn_version' in k for k in state_dict.keys())


def _load_state_dict(mod: nn.Module, state_dict: Dict[str, torch.Tensor]):
    is_spectral = _is_spectral_reparam(state_dict)
    if is_spectral:
        enable_spectral_reparam(mod, init_norm_to_current=False)
    mod.load_state_dict(state_dict, strict=True)
    if is_spectral:
        disable_parametrizations(mod)
    pass


def load_from_file(checkpoint_path: str, featurizer: Optional[nn.Module] = None, do_upsample: bool = True):
    if checkpoint_path.startswith('tile-'):
        ups_factor = int(checkpoint_path[5:-1])
        ups = TileUpsampler(ups_factor, featurizer.input_size)
        return Upsampler(featurizer, ups)
    elif checkpoint_path.startswith('bilinear-'):
        ups_factor = int(checkpoint_path.split('-')[1][:-1])
        ups = HiResWrapper(BilinearUpsampler(ups_factor), featurizer.input_size)
        return Upsampler(featurizer, ups)

    feat_chk = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    if featurizer is None:
        train_cfg = feat_chk.get('train_config', None)
        if train_cfg is None:
            raise ValueError("Trying to load a legacy checkpoint which doesn't have the training config defined. Please manually construct the featurizer.")

        featurizer, _, _ = get_featurizer(train_cfg['model_type'], **train_cfg['model_args'])

    ups_args = feat_chk['upsample_args']

    if do_upsample:
        upsampler = get_upsampler(
            model_resolution=featurizer.input_size,
            patch_size=featurizer.patch_size,
            upsampler=ups_args['type'],
            dim=featurizer.embed_dim,
            upsample_factor=ups_args['factor'],
        )
        ups_sd = get_prefix_state_dict(feat_chk['model_state'], 'upsampler')
        ups_sd = strip_prefix(ups_sd, 'module.')  # Handle the DDP poison in the checkpoint
        _load_state_dict(upsampler, ups_sd)
    else:
        upsampler = HiResWrapper(IdentityUpsampler(), featurizer.input_size)

    norm_type = ups_args['normalizer']
    normalizer = get_normalizer(norm_type, featurizer.embed_dim)
    norm_sd = get_prefix_state_dict(feat_chk['model_state'], 'normalizer')
    normalizer.load_state_dict(norm_sd)

    output_size = featurizer.input_size // featurizer.patch_size
    bias_buffer = BiasBuffer(featurizer.embed_dim, output_size)
    bias_sd = get_prefix_state_dict(feat_chk['model_state'], 'bias_buffer')
    bias_sd = strip_prefix(bias_sd, 'module.')
    if bias_sd:
        _load_state_dict(bias_buffer, bias_sd)

    cond_bias = None
    cond_bias_sd = get_prefix_state_dict(feat_chk['model_state'], 'cond_bias_transform')
    if cond_bias_sd:
        cond_bias_sd = strip_prefix(cond_bias_sd, 'module.')
        cond_bias = TxBlock(featurizer.embed_dim, num_heads=16, init_values=1e-8)
        _load_state_dict(cond_bias, cond_bias_sd)

    return Upsampler(featurizer, upsampler, normalizer, bias_buffer, cond_bias=cond_bias)


def get_prefix_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str):
    if not prefix.endswith('.'):
        prefix = prefix + '.'
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str):
    return {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }
