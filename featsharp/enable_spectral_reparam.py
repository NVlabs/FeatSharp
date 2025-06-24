# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from contextlib import contextmanager
from logging import getLogger
import math
from typing import List, Union, Tuple
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm

from timm.models.vision_transformer import Attention, Mlp

_EPS = 1e-5
_PROJ_REPARAM = True


class _SNReweight(_SpectralNorm):
    def __init__(self, weight: torch.Tensor, *args, init_norm_to_current: bool = False, alpha: float = 0.05, version: int = 2, **kwargs):
        super().__init__(weight, *args, **kwargs)

        self.alpha = alpha
        self.version = version
        self.register_buffer('_sn_version', torch.tensor(version))

        if init_norm_to_current:
            # This will set the numerator to match the denominator, which should preserve the original values
            init_scale = self._get_sigma(weight, n_power_iterations=20).item()
        else:
            init_scale = 1.0

        if version == 1:
            init_value = init_scale
        elif version == 2:
            t = init_scale - alpha
            if t < _EPS:
                getLogger("spectral_reparam").warn(f'The initialized spectral norm {init_scale} is too small to be represented. Setting to {_EPS} instead.')
                t = _EPS

            init_value = math.log(math.exp(t) - 1)
        else:
            raise ValueError(f'Unsupported version: {version}')

        # Make 2D so that weight decay gets applied
        self.sn_scale = nn.Parameter(torch.tensor([[init_value]], dtype=torch.float32, device=weight.device))

    # Re-implementing this because we need to make division by sigma safe
    def _get_sigma(self, weight: torch.Tensor, n_power_iterations: int = None) -> torch.Tensor:
        if not n_power_iterations:
            n_power_iterations = self.n_power_iterations
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            sigma = weight.norm()
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))

        return sigma + self.eps

    def forward(self, weight: torch.Tensor, *args, **kwargs):
        dtype = weight.dtype
        sigma = self._get_sigma(weight, *args, **kwargs)

        if self.version == 1:
            scale = self.sn_scale
        elif self.version == 2:
            scale = F.softplus(self.sn_scale) + self.alpha
        else:
            raise ValueError(f'Unsupported version: {self.version}')

        scale = scale.float() / sigma.float()

        y = weight * scale

        if dtype in (torch.float16, torch.bfloat16):
            y = y.to(dtype)
        return y

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version_key = f'{prefix}_sn_version'
        if version_key not in state_dict:
            self.version = 1
            state_dict[version_key] = torch.tensor(1)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class _ChunkedSNReweight(nn.Module):
    def __init__(self, weight: torch.Tensor, num_chunks: int, *args, init_norm_to_current: bool = False, **kwargs):
        super().__init__()

        self.num_chunks = num_chunks
        parts = weight.split(weight.shape[0] // num_chunks, dim=0)

        self.parts = nn.ModuleList([
            _SNReweight(p, *args, init_norm_to_current=init_norm_to_current, **kwargs)
            for p in parts
        ])

    def forward(self, weight: torch.Tensor, *args, **kwargs):
        parts = weight.split(weight.shape[0] // self.num_chunks, dim=0)

        parts = [
            fn(p)
            for fn, p in zip(self.parts, parts)
        ]

        return torch.cat(parts, dim=0)


class _AttnSNReweight(_ChunkedSNReweight):
    def __init__(self, weight: torch.Tensor, *args, init_norm_to_current: bool = False, renorm_values: bool = False, **kwargs):
        super().__init__(weight, 3, *args, init_norm_to_current=init_norm_to_current, **kwargs)

        if not renorm_values:
            self.parts[2] = nn.Identity()


def enable_spectral_reparam(model: Union[nn.Module, List[nn.Module]],
                            n_power_iterations: int = 1,
                            eps: float = 1e-6,
                            init_norm_to_current: bool = False,
                            renorm_values: bool = True,
                            renorm_mlp: bool = True,
                            renorm_proj: bool = None):
    if isinstance(model, (list, tuple)):
        for sub in model:
            enable_spectral_reparam(sub, n_power_iterations=n_power_iterations, eps=eps,
                                    init_norm_to_current=init_norm_to_current, renorm_values=renorm_values,
                                    renorm_mlp=renorm_mlp, renorm_proj=renorm_proj)
        return

    print('Enabling spectral reparametrization')
    args = dict(n_power_iterations=n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current)
    visited_prefixes = set()

    if renorm_proj is None:
        renorm_proj = _PROJ_REPARAM

    def parametrize_linear(linear: nn.Linear | nn.Conv2d):
        parametrize.register_parametrization(
            linear,
            'weight',
            _SNReweight(linear.weight, **args)
        )

    for name, mod in model.named_modules():
        pref = '.'.join(name.split('.')[:-1])
        if pref in visited_prefixes:
            continue

        if isinstance(mod, Attention) or name.endswith('.attn'):
            parametrize.register_parametrization(
                mod.qkv,
                'weight',
                _AttnSNReweight(mod.qkv.weight, renorm_values=renorm_values, **args),
            )
            if hasattr(mod, 'proj') and renorm_proj:
                parametrize_linear(mod.proj)
            visited_prefixes.add(name)
        elif name.endswith('mlp') and renorm_mlp and hasattr(mod, 'w12'):
            parametrize.register_parametrization(
                mod.w12,
                'weight',
                _ChunkedSNReweight(mod.w12.weight, num_chunks=2, **args),
            )
            parametrize_linear(mod.w3)
            visited_prefixes.add(name)
        elif isinstance(mod, nn.Linear) and 'patch_generator' not in name:
            parametrize_linear(mod)
        # elif isinstance(mod, nn.Conv2d):
        #     parametrize_linear(mod)


def configure_spectral_reparam_from_args(model: nn.Module, args):
    spectral_reparam = getattr(args, 'spectral_reparam', False)
    if isinstance(spectral_reparam, bool) and spectral_reparam:
        enable_spectral_reparam(model, init_norm_to_current=True)
    elif isinstance(spectral_reparam, dict):
        enable_spectral_reparam(model, init_norm_to_current=True, **spectral_reparam)


def disable_parametrizations(model: nn.Module):
    print('Disabling Parametrizations')
    for name, mod in model.named_modules():
        if parametrize.is_parametrized(mod):
            parametrize.remove_parametrizations(mod, 'weight')
            pass


@contextmanager
def allow_proj_reparam(allow: bool = False):
    global _PROJ_REPARAM

    prev = _PROJ_REPARAM
    _PROJ_REPARAM = allow

    yield

    _PROJ_REPARAM = prev
