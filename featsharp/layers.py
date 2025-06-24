# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import math
import os
from tqdm import tqdm

from einops import rearrange
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .util import extract_normalize

from .upsampler_modules.hadamard import get_hadamard_matrix
from .upsampler_modules.util import ChannelNorm


class PHI_Standardize(nn.Module):
    def __init__(self, dim, enable_rotation: bool = True, enable_alpha: bool = True):
        super().__init__()

        self.dim = dim
        self.enable_rotation = enable_rotation
        self.enable_alpha = enable_alpha

        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float32), persistent=True)
        self.register_buffer('alpha', torch.tensor(0.0, dtype=torch.float32), persistent=True)
        self.register_buffer('rotation', torch.zeros(dim, dim, dtype=torch.float32), persistent=True)
        self.register_buffer('inv_rotation', torch.zeros_like(self.rotation), persistent=True)

        self.initialized = False

    def forward(self, x: torch.Tensor):
        x = x - self.mean.reshape(1, -1, 1, 1)

        if self.enable_rotation:
            kernel = self.rotation[:, :, None, None]
            if self.enable_alpha:
                kernel = kernel * self.alpha

            x = F.conv2d(x, weight=kernel, bias=None, stride=1, padding=0)
        elif self.enable_alpha:
            x = x * self.alpha

        return x

    def denormalize(self, x: torch.Tensor):
        if self.enable_rotation:
            kernel = self.inv_rotation[:, :, None, None]
            if self.enable_alpha:
                kernel = kernel / self.alpha

            x = F.conv2d(x, weight=kernel, bias=self.mean, stride=1, padding=0)
        else:
            if self.enable_alpha:
                x = x / self.alpha

            x = x + self.mean.reshape(1, -1, 1, 1)

        return x

    def get_log_stats(self):
        return dict(phi_s_alpha=self.alpha.item())

    def _get_cache_file(self, model, cfg):
        model_name = cfg.model_type
        detail = getattr(model, 'detail', '')
        if detail:
            model_name += f'_{detail}'

        input_size = getattr(model, 'input_size', cfg.input_size)

        cache_file = os.path.join(cfg.output_root, 'phi-s', f'{model_name}_{input_size}.pth')
        return cache_file

    def requires_loader(self, model, cfg):
        cache_file = self._get_cache_file(model, cfg)
        return not self.initialized and not os.path.exists(cache_file)

    def initialize(self, model, loader, cfg):
        if self.initialized:
            return

        cache_file = self._get_cache_file(model, cfg)
        if os.path.exists(cache_file):
            self._load_from_cache(cache_file)
        else:
            self._train(model, loader, cfg)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self._save_to_cache(cache_file)

        self.initialized = True

    @torch.no_grad()
    def _train(self, model, loader, cfg):
        mean, cov = self._get_mean_cov(model, loader, cfg)

        L, V = torch.linalg.eigh(cov)

        L.clamp_min_(0)

        inv_normalizer = L.mean().sqrt()
        self.alpha.copy_(1 / inv_normalizer)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f'PHI-S - Normalization factor: {self.alpha.item():.5f}')

        # eye = torch.eye(L.shape[0], dtype=L.dtype, device=L.device)
        # normalizer = eye * normalizer
        # inv_normalizer = eye * inv_normalizer

        H = get_hadamard_matrix(self.dim, allow_approx=False).to(L)
        # normalize = H @ normalizer @ V.T
        # inv_normalize = V @ inv_normalizer @ H.T

        self.mean.copy_(mean)
        self.rotation.copy_(H @ V.T)
        self.inv_rotation.copy_(V @ H.T)

    @torch.no_grad()
    def _get_mean_cov(self, model, loader, cfg):
        model.cuda().eval()

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        num_batches = cfg.get('normalizer_args', dict()).get('num_batches', 3000)

        mean = torch.zeros(self.dim, dtype=torch.float64, device='cuda')
        M2 = torch.zeros(self.dim, self.dim, dtype=torch.float64, device='cuda')

        total_samples = 0

        # sanity = []
        gather_buff = None

        input_cond = getattr(model, 'input_conditioner', None)
        if input_cond is not None:
            input_cond = extract_normalize(input_cond).cuda()

        for i, batch in tqdm(enumerate(loader), desc='PHI-S', total=num_batches, disable=rank > 0, leave=True):
            if i == num_batches:
                break

            img: torch.Tensor = batch[0][0].cuda()

            img = F.interpolate(img, (cfg.input_size, cfg.input_size), mode='bilinear', align_corners=False)

            if input_cond is not None:
                img = input_cond(img)

            features: torch.Tensor = model(img)
            flat_feats = rearrange(features, 'b c h w -> (b h w) c')

            if world_size > 1:
                if gather_buff is None:
                    gather_buff = torch.empty(world_size * flat_feats.shape[0], flat_feats.shape[1], dtype=flat_feats.dtype, device=flat_feats.device)

                dist.all_gather_into_tensor(gather_buff, flat_feats)
                flat_feats = gather_buff
            flat_feats = flat_feats.double()

            # sanity.append(flat_feats)

            k = flat_feats.shape[0]
            total_samples += k
            new_mean = flat_feats.mean(dim=0)
            delta = new_mean - mean
            mean += (k / total_samples) * delta

            # Update the covariance matrix accumulator
            chunk_centered = flat_feats - new_mean
            M2 += chunk_centered.T @ chunk_centered + (delta[:, None] * delta[None, :] * (k * (total_samples - k) / total_samples))

        cov = M2 / (total_samples - 1)

        # sanity = torch.cat(sanity)
        # sanity_mean = sanity.mean(dim=0)
        # sanity_cov = torch.cov(sanity.T)

        return mean, cov

    def _load_from_cache(self, cache_path: str):
        sd = torch.load(cache_path, map_location='cpu')
        self.load_state_dict(sd)
        print(f'Loaded PHI-S parameters from "{cache_path}"')

    def _save_to_cache(self, cache_path: str):
        sd = self.state_dict()

        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        torch.save(sd, cache_path)
        print(f'Saved PHI-S parameters to "{cache_path}"')


def get_normalizer(normalizer_mode: str, dim: int):
    if normalizer_mode == 'phi-s':
        normalizer = PHI_Standardize(dim)
    elif normalizer_mode == 'phi-s-noalpha':
        normalizer = PHI_Standardize(dim, enable_alpha=False)
    elif normalizer_mode == 'phi-s-onlyalpha':
        normalizer = PHI_Standardize(dim, enable_rotation=False, enable_alpha=True)
    elif normalizer_mode == 'layernorm':
        normalizer = ChannelNorm(dim)
    elif not normalizer_mode or normalizer_mode in ('identity', 'none'):
        normalizer = nn.Identity()
    else:
        raise ValueError(f'Unsupported normalizer: {normalizer_mode}')
    return normalizer


class BiasBuffer(nn.Module):
    def __init__(self, dim: int, res: int):
        super().__init__()

        self.buffer = nn.Parameter(torch.zeros(1, dim, res, res))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        buffer = self.buffer
        if buffer.shape[-2:] != x.shape[-2:]:
            buffer = F.interpolate(buffer, x.shape[-2:], mode='bilinear', align_corners=False)

        return x + buffer
