# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import Literal

import torch
from torch import nn
from torch import distributed as dist
from torch.distributed.nn.functional import all_gather, all_reduce
from einops import rearrange

from ..util import get_rank, get_world_size
from .mmd_cuda.mmd_extension import mmd as fast_mmd, brute_force_mmd

mmd = brute_force_mmd
# mmd = fast_mmd

# class MMDLoss(nn.Module):
#     def __init__(self, max_samples: int = 500, gamma: float = None):
#         super().__init__()
#         self.max_samples = max_samples
#         self.gamma = gamma
#         self.rank = get_rank()
#         self.world_size = get_world_size()

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         if x.ndim == 4:
#             x = rearrange(x, 'b c h w -> (b h w) c')
#         if y.ndim == 4:
#             y = rearrange(y, 'b c h w -> (b h w) c')

#         x = self._limit_samples(x)
#         y = self._limit_samples(y)

#         if self.gamma is None:
#             gamma = self._find_gamma(x)
#         else:
#             gamma = self.gamma

#         mmd_xx = self._calc_distributed_mmd(x, x, gamma)
#         mmd_yy = self._calc_distributed_mmd(y, y, gamma)
#         mmd_xy = self._calc_distributed_mmd(x, y, gamma)

#         loss = mmd_xx + mmd_yy - 2 * mmd_xy
#         return loss

#     @torch.no_grad()
#     def _find_gamma(self, x: torch.Tensor):
#         xx = torch.cdist(x, x, p=2).pow_(2)

#         med = torch.median(xx.flatten())
#         gamma = 1.0 / (2 * med)

#         if self.world_size > 1:
#             dist.all_reduce(gamma, op=dist.ReduceOp.AVG)

#         return gamma

#     def _limit_samples(self, t: torch.Tensor):
#         if t.shape[0] <= self.max_samples:
#             return t

#         weights = torch.ones(t.shape[0], dtype=torch.float32, device=t.device)
#         sample_idxs = torch.multinomial(weights, self.max_samples, replacement=False)
#         return t[sample_idxs]

#     def _calc_distributed_mmd(self, a: torch.Tensor, b: torch.Tensor, gamma: torch.Tensor | float):
#         if self.world_size > 1:
#             b = torch.cat(all_gather(b), dim=0)

#         mmd_ab = torch.cdist(a, b, p=2).pow(2).mul(-gamma).exp()
#         mmd_ab = mmd_ab.mean()

#         if self.world_size > 1:
#             mmd_ab = all_reduce(mmd_ab, op=dist.ReduceOp.AVG)

#         return mmd_ab


class MMDLoss(nn.Module):
    def __init__(self, upsample_factor: int, max_samples: int = 4000, gamma: torch.Tensor | float | None = None):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.max_samples = max_samples
        self.gamma = gamma
        self.rank = get_rank()
        self.world_size = get_world_size()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        _, _, H2, W2 = y.shape
        _, C, H1, W1 = x.shape

        x = rearrange(x, 'b c (h ch) (w cw) -> (b h w) (ch cw) c', h=H1, ch=1, w=W1, cw=1)
        y = rearrange(y, 'b c (h ch) (w cw) -> (b h w) (ch cw) c', h=H1, ch=self.upsample_factor, w=W1, cw=self.upsample_factor)

        dist_xy = torch.cdist(x.detach(), y.detach(), p=2)

        min_y_idx = dist_xy.argmax(dim=2)
        min_y = y.gather(dim=1, index=min_y_idx.unsqueeze(-1).expand(-1, -1, C))

        x = x.squeeze(1)
        y = min_y.squeeze(1)

        if self.world_size > 1:
            x = torch.cat(all_gather(x), dim=0)
            y = torch.cat(all_gather(y), dim=0)

        x, sample_idxs = self._limit_samples(x)
        y, _ = self._limit_samples(y, sample_idxs)

        mmd_loss = mmd(x, y, self.gamma)

        return mmd_loss

    def _limit_samples(self, t: torch.Tensor, sample_idxs: torch.Tensor | None = None):
        if t.shape[0] <= self.max_samples:
            return t, None

        if sample_idxs is None:
            weights = torch.ones(t.shape[0], dtype=torch.float32, device=t.device)
            sample_idxs = torch.multinomial(weights, self.max_samples, replacement=False)
        return t[sample_idxs], sample_idxs
