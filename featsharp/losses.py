# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from torch.nn import functional as F

from .util import get_rank
from .metrics.mmd_loss import MMDLoss


def total_variation(img):
    b, c, h, w = img.size()
    return ((img[:, :, 1:, :] - img[:, :, :-1, :]).square().sum() +
            (img[:, :, :, 1:] - img[:, :, :, :-1]).square().sum()) / (b * c * h * w)


Distance_T = Callable[..., torch.Tensor]
DEFAULT_DISTANCE = F.smooth_l1_loss


class SampledCRFLoss(torch.nn.Module):
    def __init__(self, n_samples: int, alpha: float, beta: float, gamma: float, w1: float, w2: float, shift: float,
                 distance_fn: Distance_T = DEFAULT_DISTANCE):
        super(SampledCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift
        self.randgen = torch.Generator('cuda')
        self.randgen.manual_seed(42 + get_rank())
        self.distance_fn = distance_fn


    def forward(self, guidance: torch.Tensor, features: torch.Tensor, valid_mask: torch.Tensor, loss_scales: torch.Tensor):
        device = features.device
        assert (guidance.shape[0] == features.shape[0])
        assert (guidance.shape[2:] == features.shape[2:])
        b = guidance.shape[0]
        h = guidance.shape[2]
        w = guidance.shape[3]

        flat_valid_mask = valid_mask.flatten(1)
        if self.n_samples < flat_valid_mask.shape[1]:
            sample_offsets = torch.multinomial(flat_valid_mask + 1e-8, self.n_samples, replacement=False, generator=self.randgen)
        else:
            sample_offsets = (torch.arange(0, flat_valid_mask.shape[1], dtype=torch.int64, device=device)
                                   .unsqueeze(0)
                                   .expand_as(flat_valid_mask))

        n_samples = sample_offsets.shape[1]

        y_coords = torch.floor_divide(sample_offsets, valid_mask.shape[-1])
        x_coords = sample_offsets % valid_mask.shape[-1]
        coords = torch.stack([y_coords, x_coords], dim=1)

        norm_coords = coords / torch.tensor([h, w], device=guidance.device, dtype=features.dtype).unsqueeze(-1)

        selected_valid = torch.gather(flat_valid_mask, dim=1, index=sample_offsets)
        selected_feats = torch.gather(features.flatten(2), dim=2, index=sample_offsets.unsqueeze(1).expand(*features.shape[:2], -1))
        selected_guidance = torch.gather(guidance.flatten(2), dim=2, index=sample_offsets.unsqueeze(1).expand(*guidance.shape[:2], -1))
        selected_loss_scales = torch.gather(loss_scales.flatten(1), dim=1, index=sample_offsets)

        coord_diff = torch.cdist(norm_coords.mT, norm_coords.mT, compute_mode='donot_use_mm_for_euclid_dist').pow(2)
        guidance_diff = torch.cdist(selected_guidance.mT, selected_guidance.mT, compute_mode='donot_use_mm_for_euclid_dist').pow(2)

        feat_diff = self.distance_fn(
            selected_feats.unsqueeze(-1).expand(-1, -1, -1, n_samples),
            selected_feats.unsqueeze(-2).expand(-1, -1, n_samples, -1),
            reduction='none'
        ).mean(dim=1)

        sim_kernel = self.w1 * torch.exp(-coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(-coord_diff / (2 * self.gamma)) - self.shift
        sim_kernel /= (self.w1 + self.w2)

        valid_diff = selected_valid.unsqueeze(2) * selected_valid.unsqueeze(1)
        unc_diff = (selected_loss_scales.unsqueeze(2) * selected_loss_scales.unsqueeze(1)).clamp_min(1e-8).sqrt()  # Geometric mean uncertainty

        diff_prod = valid_diff * feat_diff * sim_kernel
        div = valid_diff.sum().clamp_min_(valid_diff.numel() / 2)

        loss = (unc_diff * diff_prod).sum() / div
        raw_loss = diff_prod.detach().sum() / div
        return loss, raw_loss

    def normalize_features(self, features: torch.Tensor):
        std, mean = torch.std_mean(features.detach(), dim=(0, 2, 3), keepdim=True)

        return (features - mean) / std

class TVLoss(nn.Module):
    def __init__(self, distance_fn: Distance_T = DEFAULT_DISTANCE):
        super().__init__()
        self.distance_fn = distance_fn

    def forward(self, img: torch.Tensor, valid_mask: torch.Tensor, unc_scale: torch.Tensor):
        valid_mask = valid_mask.expand_as(img)
        unc_scale = unc_scale.expand_as(img)

        def up(t: torch.Tensor, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
            return op(t[:, :, 1:], t[:, :, :-1])

        def left(t: torch.Tensor, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
            return op(t[:, :, :, 1:], t[:, :, :, :-1])

        up_diff = up(img, partial(F.smooth_l1_loss, reduction='none'))
        up_valid = up(valid_mask, lambda a, b: a * b)
        up_unc_scale = up(unc_scale, lambda a, b: a * b).sqrt()  # Geometric mean uncertainty

        left_diff = left(img, partial(F.smooth_l1_loss, reduction='none'))
        left_valid = left(valid_mask, lambda a, b: a * b)
        left_unc_scale = left(unc_scale, lambda a, b: a * b).sqrt()

        up_diff = up_diff * up_valid
        up_div = up_valid.sum().clamp_min_(up_valid.numel() / 2)

        left_diff = left_diff * left_valid
        left_div = left_valid.sum().clamp_min_(left_valid.numel() / 2)

        up_loss = (up_unc_scale * up_diff).sum() / up_div
        left_loss = (left_unc_scale * left_diff).sum() / left_div

        up_raw_loss = up_diff.detach().sum() / up_div
        left_raw_loss = left_diff.detach().sum() / left_div

        loss = up_loss + left_loss
        raw_loss = up_raw_loss + left_raw_loss

        return loss, raw_loss


def apply_weight(weight, loss):
    if weight == 0 and torch.is_tensor(loss):
        loss = loss.detach()
    return weight * loss
