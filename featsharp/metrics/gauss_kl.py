# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import torch
from torch.nn import functional as F
from einops import rearrange

def gauss_kl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = rearrange(x, 'b c h w -> (b h w) c')
    if y.ndim == 4:
        y = rearrange(y, 'b c h w -> (b h w) c')
    x = x.double()
    y = y.double()

    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)

    cov_x = torch.cov(x.T)
    cov_y = torch.cov(y.T)
    eps = 1e-5
    cov_x = cov_x + eps * torch.eye(cov_x.shape[0], device=cov_x.device)
    cov_y = cov_y + eps * torch.eye(cov_y.shape[0], device=cov_y.device)

    def safe_det(t: torch.Tensor):
        e = torch.linalg.eigvalsh(t)
        e = e.clamp_min(1e-5)
        d = torch.prod(e)
        return d

    det_x = safe_det(cov_x)
    det_y = safe_det(cov_y)

    if (torch.any(det_x <= 1e-5) or torch.any(det_y <= 1e-5)) and x.shape[1] > 1:
        return _singular_gauss_kl(x, y, cov_x)

    inv_cov_y: torch.Tensor = torch.linalg.pinv(cov_y)

    tr_a: torch.Tensor = inv_cov_y @ cov_x
    tr_a = tr_a.trace()

    qp_diff = mu_y - mu_x
    b = torch.mv(inv_cov_y, qp_diff)
    b = torch.dot(qp_diff, b)

    ln_div = det_y.log() - det_x.log()

    d = x.shape[1]

    kl = 0.5 * (tr_a + b - d + ln_div)
    return kl


def _singular_gauss_kl(x: torch.Tensor, y: torch.Tensor, cov_x: torch.Tensor) -> torch.Tensor:
    e, L = torch.linalg.eigh(cov_x)

    L_best = L[:, -L.shape[1] // 2:]

    mu_x = x.mean(dim=0, keepdim=True)
    mu_y = y.mean(dim=0, keepdim=True)

    x = (x - mu_x) @ L_best
    y = (y - mu_y) @ L_best

    mu_x = mu_x @ L_best
    mu_y = mu_y @ L_best

    x = x + mu_x
    y = y + mu_y

    return gauss_kl(x, y)
