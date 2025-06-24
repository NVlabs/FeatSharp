# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import torch


BASE_GRID_CACHE = dict()
def generate_homography_grid(homography: torch.Tensor, size):
    global BASE_GRID_CACHE

    N, C, H, W = size
    if size not in BASE_GRID_CACHE:
        base_grid = homography.new(N, H, W, 3)
        linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
        base_grid[:, :, :, 2] = 1
        BASE_GRID_CACHE[size] = base_grid
    else:
        base_grid = BASE_GRID_CACHE[size]

    grid = torch.bmm(base_grid.view(N, H * W, 3), homography.transpose(1, 2))
    grid = grid.view(N, H, W, 3)
    grid[:, :, :, 0] = grid[:, :, :, 0] / grid[:, :, :, 2]
    grid[:, :, :, 1] = grid[:, :, :, 1] / grid[:, :, :, 2]
    grid = grid[:, :, :, :2].float()
    return grid
