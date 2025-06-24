# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import torch


class Quad:
    def __init__(self, image: torch.Tensor):
        self.bounds = torch.tensor([
            [0, 0],
            [image.shape[-1], 0],
            [image.shape[-1], image.shape[-2]],
            [0, image.shape[-2]],
        ], dtype=torch.float32)

    def apply_stm(self, stm: torch.Tensor, **kwargs):
        '''
        Applies a homogenous transformation matrix
        '''
        self.bounds = _apply_single_stm(self.bounds, stm)

    def translate(self, delta_vector: torch.Tensor):
        self.bounds += delta_vector

    def scale(self, scale_vector: torch.Tensor, **kwargs):
        self.bounds *= scale_vector

    def rotate(self, rot_mat: torch.Tensor):
        self.bounds = self.bounds @ rot_mat.T

    def flip(self, x: float):
        self.bounds[:, 0] -= x
        self.bounds[:, 0] *= -1
        self.bounds[:, 0] += x


def _apply_single_stm(vertices: torch.Tensor, stm: torch.Tensor):
    """
    Applies the single homogeneous transformation matrix.

    Args:
        vertices (torch tensor): Array of 2D vertices that form the polyline.
        stm (torch.tensor): 3x3 homogeneous matrix.
    """
    homogenous_vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim=1)
    transformed = torch.matmul(homogenous_vertices, stm)
    norm_factor = 1.0 / transformed[:, 2:]
    # Handle divide by zero case.
    norm_factor[transformed[:, 2:] == 0] = 0
    return transformed[:, :2].contiguous() * norm_factor
