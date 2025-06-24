# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import List, Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .generate_homography_grid import generate_homography_grid


def _draw_grid(img: torch.Tensor, spacing: int):
    for i in range(0, img.shape[-2], spacing):
        img[0, i:i+1].fill_(1)
    for i in range(0, img.shape[-1], spacing):
        img[0, ..., i:i+1].fill_(1)


def equivariant_collate(num_images: int, debug: bool = False, patch_sizes: List[int] = None):
    sample_bounds = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)

    def _quad_from_shape(shape):
        H, W = shape[-2:]
        return np.array([
            [0, 0], [W, 0], [W, H], [0, H],
        ], dtype=np.float32)

    def _stage(batch: List[List[Any]]):
        groups = [[] for _ in batch[0]]
        valid_masks = [[] for _ in range(num_images)]
        transforms = [[] for _ in range(num_images - 1)]
        overlay_students = [[] for _ in range(num_images - 1)]
        overlay_teachers = [[] for _ in range(num_images - 1)]
        for ex in batch:
            for i in range(len(ex)):
                if i < num_images:
                    im, bds = ex[i]
                    orig_bounds = _quad_from_shape(im.original_shape)
                    im = im()

                    if debug:
                        _draw_grid(im, patch_sizes[i])

                    teacher_crop_bounds = _quad_from_shape(im.shape)

                    valid_mask = torch.zeros_like(im[0])
                    cv2.fillPoly(valid_mask.numpy(), pts=[bds.bounds.round().int().numpy()], color=(1,))

                    groups[i].append(im)
                    valid_masks[i].append(valid_mask)

                    if i > 0:
                        # Get the transformation that projects the student image to the teacher image
                        student_tx_bounds = ex[0][1].bounds.numpy()

                        student_crop_bounds = _quad_from_shape(groups[0][-1].shape)

                        teacher_tx_bounds = bds.bounds.numpy()

                        # [-1, 1] -> [0, tH|tW]  (From normalized teacher coordinates to absolute cropped teacher coordinates)
                        tx_sample_to_teacher_crop = cv2.getPerspectiveTransform(sample_bounds, teacher_crop_bounds)

                        tx_student_to_teacher = cv2.getPerspectiveTransform(teacher_tx_bounds, student_tx_bounds)

                        # [0, sH|sW] -> [-1, 1]  (From absolute cropped student coordiantes to normalized student coordinates)
                        tx_crop_student_to_sample = cv2.getPerspectiveTransform(student_crop_bounds, sample_bounds)

                        # Now we start left multiplying these transforms to condense into a single teacher to student transform
                        tx_teacher_sample_to_student_crop = np.matmul(tx_student_to_teacher, tx_sample_to_teacher_crop)

                        # Finally, the transform that takes us from [-1, 1] in the teacher space to [-1, 1] in the student space
                        tx_teacher_sample_to_student_sample = np.matmul(tx_crop_student_to_sample, tx_teacher_sample_to_student_crop)

                        transform_mat = torch.from_numpy(tx_teacher_sample_to_student_sample).float()
                        transforms[i - 1].append(transform_mat)

                        # This allows visualizing the student and teacher images
                        if debug:
                            grid = generate_homography_grid(transform_mat[None], im[None].shape)
                            im_s_to_t = F.grid_sample(groups[0][-1][None], grid, mode='bicubic', align_corners=True)[0]

                            # viz_t = torch.cat([im, im_s_to_t], dim=2)
                            viz_t = (im + im_s_to_t) / 2
                            _draw_grid(viz_t, patch_sizes[i])
                            overlay_teachers[i - 1].append(viz_t)

                            grid_t_to_s = generate_homography_grid(torch.linalg.inv(transform_mat[None]), groups[0][-1][None].shape)
                            im_t_to_s = F.grid_sample(groups[i][-1][None], grid_t_to_s, mode='bicubic', align_corners=True)[0]

                            viz_t = (groups[0][-1] + im_t_to_s) / 2
                            _draw_grid(viz_t, patch_sizes[0])
                            overlay_students[i - 1].append(viz_t)
                            pass

                else:
                    groups[i].append(ex[i])
        for g in range(len(groups)):
            if g < num_images:
                groups[g] = torch.stack(groups[g])
            else:
                groups[g] = torch.tensor(groups[g])
        for vm in range(len(valid_masks)):
            valid_masks[vm] = torch.stack(valid_masks[vm])
        for t in range(len(transforms)):
            transforms[t] = torch.stack(transforms[t])

        for g in range(num_images):
            grp = (groups[g], valid_masks[g])
            if g > 0:
                grp = (*grp, transforms[g-1])
            groups[g] = grp

        if debug:
            for g in range(len(overlay_teachers)):
                overlay_students[g] = torch.stack(overlay_students[g])
                overlay_teachers[g] = torch.stack(overlay_teachers[g])

            _save_img('student.jpg', groups[0][0].permute(1, 2, 0, 3).flatten(2))
            for i in range(1, num_images):
                _save_img(f'teacher_{i-1}.jpg', groups[i][0].permute(1, 2, 0, 3).flatten(2))
                _save_img(f'student_{i-1}_overlay.jpg', overlay_students[i-1].permute(1, 2, 0, 3).flatten(2))
                _save_img(f'teacher_{i-1}_overlay.jpg', overlay_teachers[i-1].permute(1, 2, 0, 3).flatten(2))
            pass
        return groups
    return _stage


def _save_img(path: str, tensor: torch.Tensor):
    cv2.imwrite(path, cv2.cvtColor(tensor.mul(255).permute(1, 2, 0).byte().numpy(), cv2.COLOR_RGB2BGR))
