# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
"""
Defer image model.

This provides a wrapper around a torch tensor that can efficiently move through
the data loading, transformation and augmentation pipeline.
"""
from typing import List

import logging

import torch
import torch.cuda
from data.transforms.quad import _apply_single_stm

from ..utils import options

from .spatial_transform.spatial_transform_extension import spatial_transform

logger = logging.getLogger(__name__)

# NOTE: Many of the operations in this module may _seem_ like they're the inverse of what's
# being requested, such as translations moving in the opposite direction, or scales being
# the inverse. This is because `apply_spatial_transform` appears to be operating in the inverse
# manner. The best mental model I can think of is if you imagine that there's a window
# sitting on top of your image. Instead of your operations manipulating the image, they're
# manipulating the window. So the act of moving the window to the right has the identical
# inverse effect of moving the image to the left.


class DeferImage(object):
    """Wrapper around a torch tensor that provides support for spatial transforms."""

    def __init__(self, image: torch.Tensor, allow_eager=True):
        """
        Initializer.

        The image can be either a single image, or a batch of images.

        Args:
            image (torch.Tensor [(B)CHW]): The image(s) to wrap.
        """
        super().__init__()
        self.image = image
        self.target_size = tuple(image.shape)
        self.stm = torch.eye(3, dtype=torch.float32)
        self.fresh = allow_eager
        self.original_shape = image.shape

    @property
    def shape(self):
        """Returns the destination shape of the image after transforms are applied."""
        return self.target_size

    @property
    def dtype(self):
        return self.image.dtype

    def can_collate(self, other_defer):
        """Decides whether two `DeferImage` objects can be combined into one."""
        if not isinstance(other_defer, DeferImage):
            return False

        return (self.image.shape[-3:] == other_defer.image.shape[-3:] and
                self.target_size[-3:] == other_defer.target_size[-3:] and
                self.image.dtype == other_defer.image.dtype)

    def collate(self, other_defers, output_buffers=None):
        """
        Combines one or more other `DeferImage` objects into this one.

        If `output_buffers` is supplied, then all of the memory concatenation will be performed
        inside the provided buffers. The buffers must be a dict with the following format:
        {
            'image': (torch.Storage, list<output_offset>),
            'stm': (torch.Storage, list<output_offset>),
        }

        NOTE: `list<output_offset>` must be a list with a single element that defines the starting
              position to write data into the provided tensor. After the data is written, the offset
              will be updated so that consecutive calls may be performed.

        Args:
            other_defers (list<DeferImage>): List of other defers to combine.
                                             NOTE: It is assumed that `can_collate` has returned
                                             `True` for every item in this list.
            output_buffers (dict): See notes.
        """
        images = [self.image]
        stms = [self.stm]

        for df in other_defers:
            images.append(df.image)
            stms.append(df.stm)

        if output_buffers is None:
            self.image = torch.stack(images, dim=0)
            self.stm = torch.stack(stms, dim=0)
        else:
            for k, group in (('image', images), ('stm', stms)):
                output_buffer, output_offset = output_buffers[k]
                req_size = len(group) * group[0].nelement()
                if output_buffer.size() < output_offset[0] + req_size:
                    raise ValueError("The provided buffer for {} isn't large enough!")

                # This will create a new tensor (with no memory allocation),
                # and set_ will change the stored buffer. This allows us to share
                # the underlying storage, which itself should be shared!
                op_tensor = group[0].new()
                op_tensor.set_(output_buffer, output_offset[0], (len(group), *group[0].shape))

                torch.stack(group, dim=0, out=op_tensor)
                setattr(self, k, op_tensor)
                output_offset[0] += req_size

    def pin_memory(self):
        """Pins the memory for all owned tensors."""
        self.image = self.image.pin_memory()
        self.stm = self.stm.pin_memory()
        return self

    def cuda(self, **kwargs):
        """Moves all tensors to the default cuda device."""
        self.image = self.image.cuda(**kwargs)
        self.stm = self.stm.cuda(**kwargs)
        return self

    def __call__(self, pad_color: float = 0.0):
        """
        Applies all of the deferred operations, and returns the result.

        NOTE: This operation does not cache the result.
        """
        if self.fresh:
            return self.image

        out_height, out_width = self.target_size[-2:]

        image = self.image
        stm = self.stm.T
        was_3d = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            stm = stm.unsqueeze(0)
            was_3d = True

        if image.dtype == torch.uint8:
            image = image.float().div_(255)

        # TODO(mranzinger): Perhaps we should make `apply_spatial_transform`
        # support non-contiguous arrays?
        image = image.contiguous()
        stm = stm.contiguous()

        image = spatial_transform(image, stm, out_width, out_height,
                                  'bicubic', pad_color, False)

        if was_3d:
            image = image[0]

        return image

    def apply_stm(self, stm):
        """
        Applies the homogeneous transformation matrix.

        Args:
            stm (np.array): 3x3 numpy homogeneous matrix.
        """
        self.stm = self.stm @ stm
        self.fresh = False

    def clip_translate(self, x, y, width, height):
        """
        Clips the image at the specified coordinates.

        Args:
            min_x (float): The minimum bounding x coordinate.
            max_x (float): The maximum bounding x coordinate.
            min_y (float): The minimum bounding y coordinate.
            max_y (float): The maximum bounding y coordinate.
        """
        stm = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [x, y, 1],
        ], dtype=torch.float32)

        self.apply_stm(stm.T)
        self.target_size = self.target_size[:-2] + (height, width)

    def scale(self, scale_vector, retain_dims=False):
        """
        Scales the image by the given 2d size vector.

        Format: [x_scale, y_scale]

        If retain_dims is set to true, then the size of the resulting image is not updated.

        Args:
            scale_vector (torch.Tensor): The scale config.
            retain_dims (bool): Whether to update the image size or not.
        """
        stm = torch.tensor([
            [1.0 / scale_vector[0], 0, 0],
            [0, 1.0 / scale_vector[1], 0],
            [0, 0, 1],
        ], dtype=torch.float32)

        self.apply_stm(stm)

        if not retain_dims:
            target_height = self.target_size[-2] * scale_vector[-1].item()
            target_width = self.target_size[-1] * scale_vector[-2].item()
            self.target_size = self.target_size[:-2] + (target_height, target_width)

    def translate(self, trans_vector):
        """
        Translates the image by the supplied translation vector.

        Format: [x_translation, y_translation]

        Args:
            trans_vector (torch.Tensor): The translation vector.
        """
        stm = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [-trans_vector[0], -trans_vector[1], 1],
        ], dtype=torch.float32)

        self.apply_stm(stm.T)

    def flip(self, x):
        """
        Flips the image horizontally about the specified x-axis.

        Args:
            x (float): The x-axis to flip about.
        """
        self.apply_stm(torch.tensor([
            [   -1, 0, 0],  # noqa
            [    0, 1, 0],  # noqa
            [2 * x, 0, 1]
        ], dtype=torch.float32).T)

    def rotate(self, rot_mat):
        stm = torch.eye(3, **options(rot_mat))
        stm[:2, :2].copy_(rot_mat)

        self.apply_stm(stm.T)

        h2, w2 = self.target_size[-2] / 2, self.target_size[-1] / 2
        bds = torch.tensor([
            [-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]
        ], dtype=torch.float32)

        bds = _apply_single_stm(bds, stm)

        min_bds = bds.amin(dim=0)
        max_bds = bds.amax(dim=0)

        new_width = (max_bds[0] - min_bds[0]).item()
        new_height = (max_bds[1] - min_bds[1]).item()

        self.target_size = self.target_size[:-2] + (new_height, new_width)

    def get_bds(self):
        bds = torch.tensor([
            [0, 0],
            [self.original_shape[-1], 0],
            [self.original_shape[-1], self.original_shape[-2]],
            [0, self.original_shape[-2]]
        ], dtype=torch.float32)

        bds = _apply_single_stm(bds, torch.inverse(self.stm).T)

        return bds


def get_cuda_images(defer_image_groups, augmentation=None, additional=[]):
    """
    Converts the image groups (on CPU) to a contiguous GPU batch.

    It is expected that there are one or more "image groups", represented as a list of
    `DeferImage` objects. Each of these objects may have deferred spatial operations
    that need to be performed on the GPU.

    Operations Performed:
        1) Move all image groups to GPU
        2) Apply pending spatial operations
        3) Concatenate all groups into a single batch of images (on GPU)
        4) Convert images to float32
            NOTE: If dtype==uint8, then values are scaled by 1/255
    """
    images = []
    for i, defer in enumerate(defer_image_groups):
        defer.cuda(non_blocking=True)
        images.append(defer())

    cuda_images = torch.cat(images, dim=0)

    # Convert to floating point
    if cuda_images.dtype == torch.float16:
        cuda_images = cuda_images.to(torch.float32)
    elif cuda_images.dtype == torch.uint8:
        cuda_images = cuda_images.float().div_(255.0)

    # This is the mean color that most pretrained models expect. In particular, they expect pre-whitened inputs,
    # which involves subtracting this value and then dividing by std. So, if we set the invalid values to be the mean
    # value, then they'll be subtracted at input, thus yielding ~0 values for invalid positions
    invalid_color = torch.tensor([0.485, 0.456, 0.406], **options(cuda_images)).reshape(1, -1, 1, 1).expand_as(cuda_images)
    # Use this mask to prevent color information from leaking into the invalid regions
    invalid_mask = torch.any(cuda_images < 0, dim=1, keepdim=True).expand_as(cuda_images)

    if augmentation is not None:
        cuda_images: torch.Tensor = augmentation(cuda_images, *additional)

    cuda_images = torch.where(invalid_mask, invalid_color, cuda_images)

    return cuda_images
