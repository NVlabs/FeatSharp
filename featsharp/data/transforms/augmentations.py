# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
"""
Image augmentations module.

The difference, currently, between a transform and an augmentation is that augmentations are both
non-geometric, and are applied explicitly on batches of images on the GPU. Future implementations
could potentially combine transforms and augmentations into a single pipe.

NOTE: The usage of randomized augmentations is still allowed in a deterministic setting, if and only
      if the torch and numpy random seeds (`torch.manual_seed(seed)` and `numpy.random.seed(seed)`)
      are deterministically set *before* any of these modules are used.
"""

import logging
from typing import Iterable

import numpy
import torch
from torch.distributions import Bernoulli, Binomial, Normal, Uniform
import torch.nn.functional as F
from utils import options

logger = logging.getLogger(__name__)

_DIST_OPTS = { 'dtype': torch.float32, 'device': 'cuda' }

class AugmentationBase(object):
    """Base class for data augmentations. Enables batched processing of images."""

    def __init__(self, prob_apply=0.5):
        """
        Initializer.

        Args:
            prob_apply (float [0, 1]): The probability of applying the augmentation to a single
                                       image.
        """
        if prob_apply < 0 or prob_apply > 1:
            raise ValueError("`prob_apply` must be between 0 and 1 inclusive.")

        self.prob_apply = prob_apply
        self.num_image_distribution = None

    def __call__(self, images, *additional):
        """
        Applies the specified augmentation to the provided images.

        The number of images that the transform is applied to is stochastic, controlled by the
        `prob_apply` variable supplied in the initializer.

        WARNING: The augmentations are applied in-place, but the return value should be used
        to ensure correct results.

        Args:
            images (torch.Tensor [BCHW]): The batched image tensor to apply the augmentations.

        Returns:
            The augmented images. See warnings.
        """
        # We use a binomial distribution to select the number of images to which the given
        # augmentation should be applied. Since the augmentations operate on a batch of images,
        # the basic goal is to treat them as a contiguous block so that we can apply the aug
        # on all images at once.
        if self.num_image_distribution is None or self.num_image_distribution.total_count != images.shape[0]:
            self.num_image_distribution = Binomial(total_count=images.shape[0],
                                                   probs=torch.tensor([self.prob_apply]))

        num_images = int(self.num_image_distribution.sample().item())

        if num_images == 0:
            return images

        if num_images < images.shape[0]:
            offset = numpy.random.randint(0, images.shape[0] - num_images)
        else:
            offset = 0

        sub_images = images[offset:offset + num_images]

        applied_images = self.get_applied_images(sub_images, *additional)

        parts = [images[:offset], applied_images, images[offset + num_images:]]

        return torch.cat(parts, dim=0)

    def get_applied_images(self, images, *additional):
        """Perform the data augmentation on all supplied images."""
        raise NotImplementedError("Subclasses must implement this!")


class GaussianBlurAugmentation(AugmentationBase):
    """Applies gaussian blurring with the specified kernel size."""

    def __init__(self, kernel_size=3, **kwargs):
        """
        Initialization.

        Args:
            kernel_size (int): The size of the gaussian kernel.
            **args (See AugmentationBase)
        """
        super().__init__(**kwargs)

        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("The kernel size must be odd and >= 3.")

        self.kernel_size = kernel_size

        kernel = torch.zeros((kernel_size,), dtype=torch.float32)

        # Use pascal's triangle to simulate the gaussian kernel
        #     1
        #    1 1
        #   1 2 1   <--- kernel_size=3
        #  1 3 3 1
        # 1 4 6 4 1 <--- kernel_size=5
        kernel[0] = 1
        for _ in range(1, kernel_size):
            for i in range(kernel_size - 1, 0, -1):
                pv = kernel[i - 1] if i > 0 else 0
                v = kernel[i]
                kernel[i] = pv + v

        # Take the outer product of the 1d kernel to produce a 2d kernel
        # TODO(mranzinger): Is it faster to launch the 2d kernel, or apply 2x 1d kernels?
        kernel = kernel.reshape(kernel_size, 1)
        kernel = kernel @ kernel.t()

        # Normalize the filter
        kernel /= kernel.sum()

        self.weight = kernel.reshape(1, 1, kernel_size, kernel_size)
        self.padding = kernel_size // 2

    def get_applied_images(self, images, *additional):
        """
        Apply gaussian blurring to the specified images.

        Args:
            images (torch.Tensor [BCHW]): Batched images tensor.

        Returns:
            torch.Tensor with blurred images.
        """
        # Move the weights to the same device as `images`. No-op if they're already the same.
        self.weight = self.weight.to(images)

        # Combine the channel and batch dimensions.
        s_images = images.reshape(-1, 1, *images.shape[-2:])

        # Since we're combining the channel and batch dimensions, we want to disable
        # this flag so that kernel strategies aren't benchmarked everytime the size of this
        # changes.
        # TODO(mranzinger): Verify that perf is better this way.
        was_bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False

        blurred = F.conv2d(s_images, self.weight, padding=(self.padding, self.padding))

        torch.backends.cudnn.benchmark = was_bench

        return blurred.reshape(*images.shape)


class BlurAugmentation(AugmentationBase):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)

        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("The kernel size must be odd and >= 3!")

        self.kernel_size = kernel_size

        kernel = torch.full((kernel_size, kernel_size), 1 / (kernel_size ** 2), dtype=torch.float32)

        self.weight = kernel.reshape(1, 1, kernel_size, kernel_size)
        self.padding = kernel_size // 2

    def get_applied_images(self, images, *additional):
        self.weight = self.weight.to(images)

        # Combine the channel and batch dimensions since we want to treat each channel independently
        s_images = images.reshape(-1, 1, *images.shape[-2:])

        padded = F.pad(s_images, (self.padding,) * 4, mode='reflect')

        blurred = F.conv2d(padded, self.weight)

        return blurred.reshape(*images.shape)

class GrayscaleAugmentation(AugmentationBase):
    def get_applied_images(self, images, *additional):
        means = torch.mean(images, dim=1, keepdim=True)

        return means.expand_as(images)

class RandomNoiseAugmentation(AugmentationBase):
    """Applies random gaussian noise to each pixel in the image."""

    def __init__(self, std_dev, mean=0.0, **kwargs):
        """
        Initialization.

        Args:
            std_dev (float): The standard deviation of the normal distribution to pull samples from.
            mean (float): The mean of the normal distribution.
            **kwargs: See AugmentationBase
        """
        super().__init__(**kwargs)
        self.variance = std_dev ** 2
        self.mean = mean

    def get_applied_images(self, images, *additional):
        """
        Applies the gaussian noise to the specified images.

        Args:
            images (torch.Tensor [BCHW]): Batched images tensor.

        Returns:
            torch.Tensor with blurred images.
        """
        # Pick a variance at random between [0, self.variance], global to the image
        applied_variance = torch.rand((images.shape[0], 1, 1, 1), **options(images)).mul_(self.variance)
        # Sample the noise and scale
        noise = torch.randn_like(images).mul_(applied_variance).add_(self.mean)

        return images + noise


class CompositeAugmentation(object):
    """Augmentation that applies multiple augmentations in a single call."""

    def __init__(self, augmentations):
        """
        Initialization.

        Args:
            augmentations (list): List of child augmentations.
        """
        self.augmentations = augmentations

    def __call__(self, images, *additional):
        """
        Applies the set of augmentations to the images.

        Args:
            images (torch.Tensor [BCHW]): The batch of images.
        """
        for augmentation in self.augmentations:
            images = augmentation(images, *additional)
        return images
