# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
"""
Spatial transforms module.

The difference, currently, between a transform and an augmentation is that augmentations are both
non-geometric, and are applied explicitly on batches of images on the GPU. Future implementations
could potentially combine transforms and augmentations into a single pipe.

NOTE: The usage of randomized transforms is still allowed in a deterministic setting, if and only
      if the torch and numpy random seeds (`torch.manual_seed(seed)` and `numpy.random.seed(seed)`)
      are deterministically set *before* any of these modules are used.
"""

import logging
from typing import List, Union, Tuple

import cv2
import math
import numpy
import torch
import torch.distributed

from .defer_image import DeferImage

logger = logging.getLogger(__name__)


class Identity:
    def __call__(self, *items, **kwargs):
        pass


class CropTransform(object):
    """Crops values to the specified sub-window."""

    def __init__(self, x, y, width, height):
        """
        Initializer.

        Args:
            x (int): The x-coordinate to crop to.
            y (int): The y-coordinate to crop to.
            width (int): The width of the resulting crop.
            height (int): The height of the resulting crop.
        """
        if x < 0 or y < 0:
            raise ValueError("x and y must be >= 0")
        if width < 1 or height < 1:
            raise ValueError("width and height must be >= 1")

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.translation = torch.tensor([-self.x, -self.y], dtype=torch.float32)

    def __call__(self, *items, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects that implement one of two possible
        cropping interfaces:

        If `clip_translate` is defined, it will be preferred.
            Signature: (x, y, max_x, max_y)

        otherwise, if `translate` is defined, it will be invoked.
            Signature: (torch.Tensor([x, y], float32))

        NOTE: Operations are expected to be performed in-place.
        """
        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(self.x, self.y, self.width, self.height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(self.translation)


class RandomTransformBase:
    def _prepare_rng(self, rngs: Union[numpy.random.Generator, List[numpy.random.Generator]]):
        is_prepared = getattr(self, '_is_rng_worker_prepared', False)

        if is_prepared:
            return

        if not isinstance(rngs, (list, tuple)):
            rngs = [rngs]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # This will get every worker on every rank to a different point in the bit sequence, to prevent
            # redundant transforms
            for _ in range(worker_id * 31):
                for rng in rngs:
                    rng.bit_generator.random_raw()

        setattr(self, '_is_rng_worker_prepared', True)


class RandomCropTransform(RandomTransformBase):
    """
    Randomly crops the input image to be of a specific size.

    NOTE: The first supplied item to `__call__` must have a `.shape` attribute, which is used
          to determine the random parameters. All other supplied items will be cropped
          accordingly.
    """

    def __init__(self, width, height, random_seed: int = None, quant_shift: int = None, jitter: bool = False, jit_seed: int = None):
        self.width = width
        self.height = height

        self.rng = numpy.random.default_rng(random_seed)
        self.quant_shift = quant_shift
        self.jitter = jitter
        self.jitrng = numpy.random.default_rng(jit_seed)

    def __call__(self, *items, crop_window=None, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects that implement one of two possible
        cropping interfaces:

        If `clip_translate` is defined, it will be preferred.
            Signature: (x, y, max_x, max_y)

        otherwise, if `translate` is defined, it will be invoked.
            Signature: (torch.Tensor([x, y], float32))

        NOTE: Operations are expected to be performed in-place.
        """
        self._prepare_rng([self.rng, self.jitrng])

        image_shape = items[0].shape

        crop_width = min(self.width, image_shape[-1])
        crop_height = min(self.height, image_shape[-2])
        if crop_window is not None:
            crop_width = crop_window[2] - crop_window[0]
            crop_height = crop_window[3] - crop_window[1]

        slide_x = image_shape[-1] - crop_width
        slide_y = image_shape[-2] - crop_height

        x_fact = self.rng.random()
        y_fact = self.rng.random()

        x = x_fact * slide_x
        y = y_fact * slide_y

        if self.jitter:
            assert self.quant_shift is not None, "`quant_shift` must be specified when jittering"
            x_jit = round(self.jitrng.normal(scale=0.5))
            y_jit = round(self.jitrng.normal(scale=0.5))

            x2 = x + x_jit * self.quant_shift
            y2 = y + y_jit * self.quant_shift

            if 0 <= x2 <= slide_x:
                x = x2
            if 0 <= y2 <= slide_y:
                y = y2
        elif self.quant_shift is not None:
            x = int(x / self.quant_shift) * self.quant_shift
            y = int(y / self.quant_shift) * self.quant_shift

        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(x, y, crop_width, crop_height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(torch.tensor([-x, -y], dtype=torch.float32))
        pass


class CenterCropTransform:
    """
    Randomly crops the input image to be of a specific size.

    NOTE: The first supplied item to `__call__` must have a `.shape` attribute, which is used
          to determine the random parameters. All other supplied items will be cropped
          accordingly.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, *items, crop_window=None, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects that implement one of two possible
        cropping interfaces:

        If `clip_translate` is defined, it will be preferred.
            Signature: (x, y, max_x, max_y)

        otherwise, if `translate` is defined, it will be invoked.
            Signature: (torch.Tensor([x, y], float32))

        NOTE: Operations are expected to be performed in-place.
        """
        image_shape = items[0].shape

        crop_width = self.width
        crop_height = self.height
        if crop_window is not None:
            crop_width = crop_window[2] - crop_window[0]
            crop_height = crop_window[3] - crop_window[1]

        # if image_shape[-2] < crop_height:
        #     raise ValueError("The image height is too small for this crop window!\n"
        #                      "Image: {}, Crop: {}"
        #                      .format(image_shape[-2], crop_height))
        # if image_shape[-1] < crop_width:
        #     raise ValueError("The image width is too small for this crop window!\n"
        #                      "Image: {}, Crop: {}"
        #                      .format(image_shape[-1], crop_width))

        x = (image_shape[-1] - crop_width) / 2
        y = (image_shape[-2] - crop_height) / 2

        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(x, y, crop_width, crop_height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(torch.tensor([-x, -y], dtype=torch.float32))


class ScaleTransform(object):
    """
    Scales values to be of the specified size.

    NOTE: The first supplied item to `__call__` must have a `.shape` attribute, which is used
          to determine the scaling parameters. All other supplied items will be scaled
          accordingly.
    """

    def __init__(self, width, height):
        """
        Initialization.

        Args:
            target_width (int): The desired output width of the first supplied item.
            target_height (int): The desired output height of the first supplied item.
        """
        if width < 1 or height < 1:
            raise ValueError("Target sizes must be at least 1.")

        self.target_width = width
        self.target_height = height

    def __call__(self, *items, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects the implement the following interface:
            `scale` with signature `(torch.Tensor([width_factor, height_factor] float32))`

        NOTE: Operations are expected to be performed in-place.
        """
        image_size = items[0].shape

        height_factor = float(self.target_height) / image_size[-2]
        width_factor = float(self.target_width) / image_size[-1]

        for item in items:
            apply_fn = getattr(item, 'scale', None)
            if apply_fn:
                apply_fn(torch.tensor([width_factor, height_factor]))


class MinSizeTransform(object):
    """
    Scales values to be of *at least* the specified size, preserving aspect ratio.

    NOTE: The first supplied item to `__call__` must have a `.shape` attribute, which is used
          to determine the scaling parameters. All other supplied items will be scaled
          accordingly.
    """

    def __init__(self, min_width, min_height):
        if min_width < 1 or min_height < 1:
            raise ValueError("Target sizes must be at least 1.")
        self.min_width = int(round(min_width))
        self.min_height = int(round(min_height))
        self.aspect = self.min_width / self.min_height

    def __call__(self, *items, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects the implement the following interface:
            `scale` with signature `(torch.Tensor([width_factor, height_factor] float32))`

        NOTE: Operations are expected to be performed in-place.
        """
        image_size = items[0].shape

        image_aspect = image_size[-1] / image_size[-2]

        # Width relatively larger than height, so constrain the height
        if self.aspect < image_aspect:
            scale = self.min_height / image_size[-2]
        else:
            scale = self.min_width / image_size[-1]

        if scale <= 1.0:
            return

        scale_tensor = torch.tensor([scale, scale], dtype=torch.float32)

        for item in items:
            apply_fn = getattr(item, 'scale', None)
            if apply_fn:
                apply_fn(scale_tensor)


class MaxSizeTransform(object):
    def __init__(self, width: float, height: float, smallest: bool=False):
        self.width = width
        self.height = height
        self.smallest = smallest

    def __call__(self, *items, **kwargs):
        image_size = items[0].shape
        scale_1 = self.width / image_size[-1]
        scale_2 = self.height / image_size[-2]

        def comp(a, b):
            if not self.smallest:
                return a < b
            else:
                return b < a

        scale = scale_1 if comp(scale_1, scale_2) else scale_2

        scale_tensor = torch.tensor([scale, scale], dtype=torch.float32)

        for item in items:
            apply_fn = getattr(item, 'scale', None)

            if apply_fn:
                apply_fn(scale_tensor)
        pass


def linear_sample(a, b, rng: numpy.random.Generator = None):
    rval = rng.random() if rng is not None else numpy.random.random()

    return (b - a) * rval + a


def log_linear_sample(a, b, rng: numpy.random.Generator = None):
    la = math.log(a)
    lb = math.log(b)

    s = linear_sample(la, lb, rng=rng)

    r = math.exp(s)
    return r


def clamped_normal(std, min_val, max_val, rng: numpy.random.Generator = None, **kwargs):
    def _sample():
        return rng.normal(**kwargs) if rng is not None else numpy.random.randn(**kwargs)

    if min_val == max_val:
        return std * _sample()

    s = std * _sample()
    invalid = numpy.logical_or(min_val > s, max_val < s)
    while numpy.any(invalid):
        s2 = std * _sample()
        s = numpy.where(invalid, s2, s)
        invalid = numpy.logical_or(min_val > s, max_val < s)
    return s

class RandomZoomTransform(RandomTransformBase):
    """
    Randomly "zooms" the items to be within some scale of the original.

    NOTE: The first supplied item to `__call__` must have a `.shape` attribute, which is used
          to determine the scaling parameters. All other supplied items will be scaled
          accordingly.

    NOTE: The zooming is performed with the bottom center of the image anchored.
    """

    def __init__(self, min_ratio, max_ratio, retain_dims=True, batch_size=1, fixed_width=False, fixed_height=False, random_seed: int = None):
        """
        Initializer.

        Args:
            min_ratio (float): The minimum scale factor that the zoom can realize.
            max_ratio (float): The maximum scale factor that the zoom can realize.
            batch_size (int): The number of consecutive calls that will get the same
                              cropping values.
        """
        if min_ratio <= 0:
            raise ValueError("Min ratio must be > 0")
        if max_ratio <= min_ratio:
            raise ValueError("max_ratio must be > min_ratio")

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.batch_size = batch_size
        self.counter = 0
        self.scale = 1.0
        self.retain_dims = retain_dims
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects that implement *both* of the following
        interfaces:
            `translate` with signature (torch.Tensor([x, y], float32))
            `scale` with signature (torch.Tensor([width_factor, height_factor], float32),
                                    retain_dims=True)

            `retain_dims` in this context means that the size of the image should be unchanged
                          as a result of this function call.

        NOTE: Operations are expected to be performed in-place

        NOTE: The first value provided in `items` must have the `.shape` attribute and will define
              the extent of the values that are allowed to be realized.
        """
        self._prepare_rng(self.rng)

        image_size = items[0].shape

        if self.counter == 0:
            if self.min_ratio < 1 and self.max_ratio > 1:
                self.scale = log_linear_sample(self.min_ratio, self.max_ratio, self.rng)
            else:
                self.scale = linear_sample(self.min_ratio, self.max_ratio, self.rng)
        self.counter = (self.counter + 1) % self.batch_size

        scale_w = self.scale if not self.fixed_width else 1
        scale_h = self.scale if not self.fixed_height else 1
        scale_tensor = torch.tensor([scale_w, scale_h], dtype=torch.float32)

        trans_tensor = torch.tensor([-image_size[-1] / 2, -image_size[-2] / 2], dtype=torch.float32)
        trans_tensor2 = torch.tensor([image_size[-1] / 2 * scale_w, image_size[-2] / 2 * scale_h], dtype=torch.float32)

        for item in items:
            scale_fn = getattr(item, 'scale', None)
            trans_fn = getattr(item, 'translate', None)

            if (True or self.retain_dims) and trans_fn is not None:
                trans_fn(trans_tensor)

            if scale_fn is not None:
                scale_fn(scale_tensor, retain_dims=self.retain_dims)

            if (True or self.retain_dims) and trans_fn is not None:
                trans_fn(trans_tensor2)
        pass


class RandomFlipTransform(RandomTransformBase):
    """Randomly "flips" the items horizontally, about the center of the image."""

    def __init__(self, prob_apply=0.5, batch_size=1, random_seed: int = None):
        """
        Initializer.

        Args:
            prob_apply (float [0 <= p <= 1]): The probability of flipping the values.
            batch_size (int): The number of consecutive calls that will get the same
                              flip realization.
        """
        if prob_apply < 0 or prob_apply > 1:
            raise ValueError("prob_apply must be between 0 and 1 inclusive.")

        self.prob_apply = prob_apply
        self.batch_size = batch_size
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects that implement the follow interface:
            `flip` with signature (x_offset)

            x_offset is defined as x-axis across which the values should be flipped.

        NOTE: Operations are expected to be performed in-place

        NOTE: The first value provided in `items` must have the `.shape` attribute and will define
              the x_offset to run the flip on.
        """
        self._prepare_rng(self.rng)

        if self.counter == 0:
            r_val = self.rng.random()
            self.do_apply = r_val <= self.prob_apply
        self.counter = (self.counter + 1) % self.batch_size

        if not self.do_apply:
            return

        image_size = items[0].shape

        for item in items:
            flip_fn = getattr(item, 'flip', None)

            if flip_fn is not None:
                flip_fn(image_size[-1] / 2)
        pass


class RandomTranslationTransform(RandomTransformBase):
    """Randomly translates values along the x axis."""

    def __init__(self, std_dev, abs_max=0, batch_size=1, random_seed: int = None):
        """
        Initializer.

        Args:
            std_dev (float): The standard deviation of the translation values.
            abs_max (float): The maximum deviation that the translation can attain.
                             If 0, then any value can be randomly selected.
            batch_size (int): The number of consecutive calls that will get the same
                              cropping values.
        """
        self.std_dev = std_dev ** 2
        self.abs_max = abs_max
        self.batch_size = batch_size
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """
        Applies the transform to the specified objects.

        This API has support for dependent objects that implement the following interface:
            `translate` with signature (torch.Tensor([x, 0], float32))

        NOTE: Operations are expected to be performed in-place.
        """
        self._prepare_rng(self.rng)

        if self.counter == 0:
            translation = numpy.array([
                clamped_normal(self.std_dev, -self.abs_max, self.abs_max, rng=self.rng),
                clamped_normal(self.std_dev, -self.abs_max, self.abs_max, rng=self.rng),
            ])
            self.translation = translation
        self.counter = (self.counter + 1) % self.batch_size

        trans_vec = torch.from_numpy(translation)

        for item in items:
            trans_fn = getattr(item, 'translate', None)

            if trans_fn is not None:
                trans_fn(trans_vec)


class RandomRotationTransform(RandomTransformBase):
    """Randomly rotates the tensor."""

    def __init__(self, abs_max=90, batch_size=1, random_seed: int = None):
        self.abs_max = abs_max
        self.batch_size = batch_size
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        self._prepare_rng(self.rng)

        if self.counter == 0:
            rotation = linear_sample(-self.abs_max, self.abs_max, rng=self.rng)
            rotation = (numpy.pi / 180) * rotation
            self.rotation = rotation
        self.counter = (self.counter + 1) % self.batch_size

        image_size = items[0].shape

        translate_x = image_size[-1] / 2
        translate_y = image_size[-2] / 2

        trans_vec = torch.tensor([-translate_x, -translate_y],
                                 dtype=torch.float32)

        rotation = torch.tensor([
            [math.cos(self.rotation), -math.sin(self.rotation)],
            [math.sin(self.rotation), math.cos(self.rotation)]
        ], dtype=torch.float32)

        for item in items:
            trans_fn = getattr(item, 'translate', None)
            rot_fn = getattr(item, 'rotate', None)

            if (trans_fn is None) ^ (rot_fn is None):
                raise ValueError("One of 'translate' or 'rotate' is defined, but not both.")

            if trans_fn is not None:
                trans_fn(trans_vec)
                rot_fn(rotation)

                tv2 = torch.tensor([
                    items[0].shape[-1] / 2,
                    items[0].shape[-2] / 2
                ], dtype=torch.float32)

                trans_fn(tv2)
        pass


class RandomPerspectiveTransform(RandomTransformBase):
    """Randomly applies a perspective deformation of the image/polygons."""

    def __init__(self, scale: Union[float, Tuple[float, float]] = (0.05, 0.1), batch_size=1, prob_apply=0.5, random_seed: int = None):
        if not isinstance(scale, tuple):
            scale = (0.0, scale)
        self.scale = scale
        self.batch_size = batch_size
        self.prob_apply = prob_apply
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        self._prepare_rng(self.rng)

        if self.counter == 0:
            scale = linear_sample(*self.scale, self.rng)
            offsets = clamped_normal(scale, -0.45, 0.45, self.rng, size=(4, 2)).astype(numpy.float32)
            self.sampled_offsets = offsets

        if self.rng.random() > self.prob_apply:
            return

        self.counter = (self.counter + 1) % self.batch_size

        image: DeferImage = items[0]

        h, w = image.shape[-2:]
        in_bds = numpy.array([
            [0, 0], [w, 0], [w, h], [0, h],
        ], dtype=numpy.float32)

        # in_bds = image.get_bds()[:, :2].numpy().copy()

        offsets = numpy.copy(self.sampled_offsets)
        offsets[:, 0] *= w
        offsets[:, 1] *= h

        out_bds = in_bds + offsets
        min_out = numpy.min(out_bds, axis=0)
        out_bds -= min_out

        H = cv2.getPerspectiveTransform(in_bds, out_bds)

        H = torch.from_numpy(H).float()

        H_inv = torch.inverse(H)

        image.apply_stm(H_inv)
        for i in range(1, len(items)):
            items[i].apply_stm(H.T, perspective=True)

        target_width = numpy.max(out_bds[:, 0]) - numpy.min(out_bds[:, 0])
        target_height = numpy.max(out_bds[:, 1]) - numpy.min(out_bds[:, 1])

        image.target_size = image.target_size[:-2] + (target_height, target_width)


class PadToTransform(RandomTransformBase):
    def __init__(self, width: int, height: int, random_seed: int = None, quant_pad: int = None, rand_pad: bool = True):
        self.width = width
        self.height = height
        self.quant_pad = quant_pad
        self.rand_pad = rand_pad

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, crop_window=None, **kwargs):
        self._prepare_rng(self.rng)

        crop_width = self.width
        crop_height = self.height
        if crop_window is not None:
            crop_width = crop_window[2] - crop_window[0]
            crop_height = crop_window[3] - crop_window[1]

        image_size = items[0].shape
        crop_width = max(crop_width, image_size[-1])
        crop_height = max(crop_height, image_size[-2])

        pad_w = crop_width - image_size[-1]
        pad_h = crop_height - image_size[-2]

        # No padding necessary
        if pad_w == 0 and pad_h == 0:
            return

        x = (self.rng.random() if self.rand_pad else 0) * -pad_w
        y = (self.rng.random() if self.rand_pad else 0) * -pad_h

        if self.quant_pad is not None:
            x = int(x / self.quant_pad) * self.quant_pad
            y = int(y / self.quant_pad) * self.quant_pad

        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(x, y, crop_width, crop_height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(torch.tensor([-x, -y], dtype=torch.float32))
        pass


class MosaicTransform:
    def __init__(self, width: int, height: int, prob_apply: float=0.1):
        self.width = width
        self.height = height
        self.prob_apply = prob_apply

    def __call__(self, *items, **kwargs):
        # This object just acts as a placeholder, so that it can be defined in the experiment spec
        # regularly. However, because it requires special sampling and composition mechanics, the
        # actual functionality gets lifted into the data/mosaic_sampler.py and data/mosaic_dataset.py
        # objects. We just use this object to configure the settings for the algorithm.
        pass


class RandomChoiceTransform(object):
    """Randomly selects a sub-transform from a list of transforms."""

    def __init__(self, choices, weights=None, batch_size=1):
        """
        Initializer.

        If `weights` is not specified, then the transform is selected using a uniform distribution.
        Otherwise, the transform is selected from the multinomial distribution.

        Args:
            choices (list<transform>): The set of possible transforms to apply.
            weights (None, torch.Tensor): Optional. The multinomial distribution.
            batch_size (int): The number of consecutive calls that will get the same
                              cropping values.
        """
        self.choices = choices
        if weights is None:
            weights = torch.empty((len(choices),), dtype=torch.float32).fill_(1.0 / len(choices))

        if weights.dim() != 1:
            raise ValueError("Provided distribution must be 1d")
        if weights.shape[0] != len(choices):
            raise ValueError("Provided distribution must have the same number of weights as there "
                             "are choices.")

        self.weights = weights
        self.batch_size = batch_size
        self.counter = 0

    def __call__(self, *items, **kwargs):
        """
        Applies one of the sub-transforms to the specified objects.

        NOTE: Operations are expected to be performed in-place.
        """
        if self.counter == 0:
            self.which = torch.multinomial(self.weights, 1)[0]
        self.counter = (self.counter + 1) % self.batch_size

        which = self.which

        self.choices[which](*items, **kwargs)


class CompositeTransform(object):
    """Applies a set of sub-transforms in the specified order."""

    def __init__(self, transforms):
        """
        Initializer.

        Args:
            transforms (list<transform>): The list of sub-transforms to apply in order.
        """
        self.transforms = transforms

    def __call__(self, items, **kwargs):
        """
        Applies each transform, in order, to the specified objects.

        NOTE: Operations are expected to be performed in-place.
        """
        mark_dirties = []
        for item in items:
            coal_fn = getattr(item, 'coalesce_homogeneous', None)
            if coal_fn is not None:
                coal_fn()
                mark_dirties.append(item)

        for tx in self.transforms:
            tx(*items, **kwargs)

        for item in mark_dirties:
            item.mark_dirty()
        if len(items) > 1:
            return items
        return items[0]

    def set_dataset_name(self, name):
        for tx in self.transforms:
            st = getattr(tx, 'set_dataset_name', None)
            if st is not None:
                st(name)
