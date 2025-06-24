# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import random
from typing import Tuple

import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import ToPILImage

from timm.layers import to_2tuple

from .utils import compute_md5_hash

_SYNTH_WORDS = ['hello', 'world', 'foo', 'bar', 'baz', 'nvidia', 'adlr', 'yeti', 'mountain', 'bike']


class SyntheticDataset(IterableDataset):
    def __init__(self, img_size: Tuple[int, Tuple[int, int]]):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.to_pil_image = ToPILImage(mode='RGB')

    def __iter__(self):
        image_height, image_width = self.img_size

        while True:
            gen_height_f = 1 + random.random() * 0.5
            gen_width_f = 1 + random.random() * 0.5

            gen_height = int(image_height * gen_height_f)
            gen_width = int(image_width * gen_width_f)

            image = torch.randn(3, gen_height, gen_width, dtype=torch.float32)

            image = self.to_pil_image(image)

            words = random.choices(_SYNTH_WORDS, k=24)

            text = ' '.join(words)
            md5 = compute_md5_hash(text).hexdigest()

            json = { 'md5': md5 }

            yield (image, text, json)
