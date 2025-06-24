# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import math
from typing import Dict
from PIL import Image

import webdataset as wds


class UniformColorFilter(wds.PipelineStage):
    def __init__(self,
                 image_tuple_idx: int = 0,
                 threshold: int = 1,
                 verbose: bool = False,
    ):
        self.image_tuple_idx = image_tuple_idx
        self.threshold = threshold

        self.num_seen = 0
        self.num_filtered = 0
        self.verbose = verbose

    def run(self, src):
        for sample in src:
            im: Image = sample[self.image_tuple_idx]

            extrema = im.getextrema()

            val_range = math.sqrt(
                sum((e[1] - e[0]) ** 2 for e in extrema)
            )

            self.num_seen += 1

            if val_range > self.threshold:
                yield sample
            else:
                self.num_filtered += 1

                if self.verbose or (self.num_filtered % 100) == 0:
                    pct_filtered = self.num_filtered / self.num_seen * 100
                    print(f'Filtered uniform image. Value range: {extrema}. Percent Filtered: {pct_filtered:0.3f}%')
