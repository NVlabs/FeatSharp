# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import random
from typing import Union, List

import webdataset as wds

from .utils import SharedEpoch


class URLPartitioner:
    def __init__(self,
                 epoch: Union[int, SharedEpoch] = -1,
                 deterministic: bool = True,
    ):
        self.epoch = epoch
        self.deterministic = deterministic

        self.rng = random.Random()

    def __call__(self, urls: List[str]):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # We want every worker and every rank to get the same seed,
            # that way they shuffle identically
            seed = 1001 * epoch
            self.rng.seed(seed)

        urls = list(urls)  # Make a copy
        self.rng.shuffle(urls)

        urls = wds.split_by_node(urls)
        urls = list(wds.split_by_worker(urls))
        return urls
