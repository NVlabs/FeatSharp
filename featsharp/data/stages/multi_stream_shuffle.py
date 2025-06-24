# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import random
from typing import List, Union, Optional, Callable

from torch.utils.data import IterableDataset

import webdataset as wds

from .utils import SharedEpoch, expand_urls, pytorch_worker_seed, iterator_exhauster, ignore_and_log

class MultiStreamShuffle(IterableDataset):
    def __init__(self,
                 urls: List[str],
                 num_streams: int = 4,
                 epoch: Union[int, SharedEpoch] = -1,
                 deterministic: bool = True,
                 reduce_urls_fn: Optional[Callable[[List[str]], List[str]]] = None):
        super().__init__()

        self.urls = sorted(urls if isinstance(urls, (list, tuple)) else expand_urls(urls)[0])
        self.num_streams = num_streams
        self.epoch = epoch
        self.deterministic = deterministic
        self.reduce_urls_fn = reduce_urls_fn

        self.rng = random.Random()

    def __iter__(self):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            seed = pytorch_worker_seed(epoch)
            self.rng.seed(seed)

        # This can potentially reduce a full set of urls into a partitioned
        # set assigned exclusively to this rank/worker
        urls = self._reduce_urls()

        streams = [
            iterator_exhauster(self._get_url_sampler(urls))
            for _ in range(self.num_streams)
        ]

        while True:
            for stream in streams:
                yield next(stream)


    def _get_url_sampler(self, urls: List[str]):
        def sample_fn():
            url = dict(url=self.rng.choice(urls))
            for sample in wds.tarfile_samples([url], handler=ignore_and_log):
                yield sample
        return sample_fn

    def _reduce_urls(self):
        if self.reduce_urls_fn is not None:
            return self.reduce_urls_fn(self.urls)
        return self.urls
