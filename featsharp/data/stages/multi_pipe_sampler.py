# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import random
from typing import List, Iterable, Union

from torch.utils.data import IterableDataset

from .utils import SharedEpoch, pytorch_worker_seed, iterator_exhauster

class MultiPipeSampler(IterableDataset):
    def __init__(self,
                 pipes: List[Iterable],
                 rates: List[float],
                 epoch: Union[int, SharedEpoch] = -1,
                 deterministic: bool = True,
    ):
        super().__init__()

        total_rate = sum(rates)
        self.pipes = pipes
        self.rates = [r / total_rate for r in rates]
        self.epoch = epoch
        self.deterministic = deterministic

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

        streams = [
            iterator_exhauster(self.get_pipe_iterator(p))
            for p in self.pipes
        ]

        while True:
            stream = self.rng.choices(streams, self.rates)[0]

            yield next(stream)

    def get_pipe_iterator(self, pipe):
        def fn():
            iterator = iter(pipe)
            yield from iterator
        return fn
