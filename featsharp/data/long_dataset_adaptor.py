# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import Optional

import torch
from torch.utils.data import IterableDataset
from logging import getLogger

from .stages import SharedEpoch

_LOGGER = getLogger(__name__)


def _forever_iterator(ds: IterableDataset,
                      shared_epoch: Optional[SharedEpoch]):
    def _get_iter():
        vals = iter(ds)
        if shared_epoch is not None:
            shared_epoch.increment()
        return vals

    vals = _get_iter()
    while True:
        try:
            v = next(vals)
            yield v
        except StopIteration:
            _LOGGER.info('Dataset iterator exhausted. Fetching a new one.')
            vals = _get_iter()


class LongDatasetAdaptor:
    def __init__(self, loader: IterableDataset,
                 steps_per_epoch: int, reset_each_epoch: bool = False,
                 shared_epoch: SharedEpoch = None):
        self.loader = loader
        self.steps_per_epoch = steps_per_epoch
        self.reset_each_epoch = reset_each_epoch
        self.shared_epoch = shared_epoch

        if reset_each_epoch:
            next(iter(loader))
        else:
            self._iter = _forever_iterator(self.loader, self.shared_epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        if self.reset_each_epoch:
            self._iter = iter(self.loader)

        for _ in range(self.steps_per_epoch):
            v = next(self._iter)
            yield v
