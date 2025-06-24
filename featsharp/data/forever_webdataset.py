# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import webdataset as wds

_ITER_KEY = '__iter_impl'

class ForeverWebDataset(wds.WebDataset):
    def __iter__(self):
        iter_impl = self._get_iter()
        while True:
            try:
                yield next(iter_impl)
            except StopIteration:
                setattr(self, _ITER_KEY, None)
                iter_impl = self._get_iter()

    def _get_iter(self):
        iter_impl = getattr(self, _ITER_KEY, None)
        if iter_impl is None:
            iter_impl = self.iterator()
            setattr(self, _ITER_KEY, iter_impl)
        return iter_impl
