# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from collections.abc import Mapping
from typing import Iterator, Tuple, Any, Union, List, Dict

import torch
from torch.utils.data import IterableDataset

DATA_T = Union[int, float, torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]

class Prefetcher:
    """Follows same interface as Fetcher, but implements overlapped
    prefetching in a separate stream.
    """

    def __init__(self, loader: Iterator[Tuple[DATA_T, DATA_T]]):
        self.next_data = None

        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()
        self.done = False

    def __len__(self):
        """Enables len(fetcher)"""
        return len(self.loader)

    def preload(self):
        """Fetch the next input/target pair on a separate stream.
        Trigger H2D copy
        """
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.done = True
            return
        with torch.cuda.stream(self.stream):
            self.next_data = _to_cuda(self.next_data)

    def next(self):
        """Return prefetched image/target and trigger next h2d.
        We first wait for the h2d op to be complete, which is precautionary
        since the compute stream is expected to be busy much longer.
        The call to record_stream marks the tensor as being used by a different
        stream than the one in which it was allocated (which is the prefetch stream),
        and prevents premature return to the mem pool.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.record_stream
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            _record_stream(data, torch.cuda.current_stream())
        self.preload()
        return data


class IterablePrefetcher(IterableDataset):
    def __init__(self, inner: IterableDataset):
        super().__init__()
        self.inner = inner

    def __iter__(self):
        prefetcher = Prefetcher(iter(self.inner))

        while not prefetcher.done:
            yield prefetcher.next()


def _record_stream(data: DATA_T, stream: torch.cuda.Stream):
    if torch.is_tensor(data):
        data.record_stream(stream)
    elif isinstance(data, (list, tuple)):
        for d in data:
            _record_stream(d, stream)
    elif isinstance(data, Mapping):
        for v in data.values():
            _record_stream(v, stream)

def _to_cuda(data: DATA_T):
    if torch.is_tensor(data):
        return data.cuda(non_blocking=True)
    if isinstance(data, (list, tuple)):
        return [_to_cuda(d) for d in data]
    if isinstance(data, Mapping):
        return {k: _to_cuda(v) for k, v in data.items()}
    return data
