# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset


def options(t: torch.Tensor):
    return dict(dtype=t.dtype, device=t.device)


class FewImageDataset(Dataset):
    def __init__(self, ds, l: int):
        self.samples = []
        for z, sample in enumerate(ds):
            self.samples.append(sample)
            if z == l - 1:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
