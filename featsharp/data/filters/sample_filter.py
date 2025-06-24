# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from collections import defaultdict
from typing import Set, Callable, Dict, Optional, Union, List
from logging import getLogger

import webdataset as wds


_LOGGER = getLogger(__name__)


class SampleFilterStage(wds.PipelineStage):
    def __init__(self,
                 blacklist: Set[str],
                 key_extractor: Callable[[Dict[str, bytes]], str],
                 filter_callback: Optional[Callable[[str, int, Dict[str, bytes]], None]] = None,
                 verbose: bool = False,
    ):
        self.blacklist = blacklist
        self.key_extractor = key_extractor
        self.filter_callback = filter_callback

        self.occurrences = defaultdict(lambda: 0)

        self.num_passed = 0
        self.num_filtered = 0
        self.verbose = verbose

    def run(self, src):
        for sample in src:
            try:
                key = self.key_extractor(sample)
            except Exception as e:
                _LOGGER.warning(f'Failed to process sample due to error: {e}')
                continue

            if key not in self.blacklist:
                self.num_passed += 1
                # Not filtered
                yield sample
            else:
                self.occurrences[key] += 1
                self.num_filtered += 1
                occ = self.occurrences[key]

                if self.verbose or (self.num_filtered % 1000) == 0:
                    total_samples = self.num_passed + self.num_filtered
                    pct_filtered = self.num_filtered / total_samples * 100
                    print(f'Filtered key "{key}". Occurrences: {occ}. Total: {self.num_filtered}, Percent: {pct_filtered:0.3f}%')

                if self.filter_callback is not None:
                    self.filter_callback(key, occ, sample)


def get_json_key_extractor(key_names: Union[bytes, List[bytes]], decode: bool = True):
    if not isinstance(key_names, (list, tuple)):
        key_names = [key_names]

    start_keys = [
        b'"' + key_name + suffix
        for key_name in key_names
        for suffix in [b'":"', b'": "']
    ]
    end_key = b'"'

    def key_extractor(data: Dict[str, bytes]):
        j_data = data['json']

        for start_key in start_keys:
            key_start = j_data.find(start_key)

            if key_start != -1:
                break

        if key_start == -1:
            return None

        key_end = j_data.find(end_key, key_start + len(start_key))
        b_extract = j_data[key_start + len(start_key) : key_end]
        if decode:
            return b_extract.decode()
        return b_extract

    return key_extractor
