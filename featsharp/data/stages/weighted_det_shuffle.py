# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from logging import Logger
import math
import sqlite3
from typing import Union, Callable, Any, List

import torch

import webdataset as wds

from .utils import SharedEpoch, pytorch_worker_seed


class WeightedDetShuffle(wds.PipelineStage):
    def __init__(self,
                 database: Union[str, sqlite3.Connection],
                 key_extractor: Callable[[Any], str],
                 weight_mode: str = 'frequency',
                 bufsize: int = 1000,
                 batch_size: int = 512,
                 seed: int = 0,
                 epoch: int = -1,
                 deterministic: bool = True,
                 logger: Logger = None,
    ):
        self.bufsize = bufsize
        self.seed = seed
        self.epoch = epoch
        self.logger = logger
        self.deterministic = deterministic
        self.batch_size = batch_size

        self.key_extractor = key_extractor

        if not weight_mode or weight_mode in ('none', 'uniform'):
            weight_fn = lambda v: 1
        elif weight_mode == 'inv_frequency':
            weight_fn = lambda v: 1 / v
        elif weight_mode == 'inv_sq_frequency':
            weight_fn = lambda v: 1 / (v ** 2)
        elif weight_mode == 'inv_log_frequency':
            weight_fn = lambda v: 1 / math.log(v + 1)
        else:
            raise ValueError(f'Unsupported weight function: {weight_fn}')
        self.weight_fn = weight_fn

        if isinstance(database, str):
            logger.info('Loading database...')
            self.conn = sqlite3.connect(database, isolation_level=None, check_same_thread=False, uri=True)

            # Set the connection to read-only mode
            self.conn.execute('PRAGMA query_only = 1')
            self.conn.execute('PRAGMA read_uncommitted = true')
            logger.info('Done')
        else:
            self.conn = database

    def _get_weights(self, keys: List[Union[str, bytes]]):
        query = 'SELECT md5, count FROM md5_values WHERE md5 IN ({})'.format(', '.join(['?'] * len(keys)))

        self.cursor.execute(query, tuple(keys))
        results = self.cursor.fetchall()

        weights = []
        count_map = {key: 1 for key in keys}

        for key, count in results:
            count_map[key] = count

        for key in keys:
            weight = self.weight_fn(count_map[key])
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch

        rng = torch.Generator()
        if self.deterministic:
            seed = pytorch_worker_seed(1001 * epoch)
            # We want every worker and every rank to get the same seed,
            # that way they shuffle identically
            rng.manual_seed(seed)

        def get_next_samples(count: int):
            dataset = [
                next(src)
                for _ in range(count)
            ]
            keys = [
                self.key_extractor(d)
                for d in dataset
            ]
            weights = self._get_weights(keys)
            return weights, keys, dataset

        self.cursor = self.conn.cursor()

        # Fill the buffers initially
        weights, keys, samples = get_next_samples(self.bufsize)

        done = False
        def replace_idxs(idxs: List[int]):
            new_weights, new_keys, new_samples = get_next_samples(len(idxs))

            # Refill the reservoir with new values
            for i, idx in enumerate(idxs):
                samples[idx] = new_samples[i]
                weights[idx] = new_weights[i]

        iter_ct = 0
        while not done:
            # Select `batch_size` worth of samples from the reservoir
            idxs = torch.multinomial(weights, self.batch_size, replacement=False, generator=rng).tolist()

            # Release all of the sampled values
            for idx in idxs:
                yield samples[idx]

            replace_idxs(idxs)

            iter_ct += 1

            if iter_ct % 10 == 0:
                # Purge the worst weights
                idxs = torch.topk(weights, self.batch_size, largest=False, sorted=False).indices.tolist()
                replace_idxs(idxs)

        self.cursor.close()
