# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import os
import sqlite3
import sys


_CONN_MAP = dict()

def get_hash_database(dataset_root: str):
    global _CONN_MAP

    for _ in range(3):
        db_path = os.path.join(dataset_root, "hash_counts.db")
        if os.path.exists(db_path):
            break
        dataset_root = os.path.dirname(dataset_root)
    else:
        return None

    if db_path in _CONN_MAP:
        return _CONN_MAP[db_path]

    db = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, uri=True)
    db.execute('PRAGMA query_only = 1')
    db.execute('PRAGMA read_uncommitted = true')

    _CONN_MAP[db_path] = db
    return db


_FILTER_MAP = dict()

def get_hash_count_filter(db: sqlite3.Connection, max_count: int = 100):
    global _FILTER_MAP

    filter_set = _FILTER_MAP.get(id(db), None)
    if filter_set is not None:
        return filter_set

    query = 'SELECT md5, count FROM md5_values WHERE count > ? ORDER BY count DESC'
    results = db.execute(query, (max_count,)).fetchall()

    filter_set = set(r[0].hex() for r in results)
    counts = list(r[1] for r in results)

    total = sum(counts)

    print(f'Filtering {len(filter_set)} hashes. Num Images: {total}. Top 10 Counts: {counts[:10]}', file=sys.stderr)

    _FILTER_MAP[id(db)] = filter_set
    return filter_set
