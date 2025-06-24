# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from functools import partialmethod
import types

from mmengine.registry import OPTIMIZERS

_inner_fn = OPTIMIZERS._register_module

def war_register_module(self, *args, **kwargs):
    try:
        _inner_fn(*args, **kwargs)
    except KeyError as e:
        em = str(e)
        if 'Adafactor' in em:
            return
        raise

OPTIMIZERS._register_module = types.MethodType(war_register_module, OPTIMIZERS)
