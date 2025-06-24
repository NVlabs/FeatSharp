# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import torch
from torch import distributed as dist

from .HFHub import get_model as get_hfhub_model

def get_featurizer(name, activation_type="key", **kwargs):
    name = name.lower()
    if name == "vit":
        from .DINO import DINOFeaturizer
        patch_size = 16
        model = DINOFeaturizer("vit_small_patch16_224", patch_size, activation_type)
        model.input_size = 224
        dim = 384
    elif name == "dinov2":
        from .DINOv2 import DINOv2Featurizer
        model = DINOv2Featurizer(**kwargs)
        patch_size = model.patch_size
        dim = model.embed_dim
    elif name == "open_clip":
        from .OpenCLIP import OpenCLIPFeaturizer
        model = OpenCLIPFeaturizer(**kwargs)
        patch_size = model.patch_size
        dim = model.embed_dim
    elif name == "hfhub":
        model = get_hfhub_model(**kwargs)
        patch_size = model.patch_size
        dim = model.embed_dim
    elif name == "radio":
        from .radio import RADIOFeaturizer
        kwargs.pop('num_classes', None)
        model = RADIOFeaturizer(**kwargs)
        patch_size = model.patch_size
        dim = model.embed_dim
    elif name == 'siglip2':
        from .SigLIP2 import get_siglip2_model
        kwargs.pop('num_classes', None)
        model = get_siglip2_model(**kwargs)
        patch_size = model.patch_size
        dim = model.embed_dim
    elif name == "sam":
        from .SAM import SAMFeaturizer
        kwargs.pop('num_classes', None)
        model = SAMFeaturizer(**kwargs)
        patch_size = model.patch_size
        dim = model.embed_dim
    else:
        raise ValueError("unknown model: {}".format(name))
    return model, patch_size, dim
