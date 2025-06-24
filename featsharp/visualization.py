# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from contextlib import ContextDecorator
import os
from PIL import Image
from typing import Dict, Any, Optional

import cv2
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter as TBWriter
from torchvision.utils import make_grid

from featsharp.data.transforms import generate_homography_grid
from featsharp.util import get_rank, pca


class VizWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, name: str, value: float, step: int):
        pass

    def add_image(self, name: str, value: torch.Tensor, step: int):
        pass

    def add_histogram(self, name: str, value: torch.Tensor, step: int):
        pass

    def commit(self, step: int):
        pass

    def state_dict(self) -> Dict[str, Any]:
        return dict()


class TBVizWriter(VizWriter):
    def __init__(self, log_dir: str, **kwargs):
        self.writer = TBWriter(log_dir)

    def add_scalar(self, name: str, value: float, step: int):
        self.writer.add_scalar(name, value, global_step=step)

    def add_image(self, name: str, value: torch.Tensor, step: int):
        self.writer.add_image(name, value, global_step=step)

    def add_histogram(self, name: str, value: torch.Tensor, step: int):
        self.writer.add_histogram(name, value, global_step=step)

    def commit(self, step: int):
        self.writer.flush()


class WandbVizWriter(VizWriter):
    def __init__(self, entity: str, project: str, group: str = None, job_type: str = None,
                 config: Optional[dict] = None,
                 state_dict: Optional[Dict[str, Any]] = None,
                 log_dir: str = None,
                 **kwargs):
        import wandb
        import wandb.util

        job_id = state_dict.get('id', None) if state_dict is not None else None
        if job_id is None:
            job_id = wandb.util.generate_id()

        if log_dir is not None:
            log_dir = os.path.join(log_dir, 'wandb')
            os.makedirs(log_dir, exist_ok=True)

        self.job_id = job_id
        self.wandb = wandb
        self.run = wandb.init(
            entity=entity,
            project=project,
            group=group,
            job_type=job_type,
            id=job_id,
            resume='allow',
            config=config,
            dir=log_dir,
        )

    def add_scalar(self, name: str, value: float, step: int):
        self.run.log({name: value}, step=step, commit=False)

    def add_image(self, name: str, value: torch.Tensor, step: int):
        if torch.is_tensor(value):
            if value.ndim == 3 and value.shape[0] in (1, 3):
                value = value.permute(1, 2, 0)
                if value.shape[2] == 1:
                    value = value.squeeze(2)
            if value.dtype in (torch.float16, torch.float32, torch.bfloat16):
                if value.amax() <= 1:
                    value = value * 255
                value = value.to(torch.uint8)
            value = value.cpu().numpy()
        value = Image.fromarray(value)
        im = self.wandb.Image(value)
        self.run.log({name: im}, step=step, commit=False)

    def add_histogram(self, name: str, value: torch.Tensor, step: int):
        hist = self.wandb.Histogram(value.cpu())
        self.run.log({name: hist}, step=step, commit=False)

    def commit(self, step: int):
        self.run.log({}, step=step, commit=True)

    def state_dict(self) -> Dict[str, Any]:
        return dict(id=self.job_id)


class append_kwargs(ContextDecorator):
    def __init__(self, obj, **kwargs_to_append):
        self.obj = obj
        self.kwargs_to_append = kwargs_to_append
        self.original_methods = {}

    def __enter__(self):
        # Wrap each callable attribute in the object with a function that appends kwargs
        for attr_name in dir(self.obj):
            if attr_name.startswith('__'):
                continue
            attr = getattr(self.obj, attr_name)
            if callable(attr):
                self.original_methods[attr_name] = attr
                wrapped = self._wrap_method(attr)
                setattr(self.obj, attr_name, wrapped)
        return self

    def __exit__(self, *exc):
        # Restore original methods
        for attr_name, original_method in self.original_methods.items():
            setattr(self.obj, attr_name, original_method)
        self.original_methods.clear()

    def _wrap_method(self, method):
        def wrapper(*args, **kwargs):
            combined_kwargs = {**kwargs, **self.kwargs_to_append}
            return method(*args, **combined_kwargs)
        return wrapper


def get_visualizer(cfg: Dict, **kwargs):
    viz_type = cfg['type']
    args = {k: v for k, v in cfg.items() if k != 'type'}

    if viz_type in ('null', 'none') or not viz_type or get_rank() > 0:
        ctor = VizWriter
    elif viz_type == 'wandb':
        ctor = WandbVizWriter
    elif viz_type == 'tensorboard':
        ctor = TBVizWriter
    else:
        raise ValueError(f'Unsupported viz type: {viz_type}')

    return ctor(**kwargs, **args)


def _id(x):
    return x


def visualize_transform(src_buffer: torch.Tensor, tgt_buffer: torch.Tensor, transform: torch.Tensor, filename: str, denorm = _id):
    tgt_grid = generate_homography_grid(transform, tgt_buffer.shape).to(tgt_buffer.dtype)

    valid_src = torch.ones_like(src_buffer)[:, :1]
    src_in_tgt = F.grid_sample(src_buffer, tgt_grid, mode='bilinear', align_corners=True)
    valid_src = F.grid_sample(valid_src, tgt_grid, mode='nearest', align_corners=True)

    if src_buffer.shape[1] != 3:
        tgt_pca, src_in_tgt_pca = [], []
        for tgt, src in zip(tgt_buffer, src_in_tgt):
            tgt2, fit_pca = pca([tgt[None]])
            tgt_pca.append(tgt2[0])

            src2, _ = pca([src[None]], fit_pca=fit_pca)
            src_in_tgt_pca.append(src2[0])
        tgt_pca = torch.cat(tgt_pca, dim=0)
        src_in_tgt_pca = torch.cat(src_in_tgt_pca, dim=0)
        src_in_tgt_pca *= valid_src
    else:
        tgt_pca = tgt_buffer
        src_in_tgt_pca = src_in_tgt

    overlay = denorm(src_in_tgt_pca * 0.5 + tgt_pca * 0.5)

    overlay = make_grid(overlay)

    overlay = overlay.mul_(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    dn = os.path.dirname(filename)
    if dn:
        os.makedirs(dn, exist_ok=True)

    cv2.imwrite(filename, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    pass


def visualize_grid(buffer: torch.Tensor, filename: str, denorm = _id):
    if buffer.shape[1] > 3:
        buffer_pca, fit_pca = pca([buffer])
        buffer = buffer_pca[0]
    else:
        fit_pca = None

    buffer = denorm(buffer)

    grid = make_grid(buffer)

    grid = grid.mul_(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    if fit_pca is not None:
        grid2, _ = pca([grid], fit_pca=fit_pca)
        grid = grid2[0]

    dn = os.path.dirname(filename)
    if dn:
        os.makedirs(dn, exist_ok=True)

    if grid.shape[-1] == 3:
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    else:
        grid = grid[..., 0]

    cv2.imwrite(filename, grid)
