from collections import defaultdict, deque
import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torchvision.transforms as T


class RollingAvg:

    def __init__(self, length):
        self.length = length
        self.metrics = defaultdict(lambda: deque(maxlen=self.length))

    def add(self, name, metric):
        self.metrics[name].append(metric)

    def get(self, name):
        return torch.tensor(list(self.metrics[name]), dtype=torch.float32).mean().item()

    def logall(self, log_func):
        for k in self.metrics.keys():
            log_func(k, self.get(k))


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean))
        self.register_buffer('std', torch.as_tensor(std))

    def forward(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X: torch.Tensor):
        self.mean_ = X.mean(dim=0, keepdim=True)
        unbiased = X - self.mean_
        cov = unbiased.T @ unbiased / (unbiased.shape[1] - 1)
        e, U = torch.linalg.eigh(cov)
        self.components_ = torch.flip(U[:, -self.n_components:], dims=(1,))
        # self.singular_values_ = e[-self.n_components:]
        # Standardize the direction to be positive along the diagonal
        self.components_ = torch.where(self.components_.diag()[None] < 0, -self.components_, self.components_)
        return self

    def transform(self, X: torch.Tensor):
        t0 = X - self.mean_
        projected = t0 @ self.components_
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None, rand_rot_seed: Optional[int] = None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear", align_corners=False)
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)

        if rand_rot_seed is not None:
            g = torch.Generator(device=device).manual_seed(rand_rot_seed)
            rr = torch.randn(dim, dim, device=device, dtype=x_red.dtype, generator=g)
            rr, _ = torch.linalg.qr(rr)
            x_red = x_red @ rr

        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


def prep_image(t, subtract_min=True):
    if subtract_min:
        t -= t.min()
    t /= t.max()
    t = (t * 255).clamp(0, 255).to(torch.uint8)

    if len(t.shape) == 2:
        t = t.unsqueeze(0)

    return t


def extract_normalize(preproc):
    if isinstance(preproc, T.Compose):
        for sub in preproc.transforms:
            ret = extract_normalize(sub)
            if ret is not None:
                return ret
        return None
    elif isinstance(preproc, T.Normalize):
        return preproc
    return None


def get_rank(group = None):
    return dist.get_rank(group) if dist.is_initialized() else 0


def get_world_size(group = None):
    return dist.get_world_size(group) if dist.is_initialized() else 1


def barrier(group = None):
    if dist.is_initialized():
        dist.barrier(group)


class rank_gate:
    '''
    Execute the function on rank 0 first, followed by all other ranks. Useful when caches may need to be populated in a distributed environment.
    '''
    def __init__(self, func = None):
        self.func = func

    def __call__(self, *args, **kwargs):
        rank = get_rank()
        if rank == 0:
            result = self.func(*args, **kwargs)
        barrier()
        if rank > 0:
            result = self.func(*args, **kwargs)
        return result

    def __enter__(self, *args, **kwargs):
        if get_rank() > 0:
            barrier()

    def __exit__(self, *args, **kwargs):
        if get_rank() == 0:
            barrier()


def seed_everything(seed: int, workers: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DistributedDataParallel(NativeDDP):
    def __getattr__(self, name: str) -> torch.Tensor | nn.Module:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def norm_of_tensors(tensors: List[torch.Tensor], norm_type: float = 2.0):
    if not tensors:
        return 0

    norms: List[torch.Tensor] = []
    norms.extend(torch._foreach_norm(tensors, norm_type))

    total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    return total_norm


def all_close(a, b):
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(all_close(a1, b1) for a1, b1 in zip(a, b))
    elif isinstance(a, dict):
        if len(a) != len(b):
            return False
        return all(all_close(a[k], b[k]) for k in a)
    elif torch.is_tensor(a):
        return torch.allclose(a, b)
    return a == b


def to_device(val, *args, **kwargs):
    if torch.is_tensor(val):
        return val.to(*args, **kwargs)
    elif isinstance(val, (tuple, list)):
        return [to_device(v2, *args, **kwargs) for v2 in val]
    elif isinstance(val, dict):
        return {k: to_device(v, *args, **kwargs) for k, v in val.items()}
    return val
