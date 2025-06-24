# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import time
from typing import List, Optional
import torch
from torch.utils import cpp_extension
import os


_EXT = None
def get_ext():
    global _EXT
    if _EXT is None:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        _prev_cuda_flags_fn = cpp_extension._get_cuda_arch_flags
        def _monkey_cuda_flags(cflags: Optional[List[str]] = None) -> List[str]:
            if cflags:
                return _prev_cuda_flags_fn(cflags)
            return []
        cpp_extension._get_cuda_arch_flags = _monkey_cuda_flags

        # Load the CUDA extension
        _EXT = cpp_extension.load(
            name='mmd_extension_cuda',
            sources=[os.path.join(current_dir, 'mmd_ops.cpp'),
                    os.path.join(current_dir, 'mmd.cu')],
            extra_cflags=['-O3'],
            extra_cuda_cflags=[
                '-O3',
                '-gencode=arch=compute_60,code=sm_60',
                '-gencode=arch=compute_61,code=sm_61',
                '-gencode=arch=compute_70,code=sm_70',
                '-gencode=arch=compute_72,code=sm_72',
                '-gencode=arch=compute_75,code=sm_75',
                '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_86,code=sm_86',
                '-gencode=arch=compute_87,code=sm_87',
                '-gencode=arch=compute_90,code=compute_90',
                '-gencode=arch=compute_90,code=sm_90',
            ],
            verbose=True
        )
        cpp_extension._get_cuda_arch_flags = _prev_cuda_flags_fn
    return _EXT


class PartialMMDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gamma, include_diag):
        # x, y: float tensors, shape (n, d), (m, d)
        # gamma: float
        # We'll get [out] = [MMD scalar]
        out = get_ext().mmd_forward(x, y, gamma, include_diag)
        # Save for backward
        ctx.save_for_backward(x, y, gamma)
        ctx.include_diag = include_diag
        return out[0]

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is scalar gradient from upstream
        x, y, gamma = ctx.saved_tensors
        grad_x, grad_y = get_ext().mmd_backward(x, y, grad_output, gamma, ctx.include_diag)
        # None for gamma, since we don't compute gradient wrt gamma
        return grad_x, grad_y, None, None


def partial_mmd(x, y, gamma, include_diag):
    """
    High-level convenience function that applies MMDFunction.
    Returns a scalar tensor.
    """
    return PartialMMDFunction.apply(x, y, gamma, include_diag)


@torch.jit.script
def brute_partial_mmd(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor, include_diag: bool):
    mmd_ab = torch.cdist(x, y, p=2.0).pow(2).mul(-gamma).exp()
    if include_diag:
        return mmd_ab.mean()

    if mmd_ab.requires_grad:
        eye = torch.eye(n=mmd_ab.shape[0], m=mmd_ab.shape[1], dtype=torch.bool, device=mmd_ab.device)
        mmd_ab = torch.where(eye, 0, mmd_ab)
    else:
        mmd_ab.fill_diagonal_(0)

    mmd_ab = mmd_ab.div((x.shape[0] - 1) * y.shape[0])
    return mmd_ab.sum()


def _get_gamma(x: torch.Tensor, gamma: torch.Tensor | float | None):
    if gamma is None:
        xx = torch.cdist(x.detach(), x.detach(), p=2)
        med = torch.median(xx.flatten())
        gamma = 1.0 / (2 * med * med)

    gamma = torch.as_tensor(gamma, dtype=x.dtype, device=x.device)
    return gamma


def _mmd(fn, x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor | float | None):
    gamma = _get_gamma(x, gamma)

    mmd_xx = fn(x, x, gamma, include_diag=False)
    mmd_yy = fn(y, y, gamma, include_diag=False)
    mmd_xy = fn(x, y, gamma, include_diag=True)

    mmd = mmd_xx + mmd_yy - 2 * mmd_xy
    return mmd


def mmd(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor | float | None = None):
    return _mmd(partial_mmd, x, y, gamma)


def brute_force_mmd(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor | float | None = None):
    return _mmd(brute_partial_mmd, x, y, gamma)


if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.rand(4000, 128, dtype=torch.float64, device='cuda')
    y = torch.rand(4000, 128, dtype=torch.float64, device='cuda')

    brute_mmd = brute_force_mmd(x, y)
    fast_mmd = mmd(x, y)

    print(f'Brute: {brute_mmd.item()}')
    print(f'Fast: {fast_mmd.item()}')

    assert torch.allclose(brute_mmd, fast_mmd), f'Invalid fast mmd!'

    gamma = torch.tensor(0.02, dtype=torch.float64, device='cuda')
    x.requires_grad_(True)
    y.requires_grad_(True)
    torch.autograd.gradcheck(mmd, (x, y, gamma), nondet_tol=1e-8, fast_mode=True)
    print(f'Gradients are correct!')

    NUM_ITERS = 1000
    def _timeit(fn, *args):
        fn(*args)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(NUM_ITERS):
            y = fn(*args)
            y.backward()
        torch.cuda.synchronize()
        end_time = time.time()
        return (end_time - start_time) / NUM_ITERS

    brute_time = _timeit(brute_force_mmd, x, y)
    fast_time = _timeit(mmd, x, y)

    print(f'Brute Time: {brute_time * 1000:.1f}ms/call')
    print(f'Fast Time: {fast_time * 1000:.1f}ms/call')
