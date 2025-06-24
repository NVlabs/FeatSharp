# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from typing import List, Optional
from torch.autograd import Function
import torch
from torch.utils import cpp_extension
import os

torch.manual_seed(42)

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
            name='adaptive_conv_cpp',
            sources=[
                os.path.join(current_dir, 'adaptive_conv_cuda.cpp'),
                os.path.join(current_dir, 'adaptive_conv_kernel.cu'),
            ],
            extra_cflags=['-O3'],
            extra_cuda_cflags=[
                '-O3',
                # '-gencode=arch=compute_60,code=sm_60',
                # '-gencode=arch=compute_61,code=sm_61',
                # '-gencode=arch=compute_70,code=sm_70',
                # '-gencode=arch=compute_72,code=sm_72',
                # '-gencode=arch=compute_75,code=sm_75',
                # '-gencode=arch=compute_80,code=sm_80',
                # '-gencode=arch=compute_86,code=sm_86',
                # '-gencode=arch=compute_87,code=sm_87',
                # '-gencode=arch=compute_90,code=compute_90',
                # '-gencode=arch=compute_90,code=sm_90',
            ],
            verbose=True
        )
        cpp_extension._get_cuda_arch_flags = _prev_cuda_flags_fn
    return _EXT


class AdaptiveConv(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.save_for_backward(filters, input)
        b, h2, w2, f1, f2 = filters.shape
        assert f1 == f2
        assert input.is_cuda and filters.is_cuda

        result = get_ext().forward(input, filters)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        filters, input = ctx.saved_tensors
        grad_input = grad_filters = None
        b, h2, w2, f1, f2 = filters.shape
        assert f1 == f2
        assert grad_output.is_cuda and input.is_cuda and filters.is_cuda

        grad_output = grad_output.contiguous()

        ext = get_ext()
        if ctx.needs_input_grad[0]:
            grad_input = ext.grad_input(grad_output, filters)
        if ctx.needs_input_grad[1]:
            grad_filters = ext.grad_filters(grad_output, input)

        return grad_input, grad_filters
