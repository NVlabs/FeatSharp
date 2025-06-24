// -----------------------------------------------------------------------------
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// This source code is licensed under the NSCLv2
// found in the LICENSE file in the root directory of this source tree.
// -----------------------------------------------------------------------------
#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA kernels (defined in mmd_cuda.cu)
torch::Tensor mmd_forward_cuda(
    torch::Tensor x, torch::Tensor y, torch::Tensor gamma, bool include_diag);

std::vector<torch::Tensor> mmd_backward_cuda(
    torch::Tensor x, torch::Tensor y,
    torch::Tensor grad_output,  // scalar gradient wrt MMD
    torch::Tensor gamma,
    bool include_diag);

// --------------- Forward ---------------
torch::Tensor mmd_forward(
    torch::Tensor x, torch::Tensor y, torch::Tensor gamma, bool include_diag)
{
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");

    return mmd_forward_cuda(x, y, gamma, include_diag);
}

// --------------- Backward ---------------
std::vector<torch::Tensor> mmd_backward(
    torch::Tensor x, torch::Tensor y,
    torch::Tensor grad_output,  // gradient from upstream
    torch::Tensor gamma,
    bool include_diag)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");

    return mmd_backward_cuda(x, y, grad_output, gamma, include_diag);
}

// Bind to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mmd_forward", &mmd_forward, "MMD forward (CUDA)");
    m.def("mmd_backward", &mmd_backward, "MMD backward (CUDA)");
}
