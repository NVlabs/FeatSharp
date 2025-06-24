// -----------------------------------------------------------------------------
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// This source code is licensed under the NSCLv2
// found in the LICENSE file in the root directory of this source tree.
// -----------------------------------------------------------------------------
#include <torch/extension.h>
#include <vector>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// ---------------------------------------------
// Kernel: MMD Forward
// ---------------------------------------------
template <bool include_diag, typename scalar_t>
__global__ void mmd_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> x,  // [n, d]
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y,  // [m, d]
    scalar_t *p_sum,  // blockwise partial results
    scalar_t *p_gamma)
{
    __shared__ scalar_t s_partial[8];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int n = x.size(0);
    const int m = y.size(0);
    const int d = x.size(1);

    const int tid = blockIdx.y * blockDim.y + threadIdx.y;
    const scalar_t gamma = *p_gamma;
    scalar_t sum_local = 0.0;

    for (int pair = tid; pair < n * m; pair += (blockDim.y * gridDim.y)) {
        const int row = pair / m;
        const int col = pair % m;

        if (!include_diag && row == col) { continue; }

        scalar_t dist2 = 0.0;
        for (int k = warp.thread_rank(); k < d; k += warp.size()) {
            const scalar_t diff = x[row][k] - y[col][k];
            dist2 += diff * diff;
        }

        // Warp reduce to lane 0
        dist2 = cg::reduce(warp, dist2, cg::plus<scalar_t>());

        if (warp.thread_rank() == 0) {
            const scalar_t rbfk = exp(-gamma * dist2);
            sum_local += rbfk;
        }
    }

    if (warp.thread_rank() == 0) {
        s_partial[threadIdx.y] = sum_local;
    }

    __syncthreads();

    // TODO: Make this parallel
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 1; i < blockDim.y; ++i) {
            sum_local += s_partial[i];
        }

        const scalar_t div = include_diag ? (n * m) : ((n - 1) * m);
        sum_local /= div;
        atomicAdd(p_sum, sum_local);
    }
}

template <bool include_diag, typename scalar_t>
__global__ void mmd_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> x,   // [n, d]
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y,   // [m, d]
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_x,    // [n, d], accumulates
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_y,    // [m, d], accumulates
    scalar_t *p_grad_output,
    scalar_t *p_gamma)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int n = x.size(0);
    const int m = y.size(0);
    const int d = x.size(1);

    const int tid = blockIdx.y * blockDim.y + threadIdx.y;
    const scalar_t gamma = *p_gamma;
    const scalar_t grad_out = *p_grad_output;
    const scalar_t div = include_diag ? (n * m) : ((n - 1) * m);
    const scalar_t const_mul = -2.0 * gamma * grad_out / div;

    const int slot = tid % 8;

    for (int pair = tid; pair < n*m; pair += blockDim.y * gridDim.y) {
        const int row = pair / m;
        const int col = pair % m;

        if (!include_diag && row == col) { continue; }

        scalar_t dist2 = 0.0;
        for (int k = warp.thread_rank(); k < d; k += warp.size()) {
            const scalar_t diff = x[row][k] - y[col][k];
            dist2 += diff * diff;
        }

        // Warp reduce to lane 0
        dist2 = cg::reduce(warp, dist2, cg::plus<scalar_t>());
        // // Broadcast to all lanes
        dist2 = warp.shfl(dist2, 0);

        const scalar_t rbfk = exp(-gamma * dist2);
        const scalar_t grad_c = rbfk * const_mul;

        for (int k = warp.thread_rank(); k < d; k += warp.size()) {
            scalar_t diff = x[row][k] - y[col][k];
            scalar_t gx = grad_c * diff;
            scalar_t gy = -gx;

            atomicAdd(&grad_x[row][slot][k], gx);
            atomicAdd(&grad_y[col][slot][k], gy);
        }
    }
}

torch::Tensor mmd_forward_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor gamma, bool include_diag)
{
    auto n = x.size(0);
    auto m = y.size(0);
    auto d = x.size(1);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto out = torch::zeros({1}, options);

    dim3 threads{32, 8};

    int grid_height = (n * m + threads.y - 1) / threads.y;
    dim3 grid{1, std::min<uint32_t>(grid_height, 65535)};

    // typedef float scalar_t;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "mmd_forward_cuda", ([&] {
        auto fn = include_diag ? mmd_forward_kernel<true, scalar_t> : mmd_forward_kernel<false, scalar_t>;
        fn<<<grid, threads>>>(
            x.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            out.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>()
        );
    }));

    return out;
}

std::vector<torch::Tensor> mmd_backward_cuda(
    torch::Tensor x, torch::Tensor y,
    torch::Tensor grad_output, torch::Tensor gamma,
    bool include_diag)
{
    auto n = x.size(0);
    auto m = y.size(0);
    auto d = x.size(1);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto grad_x = torch::zeros({(long)n, 8l, (long)d}, options);
    auto grad_y = torch::zeros({(long)m, 8l, (long)d}, options);

    dim3 threads{32, 8};

    int grid_height = (n * m + threads.y - 1) / threads.y;
    dim3 grid{1, std::min<uint32_t>(grid_height, 65535)};

    // typedef float scalar_t;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "mmd_backward_cuda", ([&] {
        auto fn = include_diag ? mmd_backward_kernel<true, scalar_t> : mmd_backward_kernel<false, scalar_t>;
        fn<<<grid, threads>>>(
            x.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            grad_x.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            grad_y.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            grad_output.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>()
        );
    }));

    return { grad_x.sum(1), grad_y.sum(1) };
}
