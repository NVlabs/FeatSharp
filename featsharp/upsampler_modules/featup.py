# https://github.com/mhamilton723/FeatUp
"""
Implemented the FeatUp JBU upsampler (http://arxiv.org/abs/2403.10516)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_conv_cuda.adaptive_conv import AdaptiveConv
from featsharp.upsampler_modules.hadamard import get_prime_factors

class JBULevel(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))
        self.range_alpha = nn.Parameter(torch.tensor(0.0))
        self.spatial_alpha = nn.Parameter(torch.tensor(0.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return self.range_alpha.exp() * F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self, device):
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        y, x = torch.meshgrid(dist_range, dist_range, indexing='ij')
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return self.spatial_alpha.exp() * torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel(source.device)
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel = combined_kernel / combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = F.interpolate(source, (GH, GW), mode='bilinear', align_corners=False)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        result =  AdaptiveConv.apply(hr_source_padded, combined_kernel)
        return result

class JBUStack(torch.nn.Module):

    def __init__(self, feat_dim: int, upsample_factor: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample_factors = get_prime_factors(upsample_factor)
        self.upsample_factor = upsample_factor
        self.num_levels = len(self.upsample_factors)

        self.ups = nn.ModuleList([
            JBULevel(3, feat_dim, 32, scale=self.upsample_factors[i], radius=3)
            for i in range(self.num_levels)
        ])

    def upsample(self, source, guidance, up, factor):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * factor, w * factor))
        upsampled = up(source, small_guidance)
        return upsampled

    def forward(self, source, guidance):
        for up, factor in zip(self.ups, self.upsample_factors):
            source = self.upsample(source, guidance, up, factor)
        return source
