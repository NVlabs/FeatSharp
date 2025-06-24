# https://github.com/zmhhmz/ReSFU
from typing import Callable, Tuple
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from .util import create_tiles, cumprod

try:
    from FNS_Attn import fns_attn
except ImportError:
    warnings.warn("FNS_Attn module not found. Please ensure it is available in your environment.")
    fns_attn = None


def create_gaussian_kernel(kernel_size, sigma=1):
    # Create a 2D Gaussian kernel for generating q_GS
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel

class GuidedFilter(nn.Module):
    def __init__(self, radius, epsilon, channels, scale=1):
        super(GuidedFilter, self).__init__()
        self.radius = radius//scale
        self.epsilon = epsilon
        self.scale=scale
        kernel = torch.ones(channels, 1, 1, 2 * self.radius + 1)
        self.register_buffer('kernel', kernel)

    def forward(self, I, p):

        if self.scale>1:
            I0 = I.clone()
            I = F.interpolate(I, scale_factor=1./self.scale, mode="nearest")
            p = F.interpolate(p, scale_factor=1./self.scale, mode="nearest")

        N = self.box_filter_separable(torch.ones_like(I))

        mean_I = self.box_filter_separable(I) / N
        mean_p = self.box_filter_separable(p) / N
        mean_Ip = self.box_filter_separable(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter_separable(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.epsilon)
        b = mean_p - a * mean_I

        mean_a = self.box_filter_separable(a) / N
        mean_b = self.box_filter_separable(b) / N

        if self.scale>1:
            mean_a = F.interpolate(mean_a, I0.shape[-2:], mode='bilinear')
            mean_b = F.interpolate(mean_b, I0.shape[-2:], mode='bilinear')
            q = mean_a * I0 + mean_b
        else:
            q = mean_a * I + mean_b
        return q

    def box_filter_separable(self, x):
        x = F.conv2d(x, self.kernel, padding=(0, self.radius), groups=x.size(1))
        x = F.conv2d(x, self.kernel.transpose(2, 3), padding=(self.radius, 0), groups=x.size(1))
        return x


class PCDC_Block(nn.Module):
    def __init__(self,in_channels,scale_factor,mid_channels=128,kernel_size=3,groups=4,gn_dim=8):
        super(PCDC_Block,self).__init__()
        out_channels=kernel_size**2
        self.dilation=scale_factor
        self.groups=groups
        self.norm=nn.GroupNorm(gn_dim,in_channels)
        self.channel_compressor=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(mid_channels,mid_channels,1,groups=groups),
            nn.ReLU(True),
            nn.GroupNorm(gn_dim,mid_channels),
            nn.Conv2d(mid_channels,out_channels, 1)
        )
        self.padding=(self.dilation*(kernel_size-1))//2
        self.weight=nn.Parameter(torch.Tensor(mid_channels, in_channels//groups, kernel_size, kernel_size))
        self.bias=nn.Parameter(torch.Tensor(1,mid_channels,1,1))
        nn.init.kaiming_normal_(self.weight,mode='fan_out',nonlinearity='relu')
        torch.nn.init.zeros_(self.bias)

    def forward(self, k, q):
        # GroupNorm with shared affine weights
        q=self.norm(q)
        k=self.norm(k)
        # PCDC layer
        k=F.pad(k,(self.padding,self.padding,self.padding,self.padding),mode='replicate')
        k=F.conv2d(k,self.weight,dilation=self.dilation,groups=self.groups)
        kernel_diff=self.weight.sum(-2,keepdim=True).sum(-1,keepdim=True)
        q=F.conv2d(q,kernel_diff,groups=self.groups)
        v=k-q+self.bias
        # channel compressor
        return self.channel_compressor(v)


class ReSFU(nn.Module):
    def __init__(self, dim_y, dim_x=None, groups=4, kernel_size=3, scale_factor=4, embedding_dim=32,
                 y_conv=True, qkv_bias=True, normx=True, normy=True,
                 gn_dim=8, radius=8, eps=0.001, lr_mult=1., gf_scale=2):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y
        self.dim_x = dim_x
        self.scale_factor = scale_factor

        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.normx = normx
        self.normy = normy

        if self.normx:
            if dim_x>=32:
                self.norm_x = nn.GroupNorm(32, dim_x)
            else:
                self.norm_x = nn.GroupNorm(dim_x, dim_x)

        if self.normy:
            if dim_y>=32:
                self.norm_y = nn.GroupNorm(32, dim_y)
            else:
                self.norm_y = nn.GroupNorm(dim_y, dim_y)

        self.GF = GuidedFilter(radius,eps,embedding_dim,scale=gf_scale)

        if y_conv:
            self.q = nn.Conv2d(dim_y, embedding_dim, kernel_size=1, bias=qkv_bias)
        else:
            assert dim_y==embedding_dim
            self.q = nn.Identity()
        self.k = nn.Conv2d(dim_x, embedding_dim, kernel_size=1, bias=qkv_bias)

        self.blur_kernels3 = create_gaussian_kernel(3).view(1, 1, 3, 3).cuda()

        self.semantic_scores = PCDC_Block(embedding_dim, scale_factor, kernel_size=kernel_size, groups=groups, gn_dim=gn_dim)
        self.detail_scores = PCDC_Block(embedding_dim, scale_factor, kernel_size=kernel_size, groups=groups, gn_dim=gn_dim)
        for name, param in self.semantic_scores.named_parameters():
            param.register_hook(lambda grad: grad * lr_mult)
        for name, param in self.detail_scores.named_parameters():
            param.register_hook(lambda grad: grad * lr_mult)
        self.apply(self._init_weights)
        self.vis_x = nn.Identity()
        self.vis_x_up = nn.Identity()

    def forward(self, y, x):
        bs = x.size(0)
        # temp = self.vis_x(F.interpolate(x,scale_factor=self.scale_factor,mode='nearest'))

        if self.normx:
            x_ = self.norm_x(x)
        else:
            x_ = x

        if self.normy:
            y = self.norm_y(y)

        q = self.q(y)
        k = self.k(x_)
        k_ = F.interpolate(k, size=q.size()[-2:], mode='bilinear')

        q_gs = F.pad(q, (1, 1, 1, 1), mode='replicate')
        q_gs = F.conv2d(q_gs, self.blur_kernels3.repeat(q.shape[1],1,1,1), groups=q.shape[1])

        q_gf = self.GF(q, k_)

        semantic_scores = self.semantic_scores(k_, q_gf)
        detail_scores = self.detail_scores(q_gs, q)

        scores = semantic_scores+detail_scores
        # for pixels out of range, remove their influences to the softmax values
        zeros = scores.new_zeros(bs, 1, *q.size()[-2:])
        pad_num = (self.scale_factor*(self.kernel_size-1))//2
        zeros = F.pad(zeros, (pad_num, pad_num, pad_num, pad_num), value=-1e7)
        zeros = F.unfold(zeros, kernel_size=self.kernel_size, padding=0, dilation=self.scale_factor).view(*scores.size())
        attn = F.softmax(scores+zeros, dim=1)

        out = fns_attn(attn.contiguous(), x.contiguous(), self.kernel_size)
        # temp = self.vis_x_up(out)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ReSFUUpsampler(nn.Module):
    def __init__(self, model_resolution: int, dim: int, upsample_factor: int):
        super().__init__()
        self.model_resolution = model_resolution
        self.upsample_factor = upsample_factor

        self.inner = ReSFU(dim_y=3, dim_x=dim, scale_factor=upsample_factor, normx=False, normy=False)

    def forward(self, source, guidance):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * self.upsample_factor, w * self.upsample_factor))
        ups = self.inner(small_guidance, source)
        return ups


class ReSFUTiledUpsampler(nn.Module):
    def __init__(self, model_resolution: int, dim: int, upsample_factor: int):
        super().__init__()
        self.model_resolution = model_resolution
        self.upsample_factors = [upsample_factor]
        self.num_levels = len(self.upsample_factors)
        self.upsample_factor = upsample_factor

        ress = cumprod(self.upsample_factors, inclusive=False)

        self.ups = nn.ModuleList([
            ReSFU(dim_y=3 + dim, dim_x=dim, scale_factor=uf, normx=False, normy=False)
            for resf, uf in zip(ress, self.upsample_factors)
        ])

    @property
    def hi_res_input(self) -> bool:
        return True

    @property
    def input_upsample_factor(self) -> int:
        return self.upsample_factor

    def forward(self, guidance: torch.Tensor, tile_generator: Callable[[torch.Tensor], torch.Tensor],
                return_summary: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tlop = create_tiles(guidance, tile_generator, self.model_resolution, [self.upsample_factor],
                            return_summary=return_summary,
        )

        if return_summary:
            summary, tile_levels, _ = tlop
        else:
            tile_levels, _ = tlop

        lr_source = source = tile_levels[0]
        for i in range(1, len(tile_levels)):
            tiles = tile_levels[i]
            small_guidance = F.adaptive_avg_pool2d(guidance, (tiles.shape[2], tiles.shape[3]))
            guide = torch.cat([small_guidance, tiles], dim=1)
            source = self.ups[i - 1](guide, source)

        if return_summary:
            return summary, lr_source, source
        return lr_source, source
