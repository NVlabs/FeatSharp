# https://github.com/mhamilton723/FeatUp
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAPAModule(nn.Module):
    def __init__(self, dim_y, dim_x=None,
                 up_factor=2, up_kernel_size=5, embedding_dim=64,
                 qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y

        self.up_factor = up_factor
        self.up_kernel_size = up_kernel_size
        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x)

        self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        self.k = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)

        self.apply(self._init_weights)

    def forward(self, y, x):
        y = y.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm_y(y)
        x_ = self.norm_x(x)

        q = self.q(y)
        k = self.k(x_)

        return self.attention(q, k, x).permute(0, 3, 1, 2).contiguous()

    def attention(self, q, k, v):
        from sapa import sim, atn

        attn = F.softmax(
            sim(q.contiguous(), k.contiguous(), self.up_kernel_size, self.up_factor),
            dim=-1
        )
        return atn(attn.contiguous(), v.contiguous(), self.up_kernel_size, self.up_factor)

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class SAPAUpsampler(torch.nn.Module):
    def __init__(self, dim_x, depth: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ups = nn.ModuleList([
            SAPAModule(dim_x=dim_x, dim_y=3)
            for _ in range(depth)
        ])

    def adapt_guidance(self, source, guidance):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return small_guidance

    def forward(self, source, guidance):
        x = source.contiguous()
        for level in self.ups:
            ada = self.adapt_guidance(x, guidance).contiguous()
            x = level(ada, x)
        return x
