# https://github.com/mhamilton723/FeatUp
import math
import torch
from torch import nn
import torch.nn.functional as F


class SimpleDownsampler(torch.nn.Module):

    def get_kernel(self):
        k = F.softplus(self.kernel_params.unsqueeze(0).unsqueeze(0))
        k /= k.sum()
        return k

    def __init__(self, kernel_size, final_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.kernel_params = torch.nn.Parameter(torch.ones(kernel_size, kernel_size))

    def forward(self, imgs, guidance):
        b, c, h, w = imgs.shape
        input_imgs = imgs.reshape(b * c, 1, h, w)
        stride = h // self.final_size

        padding = self.kernel_size // 2
        ret = F.conv2d(input_imgs, self.get_kernel(), stride=stride, padding=padding)
        ret = ret.reshape(b, c, self.final_size, self.final_size)
        return ret


class AttentionDownsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, final_size, blur_attn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.in_dim = dim

        self.attn_kernel = nn.Parameter(
            torch.randn(kernel_size * kernel_size, dim, kernel_size, kernel_size, device='cuda') * 0.01 + 1
        )
        self.kernel_bias = nn.Parameter(
            torch.randn(kernel_size * kernel_size, device='cuda') * 0.01
        )
        self.logit_bias = nn.Parameter(
            torch.randn(1, kernel_size * kernel_size, 1, 1, device='cuda') * 0.1
        )
        self.blur_attn = blur_attn

        if blur_attn:
            blur_factor = torch.arange(-2, 3, 1, dtype=torch.float32).unsqueeze(0)
            blur_factor.pow_(2).mul_(-1 / math.sqrt(0.5)).exp_()
            blur_kernel = blur_factor.T * blur_factor
            blur_kernel /= blur_kernel.sum()
            self.register_buffer('blur_kernel', blur_kernel.reshape(1, 1, 5, 5))

    def forward_attention(self, feats, guidance):
        return self.attention_net(feats.permute(0, 2, 3, 1)).squeeze(-1).unsqueeze(1)

    def forward(self, hr_feats, guidance):
        b, c, h, w = hr_feats.shape

        inputs = hr_feats
        if self.blur_attn:
            inputs = F.conv2d(inputs, self.blur_kernel.expand(inputs.shape[1], -1, -1, -1),
                              groups=self.in_dim,
                              stride=1, padding=self.blur_kernel.shape[-2] // 2)

        stride = h // self.final_size

        # B,K*K,H,W
        padding = self.kernel_size // 2
        patches = F.conv2d(inputs, weight=self.attn_kernel, bias=self.kernel_bias, stride=stride, padding=padding)

        keep_mask = torch.rand_like(patches) > 0.2
        logits = (patches * keep_mask) + self.logit_bias
        attn = F.softmax(logits, dim=1)

        uf_patches = F.unfold(inputs, self.kernel_size, stride=stride, padding=padding)
        uf_patches = uf_patches.reshape(
            b, self.in_dim, self.kernel_size * self.kernel_size, self.final_size, self.final_size
        )

        downsampled = torch.einsum("bckhw,bkhw->bchw", uf_patches, attn)

        return downsampled
