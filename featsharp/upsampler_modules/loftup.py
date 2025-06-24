# https://github.com/andrehuang/loftup
"""
Implements LoftUp (https://arxiv.org/pdf/2504.14032)
"""

import torch
import torch.nn as nn
import sys

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange

import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import numpy as np
import torchvision.transforms as T
from .util import ChannelNorm



class MinMaxScaler(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = x.shape[1]
        flat_x = x.permute(1, 0, 2, 3).reshape(c, -1)
        flat_x_min = flat_x.min(dim=-1).values.reshape(1, c, 1, 1)
        flat_x_scale = flat_x.max(dim=-1).values.reshape(1, c, 1, 1) - flat_x_min
        return ((x - flat_x_min) / flat_x_scale.clamp_min(0.0001)) - .5


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ImplicitFeaturizer(torch.nn.Module):

    def __init__(self, color_feats=True, n_freqs=10, learn_bias=False, time_feats=False, lr_feats=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_feats = color_feats
        self.time_feats = time_feats
        self.n_freqs = n_freqs
        self.learn_bias = learn_bias

        self.dim_multiplier = 2

        if self.color_feats:
            self.dim_multiplier += 3

        if self.time_feats:
            self.dim_multiplier += 1

        if self.learn_bias:
            self.biases = torch.nn.Parameter(torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32))

        self.low_res_feat = lr_feats

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1) # torch.Size([1, 30, 1, 1, 1])
        feats = (feats * freqs) # torch.Size([1, 30, 5, 224, 224])

        if self.learn_bias:
            sin_feats = feats + self.biases[0].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
            cos_feats = feats + self.biases[1].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
        else:
            sin_feats = feats
            cos_feats = feats

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        if self.low_res_feat is not None:
            upsampled_feats = F.interpolate(self.low_res_feat, size=(h, w), mode='bilinear', align_corners=False)
            all_feats.append(upsampled_feats)

        return torch.cat(all_feats, dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)  # Norm for query
        self.norm_kv = nn.LayerNorm(dim)  # Norm for key/value
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)

    def forward(self, query, key, value):
        # Apply layer normalization
        query = self.norm_q(query)
        key = self.norm_kv(key)
        value = self.norm_kv(value)

        # Multi-head attention takes (sequence_length, batch_size, embedding_dim)
        query = query.permute(1, 0, 2)  # (seq_len, batch_size, dim)
        key = key.permute(1, 0, 2)      # (seq_len, batch_size, dim)
        value = value.permute(1, 0, 2)  # (seq_len, batch_size, dim)

        # Apply multi-head attention (cross-attention)
        attn_output, _ = self.attention(query, key, value)

        # Return to original format (batch_size, seq_len, dim)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output

class CATransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttentionLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # Cross-Attention
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, query, key_value):
        for cross_attn, ff in self.layers:
            query = cross_attn(query, key_value, key_value) + query  # Cross-Attention
            # query = cross_attn(query, key_value, key_value) ## Because we are transforming imgs to features, we don't need to add the query back
            query = ff(query) + query  # Feed-Forward residual connection

        return self.norm(query)


class LoftUp(nn.Module):
    """
    We use Fourier features of images as inputs, and do cross attention with the LR features, the output is the HR features.
    """
    def __init__(self, model_resolution: int, patch_size: int, dim: int, color_feats=True, n_freqs=20, num_heads=4, num_layers=2, num_conv_layers=1, lr_pe_type="sine", upsample_factor=4):
        super(LoftUp, self).__init__()

        self._model_resolution = model_resolution
        self._patch_size = patch_size
        self._upsample_factor = upsample_factor

        if color_feats:
            start_dim = 5 * n_freqs * 2 + 3
        else:
            start_dim = 2 * n_freqs * 2

        lr_size = model_resolution // patch_size
        num_patches = lr_size * lr_size
        self.lr_pe_type = lr_pe_type
        if self.lr_pe_type == "sine":
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=5, learn_bias=True)
            self.lr_pe_dim = 2 * 5 * 2
        elif self.lr_pe_type == "learnable":
            self.lr_pe = nn.Parameter(torch.randn(1, num_patches, dim))
            self.lr_pe_dim = dim

        self.fourier_feat = torch.nn.Sequential(
                                MinMaxScaler(),
                                ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True),
                            )
        if self.lr_pe_type == "sine": # LR PE is concatenated to LR
            self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim+self.lr_pe_dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim+self.lr_pe_dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(dim+self.lr_pe_dim, dim+self.lr_pe_dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim+self.lr_pe_dim),
                                nn.ReLU(inplace=True),
                                )


            self.final_conv = torch.nn.Sequential(
                nn.Conv2d(dim+self.lr_pe_dim, dim, kernel_size=1),
                LayerNorm(dim),
            )

            self.ca_transformer = CATransformer(dim+self.lr_pe_dim, depth=num_layers, heads=num_heads, dim_head=dim//num_heads, mlp_dim=dim, dropout=0.)
        elif self.lr_pe_type == "learnable": # LR PE is added to LR
            self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True),
                                )
            self.final_conv = LayerNorm(dim)
            self.ca_transformer = CATransformer(dim, depth=num_layers, heads=num_heads, dim_head=dim//num_heads, mlp_dim=dim, dropout=0.)

    @property
    def hi_res_input(self) -> bool:
        return True

    @property
    def upsample_factor(self):
        return self._upsample_factor

    @property
    def input_upsample_factor(self):
        return 1

    def forward(self, guidance: torch.Tensor, tile_generator: nn.Module, return_summary: bool = False) -> torch.Tensor:
        lr_guidance = F.interpolate(guidance, size=(self._model_resolution,) * 2, mode='bilinear', align_corners=False)
        lr_feats = tile_generator(lr_guidance)
        if return_summary:
            summary, lr_feats = lr_feats

        hr_guidance = F.interpolate(guidance, size=(self._model_resolution // self._patch_size * self._upsample_factor,) * 2, mode='bilinear', align_corners=False)

        # Step 1: Extract Fourier features from the input image
        x = self.fourier_feat(hr_guidance) # Output shape: (B, dim, H, W)
        b, c, h, w = x.shape

        ## Resize and add LR feats to x?
        x = self.first_conv(x)

        # Reshape for attention (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Step 2: Process LR features for keys and values
        b, c_lr, h_lr, w_lr = lr_feats.shape

        if self.lr_pe_type == "sine":
            lr_pe = self.lr_pe(lr_feats)
            lr_feats_with_pe = torch.cat([lr_feats, lr_pe], dim=1)
            lr_feats_with_pe = lr_feats_with_pe.flatten(2).permute(0, 2, 1)
        elif self.lr_pe_type == "learnable":
            lr_feats = lr_feats.flatten(2).permute(0, 2, 1) # (B, H*W, C)
            if lr_feats.shape[1] != self.lr_pe.shape[1]:
                len_pos_old = int(math.sqrt(self.lr_pe.shape[1]))
                pe = self.lr_pe.reshape(1, len_pos_old, len_pos_old, c_lr).permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=(h_lr, w_lr), mode='bicubic', align_corners=False)
                pe = pe.reshape(1, c_lr, h_lr*w_lr).permute(0, 2, 1)
                lr_feats_with_pe = lr_feats + pe
            else:
                lr_feats_with_pe = lr_feats + self.lr_pe
        x = self.ca_transformer(x, lr_feats_with_pe)

        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        ret = self.final_conv(x)

        if return_summary:
            return summary, lr_feats, ret
        return lr_feats, ret

class UpsamplerwithChannelNorm(nn.Module):
    def __init__(self, upsampler, channelnorm):
        super(UpsamplerwithChannelNorm, self).__init__()
        self.upsampler = upsampler
        self.channelnorm = channelnorm

    def forward(self, lr_feats, img):
        lr_feats = self.channelnorm(lr_feats)
        return self.upsampler(lr_feats, img)

def load_loftup_checkpoint(upsampler_path, n_dim, lr_pe_type="sine", lr_size=16):
    channelnorm = ChannelNorm(n_dim)
    upsampler = LoftUp(n_dim, lr_pe_type=lr_pe_type, lr_size=16)
    ckpt_weight = torch.load(upsampler_path)['state_dict']
    channelnorm_checkpoint = {k: v for k, v in ckpt_weight.items() if 'model.1' in k} # dict_keys(['model.1.norm.weight', 'model.1.norm.bias'])
    # change the key names
    channelnorm_checkpoint = {k.replace('model.1.', ''): v for k, v in channelnorm_checkpoint.items()}
    # if the key starts with upsampler, remove the upsampler.
    upsampler_ckpt_weight = {k: v for k, v in ckpt_weight.items() if k.startswith('upsampler')}
    upsampler_ckpt_weight = {k.replace('upsampler.', ''): v for k, v in upsampler_ckpt_weight.items()}
    upsampler.load_state_dict(upsampler_ckpt_weight)
    channelnorm.load_state_dict(channelnorm_checkpoint)
    for param in upsampler.parameters():
        param.requires_grad = False
    for param in channelnorm.parameters():
        param.requires_grad = False
    # return channelnorm, upsampler
    return UpsamplerwithChannelNorm(upsampler, channelnorm)


def load_upsampler_weights(upsampler, upsampler_path, dim, freeze=True):
    channelnorm = ChannelNorm(dim)
    ckpt_weight = torch.load(upsampler_path)['state_dict']
    channelnorm_checkpoint = {k: v for k, v in ckpt_weight.items() if 'model.1' in k} # dict_keys(['model.1.norm.weight', 'model.1.norm.bias'])
    channelnorm_checkpoint = {k.replace('model.1.', ''): v for k, v in channelnorm_checkpoint.items()}
    upsampler_ckpt_weight = {k: v for k, v in ckpt_weight.items() if k.startswith('upsampler')}
    upsampler_ckpt_weight = {k.replace('upsampler.', ''): v for k, v in upsampler_ckpt_weight.items()}

    upsampler.load_state_dict(upsampler_ckpt_weight)
    channelnorm.load_state_dict(channelnorm_checkpoint)
    if freeze:
        for param in upsampler.parameters():
            param.requires_grad = False
        for param in channelnorm.parameters():
            param.requires_grad = False
    return UpsamplerwithChannelNorm(upsampler, channelnorm)
