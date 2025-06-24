# Source: https://github.com/PaulCouairon/JAFAR/blob/main/src/upsampler/jafar.py#L94

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .jafar_modules import CrossAttentionBlock, ResBlock, SFTModulation, RoPE, create_coordinate


class JAFAR(nn.Module):

    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        v_dim=384,
        feature_dim=None,
        kernel_size=1,
        num_heads=4,
        **kwargs,
    ):
        super().__init__()

        def make_encoder(in_dim, kernel_size, num_layers=2):
            return nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    qk_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    bias=False,
                ),
                *[
                    ResBlock(
                        qk_dim,
                        qk_dim,
                        kernel_size=1,
                        num_groups=8,
                        pad_mode="reflect",
                        norm_fn=nn.GroupNorm,
                        activation_fn=nn.SiLU,
                        use_conv_shortcut=False,
                    )
                    for _ in range(num_layers)
                ],
            )

        # Image encoder uses kernel_size=3 for spatial context
        self.image_encoder = make_encoder(input_dim, kernel_size=kernel_size)
        self.key_encoder = make_encoder(qk_dim, kernel_size=1)
        self.query_encoder = make_encoder(qk_dim, kernel_size=1)

        # Create Query features encoder
        self.norm = nn.GroupNorm(num_groups=8, num_channels=qk_dim, affine=False)

        # Create Key features encoder
        self.key_features_encoder = make_encoder(v_dim, kernel_size=1)
        self.cross_decode = CrossAttentionBlock(qk_dim, qk_dim, v_dim, num_heads)

        # SFT modulation for keys
        self.sft_key = SFTModulation(qk_dim, qk_dim)

        self.rope = RoPE(qk_dim)
        self.rope._device_weight_init()

    def upsample(self, encoded_image, features, output_size):
        _, _, h, w = features.shape

        # Process Queries
        queries = self.query_encoder(encoded_image)
        queries = F.adaptive_avg_pool2d(queries, output_size=output_size)
        queries = self.norm(queries)

        # Process Keys and Values.
        keys = self.key_encoder(encoded_image)
        keys = F.adaptive_avg_pool2d(keys, output_size=(h, w))
        keys = self.sft_key(keys, self.key_features_encoder(F.normalize(features, dim=1)))

        # Values
        values = features

        # Attention layer
        out = self.cross_decode(queries, keys, values)

        return out

    def forward(self, image, features, output_size):
        # Extract high-level features of image.
        encoded_image = self.image_encoder(image)

        # Apply Positional Encoding
        coords = create_coordinate(encoded_image.shape[-2], encoded_image.shape[-1])
        _, _, h, _ = encoded_image.shape
        encoded_image = rearrange(encoded_image, "b c h w -> b (h w) c")
        encoded_image = self.rope(encoded_image, coords)
        encoded_image = rearrange(encoded_image, "b (h w) c -> b c h w", h=h)

        # Get upsampled feats
        features = self.upsample(encoded_image, features, output_size)
        features = rearrange(features, "b (h w) c -> b c h w", h=output_size[0])
        return features


class JAFAR_Upsampler(nn.Module):
    def __init__(self, model_resolution: int, feat_dim: int, upsample_factor: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsampler = JAFAR(
            input_dim=3,
            qk_dim=128,
            v_dim=feat_dim,
            kernel_size=3,
            num_heads=4,
        )
        self.model_resolution = model_resolution
        self.upsample_factor = upsample_factor

    def forward(self, source: torch.Tensor, guidance: torch.Tensor):
        _, _, h, w = source.shape

        output_size = (h * self.upsample_factor, w * self.upsample_factor)

        gmin_h = min(guidance.shape[-2], output_size[0] * 2)
        gmin_w = min(guidance.shape[-1], output_size[1] * 2)
        if guidance.shape[-2] > gmin_h or guidance.shape[-1] > gmin_w:
            guidance = F.interpolate(guidance, size=(gmin_h, gmin_w), mode='bilinear', align_corners=False)

        upsampled = self.upsampler(guidance, source, output_size)
        return upsampled
