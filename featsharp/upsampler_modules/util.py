# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import math
from typing import Callable, List, Union

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from .hadamard import get_prime_factors

def create_tiles(images: torch.Tensor, tile_generator: Callable[[torch.Tensor], torch.Tensor],
                 model_resolution: int, upsample_factors: Union[int, List[int]],
                 return_summary: bool = False,
):
    if isinstance(upsample_factors, int):
        upsample_factors = get_prime_factors(upsample_factors)
    upsample_factors = [1] + upsample_factors

    state = []
    inf_inputs = []

    prev_upf = 1
    for upf in upsample_factors:
        num_tiles_at_level = prev_upf * upf
        curr_res = model_resolution * num_tiles_at_level

        curr_images = F.interpolate(images, (curr_res, curr_res), mode='bilinear', align_corners=False)

        input_tiled = rearrange(curr_images, 'b c (h th) (w tw) -> (b h w) c th tw',
                                h=num_tiles_at_level, w=num_tiles_at_level,
                                th=model_resolution, tw=model_resolution)

        state.append((num_tiles_at_level, curr_res, curr_images, input_tiled))
        inf_inputs.append(input_tiled)
        prev_upf = num_tiles_at_level

    all_inf_input = torch.cat(inf_inputs)

    all_inf_output = tile_generator(all_inf_input, return_summary=return_summary)
    if return_summary:
        all_summary, all_inf_output = all_inf_output

    outputs = []

    offset = 0
    for i, (num_tiles_at_level, curr_res, curr_images, input_tiled) in enumerate(state):
        end_offset = offset + input_tiled.shape[0]
        flat_curr_slice = all_inf_output[offset:end_offset]
        if i == 0 and return_summary:
            summary = all_summary[:end_offset]

        output_tiled = rearrange(flat_curr_slice, '(b h w) c ph pw -> b c (h ph) (w pw)',
                                 b=images.shape[0], h=num_tiles_at_level, w=num_tiles_at_level)

        outputs.append(output_tiled)

        offset = end_offset

    if return_summary:
        return summary, outputs, cumprod(upsample_factors)
    return outputs, cumprod(upsample_factors)


def untile(packed_tiles: torch.Tensor, upsample_factors: Union[int, List[int]]):
    if isinstance(upsample_factors, int):
        upsample_factors = get_prime_factors(upsample_factors)

    num_tiles = 0
    prev = 1
    num_at_levels = []
    for upf in upsample_factors:
        f = prev * upf ** 2
        num_tiles += f
        num_at_levels.append(f)
        prev = f

    bsz = packed_tiles.shape[0] // num_tiles

    outputs = []
    offset = 0
    for num_at_level in num_at_levels:
        end_offset = offset + bsz * num_at_level
        upf = int(math.sqrt(num_at_level))

        curr_slice = packed_tiles[offset:end_offset]

        output_tiled = rearrange(curr_slice, '(b h w) c ph pw -> b c (h ph) (w pw)',
                                 b=bsz, h=upf, w=upf)
        outputs.append(output_tiled)

        offset = end_offset

    return outputs


@torch.no_grad()
def create_bilerp_guides(images: torch.Tensor, tile_generator: Callable[[torch.Tensor], torch.Tensor],
                         model_resolution: int, upsample_factors: Union[int, List[int]],
                         return_summary: bool = False,
):
    if isinstance(upsample_factors, int):
        upsample_factors = get_prime_factors(upsample_factors)
    upsample_factors = [1] + upsample_factors

    outputs = []

    prev_upf = 1
    for upf in upsample_factors:
        num_tiles_at_level = prev_upf * upf
        curr_res = model_resolution * num_tiles_at_level

        curr_images = F.interpolate(images, (curr_res, curr_res), mode='bilinear', align_corners=False)

        op = tile_generator(curr_images, return_summary=return_summary)

        if return_summary:
            summary, op = op
        outputs.append(op)

    if return_summary:
        return summary, outputs, cumprod(upsample_factors)
    return outputs, cumprod(upsample_factors)


def cumprod(vals: List[int], inclusive: bool = True):
    if inclusive:
        ret = [vals[0]]
        for i in range(1, len(vals)):
            ret.append(ret[-1] * vals[i])
    else:
        ret = [1]
        for i in range(0, len(vals) - 1):
            ret.append(ret[-1] * vals[i])
    return ret


class ChannelNorm(nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        new_x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return new_x
