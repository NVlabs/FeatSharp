# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from collections import defaultdict
from dataclasses import dataclass, field
import debugpy
from functools import partial
from typing import Callable, List, Set, Tuple, Dict, Optional, Union

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist


from featsharp.enable_spectral_reparam import enable_spectral_reparam
from featsharp.data.transforms import generate_homography_grid
from featsharp.downsamplers import SimpleDownsampler, AttentionDownsampler
from featsharp.enable_spectral_reparam import enable_spectral_reparam
from featsharp.featurizers.util import get_featurizer
from featsharp.layers import get_normalizer, BiasBuffer
from featsharp.losses import TVLoss, SampledCRFLoss, apply_weight, Distance_T, DEFAULT_DISTANCE, MMDLoss
from featsharp.upsamplers import get_upsampler
from featsharp.upsampler_modules.util import create_tiles, untile
from featsharp.util import (
    extract_normalize,
    pca,
    UnNormalize,
    prep_image,
    get_rank,
    get_world_size,
    DistributedDataParallel,
    barrier,
)
from featsharp.visualization import VizWriter

from timm.models.vision_transformer import Block as TxBlock


class UncertaintyNet(nn.Module):
    def __init__(self, dim: int, keys: Set[str], kernel_size: int = 5, separate_channels: bool = False):
        super().__init__()
        self.dim = dim
        self.keys = keys
        self.separate_channels = separate_channels
        num_channels = len(keys) if separate_channels else 1

        c2 = nn.Conv2d(256, num_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.net = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, padding_mode='reflect'),
            nn.GELU(),
            c2,
        )
        with torch.no_grad():
            c2.weight.fill_(0.0)
            if c2.bias is not None:
                c2.bias.fill_(0.0)


    def forward(self, x) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        unc = self.net(x)

        scales = torch.exp(-unc)
        regs = unc / 2

        return {
            k: scales[:, i if self.separate_channels else 0]
            for i, k in enumerate(self.keys)
        }, regs


@dataclass
class ValState:
    lr_feats: Union[List[torch.Tensor], torch.Tensor] = field(default_factory=list)
    hr_feats: Union[List[torch.Tensor], torch.Tensor] = field(default_factory=list)
    real_hr_feats: Union[List[torch.Tensor], torch.Tensor] = field(default_factory=list)

    real_tile_diff: Dict[int, List[torch.Tensor]] = field(default_factory=partial(defaultdict, list))

    def finish(self):
        self._cat_gather('lr_feats')
        self._cat_gather('hr_feats')
        self._cat_gather('real_hr_feats')

        for k, v in self.real_tile_diff.items():
            if len(v):
                v = torch.stack(v).mean()
                if get_world_size() > 1:
                    dist.all_reduce(v, op=dist.ReduceOp.AVG)
                self.real_tile_diff[k] = v

    def _cat_gather(self, name):
        val = getattr(self, name)
        if val:
            val = torch.cat(val)
            setattr(self, name, val)


class TrainingHarness(nn.Module):
    def __init__(self, model_type: str, model_extra_kwargs: dict,
                 activation_type: str,
                 upsampler: str,
                 upsample_factor: int,
                 downsampler: str,
                 ds_kernel_size: int,
                 input_size: int = 0,
                 normalizer_mode: str = 'phi-s',
                 distance_fn: Distance_T = DEFAULT_DISTANCE,
                 predicted_uncertainty: bool = True,
                 tv_weight: float = 0.,
                 crf_weight: float = 0.,
                 mmd_weight: float = 0.,
                 random_projection: int = 0,
                 n_jitters: int = 5,
                 bias_buffer: bool = True,
                 grid_weight: float = 1.,
                 spectral_reparam: bool = False,
                 cond_bias_weight: float = 0.0,
    ):
        super().__init__()
        self.tv_weight = tv_weight
        self.crf_weight = crf_weight
        self.predicted_uncertainty = predicted_uncertainty
        self.distance_fn = distance_fn
        self.random_projection = random_projection
        self.n_jitters = n_jitters

        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000, **model_extra_kwargs)
        for p in self.model.parameters():
            p.requires_grad = False

        self.input_size = getattr(self.model, 'input_size', input_size)
        self.final_size = self.input_size // self.patch_size

        self.model.eval()
        self.model.cuda()
        self.normalizer = get_normalizer(normalizer_mode, self.dim)

        upsampler = get_upsampler(self.input_size, self.patch_size, upsampler, self.dim, upsample_factor)
        if spectral_reparam:
            enable_spectral_reparam(upsampler, init_norm_to_current=False)
        self.upsampler = wrap_model(upsampler)
        self.upsample_factor = upsample_factor

        if downsampler == 'simple':
            downsampler = SimpleDownsampler(ds_kernel_size, self.final_size)
        elif downsampler == 'attention':
            downsampler = AttentionDownsampler(self.dim, ds_kernel_size, self.final_size, blur_attn=True)
        else:
            raise ValueError(f'Unknown downsampler: {downsampler}')
        if spectral_reparam:
            enable_spectral_reparam(downsampler, init_norm_to_current=False)
        self.downsampler = wrap_model(downsampler)

        input_cond = getattr(self.model, 'input_conditioner', None)
        if input_cond is None:
            from featsharp.util import norm as input_cond
        self.input_conditioner = extract_normalize(input_cond).cuda()
        self.inv_input_conditioner = UnNormalize(self.input_conditioner.mean, self.input_conditioner.std).cuda()

        self.crf = SampledCRFLoss(
            alpha=.1,
            beta=.15,
            gamma=.005,
            w1=10.0,
            w2=0.0,
            shift=0.00,
            n_samples=1000,
            distance_fn=self.distance_fn,
        )
        self.tv = TVLoss(self.distance_fn)

        self.mmd_weight = mmd_weight
        self.mmd = MMDLoss(upsample_factor)

        if self.predicted_uncertainty:
            scale_net = UncertaintyNet(self.dim,
                keys=({'rec'}.union(
                    {'crf'} if True or crf_weight > 0 else set()).union(
                    {'tv'} if True or tv_weight > 0 else set()))
            )
            if spectral_reparam:
                enable_spectral_reparam(scale_net, init_norm_to_current=False)
            self.scale_net = wrap_model(scale_net)

        self._rand_proj = None

        loss_grid = torch.ones(1, 1, self.final_size * self.upsample_factor, self.final_size * self.upsample_factor)
        for offset in range(self.final_size, loss_grid.shape[-1], self.final_size):
            loss_grid[..., offset - 1:offset + 1, :].fill_(grid_weight)
            loss_grid[..., offset - 1:offset + 1].fill_(grid_weight)
        loss_grid.mul_(loss_grid.numel() / loss_grid.sum())

        self.register_buffer('loss_grid', loss_grid, persistent=False)

        self.use_bias_buffer = bias_buffer
        self.bias_buffer = wrap_model(BiasBuffer(self.dim, self.final_size) if bias_buffer else nn.Identity())

        self.use_cond_bias_transform = cond_bias_weight > 0
        self.cond_bias_weight = cond_bias_weight
        if self.use_cond_bias_transform:
            cbt = TxBlock(self.dim, num_heads=16, init_values=1e-8)
            enable_spectral_reparam(cbt, init_norm_to_current=False)
            self.cond_bias_transform = wrap_model(cbt)
            self.cb_deltas = []

        self.spectral_norms = dict()

    def forward(self, x: torch.Tensor):
        lr_feats, hr_feats = self.upsampler(x, self.get_lr_features)
        return lr_feats, hr_feats

    def trainable_params(self, lr: float, **kwargs):
        all_params = []
        regular_params = []
        wd_params = []

        jbu_regular_params = []
        jbu_wd_params = []

        def add_params(model: nn.Module):
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue

                regs = jbu_regular_params if '.jbu.' in n else regular_params
                wds = jbu_wd_params if '.jbu.' in n else wd_params

                if 'sn_scale' in n:
                    wds.append(p)
                    self.spectral_norms[n] = p
                else:
                    regs.append(p)

        add_params(self.downsampler)
        add_params(self.upsampler)
        add_params(self.bias_buffer)
        if self.use_cond_bias_transform:
            add_params(self.cond_bias_transform)

        all_params.append({'params': regular_params, 'lr': lr, **kwargs})
        if jbu_regular_params:
            all_params.append({'params': jbu_regular_params, 'lr': 1e-4, **kwargs})
        if wd_params:
            all_params.append({'params': wd_params, 'lr': lr * 0.1, 'weight_decay': 0.01, **kwargs})
        if jbu_wd_params:
            all_params.append({'params': jbu_wd_params, 'lr': 1e-4, 'weight_decay': 0.01, **kwargs})

        if self.predicted_uncertainty:
            all_params.append({'params': list(self.scale_net.parameters()), 'lr': 1e-5, **kwargs})
        return all_params

    # NOTE: The inner featurizer may have a difference autocasting setup, but we always want to
    # start with fp32 since we don't know what the inner component needs
    @torch.autocast('cuda', enabled=False)
    def get_lr_features(self, x: torch.Tensor, bias: bool = True, return_summary: bool = False):
        assert not return_summary, "`return_summary` not supported during training!"
        with torch.no_grad():
            x = self.model(x)
            x = self.normalizer(x)
        if bias:
            x = self.bias_buffer(x)
        if self.use_cond_bias_transform:
            prev_shape = x.shape
            xf = x.flatten(2).permute(0, 2, 1)
            xd = self.cond_bias_transform(xf)
            xd = xd.permute(0, 2, 1).reshape(prev_shape)
            delta = xd - x
            self.cb_deltas.append((x, delta))
            x = xd
        return x

    def project(self, feats: torch.Tensor, proj: Optional[torch.Tensor]):
        if proj is not None:
            feats = torch.einsum("bchw,bcd->bdhw", feats, proj)
        return feats

    def _prep_images(self, img: torch.Tensor):
        return self.input_conditioner(img)

    def extract_images(self, batch, idx: int) -> List[torch.Tensor]:
        which = batch[idx]
        if idx == 0:
            img, valid_mask = which
            tx = None
        else:
            img, valid_mask, tx = which

        if img.ndim == 5:
            img = img[0]
            valid_mask = valid_mask[0]
            tx = tx[0] if tx is not None else None
        return self._prep_images(img), valid_mask.unsqueeze(1), tx

    def get_rand_projection(self, lr_feats: torch.Tensor):
        if not self.random_projection:
            return None

        def _get_rand():
            rp = torch.randn(lr_feats.shape[0], lr_feats.shape[1], self.random_projection,
                            device=lr_feats.device, dtype=lr_feats.dtype)

            q, _ = torch.linalg.qr(rp, mode='reduced')
            return q

        return _get_rand()

    @property
    def input_upsample_factor(self):
        return getattr(self.upsampler, 'input_upsample_factor', 1)

    def denormalize(self, features: torch.Tensor):
        denorm_fn = getattr(self.normalizer, 'denormalize', None)
        if denorm_fn is None:
            return features
        features = denorm_fn(features)
        return features

    def load_state_dict(self, state_dict, strict = True, assign = False):
        groups = defaultdict(dict)
        for key, val in state_dict.items():
            parts = key.split('.')
            groups[parts[0]]['.'.join(parts[1:])] = val

        for key, sd in groups.items():
            mod: nn.Module = getattr(self, key)
            mod.load_state_dict(sd)
        pass

    def forward_train(self, batch: List[List[torch.Tensor]], backward_fn: Callable[[torch.Tensor], None],
                    autocast: bool = False, global_step: int = 0):
        self.upsampler.train()

        ac_dtype = torch.float16 if autocast else torch.float32
        do_autocast = lambda: torch.autocast('cuda', dtype=ac_dtype, enabled=autocast)

        hr_img, hr_valid_mask, _ = self.extract_images(batch, 0)

        lr_res = tuple(
            s // self.input_upsample_factor
            for s in hr_img.shape[2:]
        )

        img = F.interpolate(hr_img, lr_res, mode='bilinear', align_corners=False)
        valid_mask = F.interpolate(hr_valid_mask, lr_res, mode='nearest')

        target_res = tuple(
            s // self.patch_size * self.upsample_factor
            for s in lr_res
        )

        with do_autocast():
            lr_feats, hr_feats = self.upsampler(hr_img, self.get_lr_features)

        if self.use_cond_bias_transform:
            ups_deltas = list(self.cb_deltas)
            self.cb_deltas.clear()
            ups_deltas = tilize_deltas(self.upsampler, ups_deltas)
            cb_main_loss_weights = [torch.ones_like(ud[:, :1]) for ud in ups_deltas[0]]

        proj = self.get_rand_projection(lr_feats)

        if hr_feats.shape[2:] != target_res:
            hr_feats = F.interpolate(hr_feats, target_res, mode="bilinear", align_corners=False)

        if self.predicted_uncertainty:
            unc_scales, unc_reg = self.scale_net(hr_feats.detach())

            res_valid_mask = F.interpolate(hr_valid_mask, target_res, mode='nearest')
            unc_reg_loss = (unc_reg * res_valid_mask).mean()
        else:
            unc_reg_loss = 0.0

        loss_grid: torch.Tensor = self.loss_grid.expand(hr_img.shape[0], -1, -1, -1)

        num_retain = self.n_jitters - (0 if self.use_cond_bias_transform else 1)

        full_rec_loss = 0.0
        full_raw_rec_loss = 0.0
        full_raw_denorm_rec_loss = 0.0
        full_crf_loss = 0.0
        full_mmd_loss = 0.0
        full_min_dist_loss = 0.0
        full_tv_loss = 0.0
        full_total_loss = 0.0
        full_unc_scale = 0.0
        full_cb_loss = 0.0
        cb_loss_max = 0
        for i in range(self.n_jitters):
            hr_jit_img, hr_jit_valid_mask, tx_base_to_i = self.extract_images(batch, i + 1)

            lr_jit_img = F.interpolate(hr_jit_img, lr_res, mode='bilinear', align_corners=False)
            lr_jit_feats = self.get_lr_features(lr_jit_img)

            if self.use_cond_bias_transform:
                curr_delta = self.cb_deltas[0]
                self.cb_deltas.clear()

            hr_feat_grid = generate_homography_grid(tx_base_to_i, hr_feats.shape).to(hr_feats.dtype)
            hr_warp_feats = F.grid_sample(hr_feats, hr_feat_grid, mode='bilinear', align_corners=True, padding_mode='reflection')
            hr_loss_grid = F.grid_sample(loss_grid, hr_feat_grid, mode='bilinear', align_corners=True)

            with do_autocast():
                lr_warp_feats: torch.Tensor = self.downsampler(hr_warp_feats, lr_jit_img)

            lr_feat_grid = generate_homography_grid(tx_base_to_i, lr_feats.shape).to(valid_mask.dtype)
            lr_loss_grid = F.grid_sample(loss_grid, lr_feat_grid, mode='bilinear', align_corners=True)
            tx_lr_valid_mask = F.grid_sample(valid_mask, lr_feat_grid, mode='nearest', align_corners=True)
            jit_valid_mask = F.interpolate(hr_jit_valid_mask, lr_feats.shape[-2:], mode='nearest')
            jit_valid_mask = tx_lr_valid_mask * jit_valid_mask

            valid_sum = jit_valid_mask.sum().clamp_min_(jit_valid_mask.numel() / 2)

            curr_cb_loss = 0
            if self.use_cond_bias_transform:
                curr_loss_weight = torch.ones_like(tx_lr_valid_mask)

                tx_warp_to_main = tx_base_to_i.inverse()

                for tlidx, tl in enumerate(ups_deltas[0]):
                    tl_to_curr = F.grid_sample(tl.detach(), lr_feat_grid, mode='bilinear', align_corners=True)

                    sq_diff = (tl_to_curr - curr_delta[0].detach()).pow_(2).sum(dim=1, keepdim=True)
                    diff_var = ((sq_diff * jit_valid_mask).sum(dim=(1, 2, 3), keepdim=True) / jit_valid_mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)).clamp_min_(1)

                    rbf_dist = torch.exp(-sq_diff / diff_var) * jit_valid_mask

                    curr_loss_weight += rbf_dist

                    grid_to_main = generate_homography_grid(tx_warp_to_main, tl.shape).to(valid_mask.dtype)
                    curr_to_tl = F.grid_sample(curr_delta[0], grid_to_main, mode='bilinear', align_corners=True)
                    valid_to_main = F.grid_sample(jit_valid_mask, grid_to_main, mode='nearest', align_corners=True)

                    sq_diff = (curr_to_tl - tl).detach().pow_(2).sum(dim=1, keepdim=True)
                    diff_var = ((sq_diff * valid_to_main).sum(dim=(1, 2, 3), keepdim=True) / valid_to_main.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)).clamp_min_(1)
                    rbf_dist = torch.exp(-sq_diff / diff_var) * valid_to_main

                    cb_main_loss_weights[tlidx] += rbf_dist

                    pass
                curr_loss_weight /= curr_loss_weight.mean(dim=(1, 2, 3), keepdim=True)

                curr_cb_loss = (curr_loss_weight * (curr_delta[1].pow(2))).mean()
                cb_loss_max = max(cb_loss_max, curr_delta[1].abs().amax().item())
                full_cb_loss = full_cb_loss + curr_cb_loss.item()

            res_jit_valid_mask = F.interpolate(hr_jit_valid_mask, target_res, mode='nearest')
            res_warp_valid_mask = F.grid_sample(hr_valid_mask, hr_feat_grid, mode='nearest', align_corners=True)
            res_valid_mask = res_jit_valid_mask * res_warp_valid_mask

            mse = (jit_valid_mask * lr_loss_grid * self.distance_fn(lr_warp_feats, lr_jit_feats, reduction='none')).mean(dim=1)
            denorm_mse = (jit_valid_mask * F.mse_loss(
                self.denormalize(lr_warp_feats.detach()),
                self.denormalize(lr_jit_feats.detach()),
                reduction='none',
            )).mean(dim=1)
            raw_denorm_rec_loss = denorm_mse.sum() / (self.n_jitters * valid_sum)

            if self.predicted_uncertainty:
                unc_lr_warp_scale = F.grid_sample(unc_scales['rec'].unsqueeze(1), lr_feat_grid, mode='bilinear', align_corners=True, padding_mode='reflection').squeeze(1)

                raw_rec_loss = mse.detach().sum() / (self.n_jitters * valid_sum)

                unc_mse = jit_valid_mask.squeeze(1) * (unc_lr_warp_scale * mse)
                rec_loss = unc_mse.sum() / (self.n_jitters * valid_sum)
                full_unc_scale += unc_lr_warp_scale.detach().sum() / (self.n_jitters * valid_sum)
            else:
                rec_loss = mse.sum() / (self.n_jitters * valid_sum)
                raw_rec_loss = rec_loss.detach()

            full_rec_loss += rec_loss.item()
            full_raw_rec_loss += raw_rec_loss.item()
            full_raw_denorm_rec_loss += raw_denorm_rec_loss.item()

            if self.crf_weight >= 0 and i == 0:
                guidance = img
                if guidance.shape[2:] != hr_feats.shape[2:]:
                    guidance = F.interpolate(guidance, hr_feats.shape[2:], mode='bilinear', align_corners=False)

                if self.predicted_uncertainty:
                    res_warp_unc_scale = unc_scales['crf'].unsqueeze(1)
                else:
                    res_warp_unc_scale = torch.ones_like(res_valid_mask)

                crf_loss, crf_raw_loss = self.crf(guidance, self.project(hr_feats, proj), res_valid_mask, res_warp_unc_scale * hr_loss_grid)
                full_crf_loss += crf_raw_loss.item()
            else:
                crf_loss = 0.0

            if self.tv_weight >= 0 and i == 0:
                res_valid_mask = F.interpolate(hr_valid_mask, target_res, mode='nearest')

                if self.predicted_uncertainty:
                    res_unc_scale = unc_scales['tv'].unsqueeze(1)
                else:
                    res_unc_scale = torch.ones_like(res_valid_mask)

                tv_loss, tv_raw_loss = self.tv(hr_feats, res_valid_mask, res_unc_scale * loss_grid)
                full_tv_loss += tv_raw_loss.item()
            else:
                tv_loss = 0.0

            if self.mmd_weight >= 0 and i == 0:
                mmd_loss = self.mmd(lr_feats, hr_feats)
                full_mmd_loss += mmd_loss.item()
            else:
                mmd_loss = 0.0

            loss_no_unc_reg = (
                rec_loss
                + apply_weight(self.crf_weight, crf_loss)
                + apply_weight(self.tv_weight, tv_loss)
                + apply_weight(self.mmd_weight, mmd_loss)
                + (self.cond_bias_weight / (self.n_jitters + 1)) * curr_cb_loss
            )
            loss = loss_no_unc_reg + (unc_reg_loss / self.n_jitters)
            full_total_loss += loss_no_unc_reg.item()

            backward_fn(loss, retain_graph=i < num_retain)
            pass

        if self.use_cond_bias_transform:
            cb_main_loss = 0
            for delta, loss_weight in zip(ups_deltas[1], cb_main_loss_weights):
                loss_weight /= loss_weight.mean(dim=(1, 2, 3), keepdim=True)
                curr_loss = (loss_weight * (delta.pow(2))).mean()
                cb_main_loss = cb_main_loss + (curr_loss / len(cb_main_loss_weights))
                cb_loss_max = max(cb_loss_max, delta.abs().amax().item())
            full_cb_loss = full_cb_loss + cb_main_loss.item()
            backward_fn(self.cond_bias_weight * cb_main_loss, retain_graph=False)


        components = {
            'loss/crf': full_crf_loss,
            'loss/tv': full_tv_loss,
            'loss/rec': full_rec_loss,
            'loss/rec_raw': full_raw_rec_loss,
            'loss/rec_raw_denorm': full_raw_denorm_rec_loss,
            'loss/mmd': full_mmd_loss,
            'loss/min_dist': full_min_dist_loss,
            'loss/total': full_total_loss,
            'loss/cb_delta': full_cb_loss,
            'loss/cb_delta_max': cb_loss_max,
        }
        if get_rank() == 0 and self.use_cond_bias_transform:
            print(f'CB Loss: {full_cb_loss:.8f}, Max Delta: {cb_loss_max:.8f}')
        if self.predicted_uncertainty:
            components.update({
                'loss/rec_unc_scale': full_unc_scale.item(),
                'loss/rec_unc_reg': unc_reg_loss.mean().item(),
            })
        norm_stats_fn = getattr(self.normalizer, 'get_log_stats', None)
        if norm_stats_fn is not None:
            components.update(norm_stats_fn())

        if 'phi_s_alpha' in components:
            variance = components['phi_s_alpha'] ** -2
            fidelity = variance / components['loss/rec_raw_denorm']
            components['dist_variance'] = variance
            components['fidelity'] = fidelity

        norms = None
        with torch.no_grad():
            if self.spectral_norms:
                norms = F.softplus(torch.cat(list(self.spectral_norms.values())).detach()) + 0.05
            elif (global_step + 1) % 500 == 0:
                rank = get_rank()
                world_size = get_world_size()
                off = 0
                norms = []
                for n, mod in self.named_modules():
                    if not hasattr(mod, 'weight'):
                        continue

                    w: torch.Tensor = mod.weight
                    if not w.requires_grad or w.numel() < 2 or w.ndim < 2:
                        continue

                    w = w.flatten(1)
                    if any(s < 2 for s in w.shape):
                        continue

                    if off % world_size == rank:
                        S = torch.linalg.svdvals(w)
                        spect = S[0]
                        norms.append(spect)
                    off += 1

                norms = (torch.stack(norms) if norms else torch.tensor([], dtype=torch.float32)).cpu()
                if world_size > 1:
                    all_norms = [None for _ in range(world_size)]
                    dist.gather_object(norms, all_norms if rank == 0 else None, dst=0)
                    if rank == 0:
                        norms = torch.cat(all_norms)

        if norms is not None and norms.numel() > 0:
            components.update({
                'spectral/mean': norms.mean(),
                'spectral/median': norms.median(),
                'spectral/max': norms.amax(),
            })


        return components

    @torch.no_grad()
    def forward_validation(self, batch: List[List[torch.Tensor]], batch_idx: int, viz: VizWriter, global_step: int,
                        state: Optional[ValState] = None):
        global first_val, first_val_step
        if first_val:
            self.upsampler.eval()
            first_val = False
            first_val_step = global_step

        if state is None:
            state = ValState()

        barrier()

        proj_gen = torch.Generator('cuda')
        proj_gen.manual_seed(batch_idx)

        hr_img, hr_valid_mask, _ = self.extract_images(batch, 0)

        lr_res = tuple(
            s // self.input_upsample_factor
            for s in hr_img.shape[2:]
        )

        img = F.interpolate(hr_img, lr_res, mode='bilinear', align_corners=False)

        target_res = tuple(
            s // self.patch_size * self.upsample_factor
            for s in lr_res
        )

        lr_feats, hr_feats = self.upsampler(hr_img, self.get_lr_features)
        if self.use_cond_bias_transform:
            cond_deltas = list(self.cb_deltas)

        if hr_feats.shape[2:] != target_res:
            hr_feats = F.interpolate(hr_feats, target_res, mode="bilinear", align_corners=False)

        bl_hr_feats = F.interpolate(lr_feats, target_res, mode='bilinear', align_corners=False)

        def get_diff_buffer(pred: torch.Tensor, gt: torch.Tensor):
            diff = (pred - gt).pow(2).sum(dim=1).sqrt_()
            diff_std, diff_mean = torch.std_mean(diff, dim=(1, 2), keepdim=True)
            diff_max = diff.amax(dim=(1, 2), keepdim=True)
            diff_max = torch.minimum(diff_mean + 10 * diff_std, diff_max)
            diff_max = torch.where(diff_max > 5, diff_max.clamp_min(85), diff_max)

            diff_scale = torch.where(torch.logical_and(diff_max < 255, diff_max > 20), 255 / diff_max, torch.where(diff_max < 5, 5 / diff_max, 1))
            diff_img = diff.mul(diff_scale).clamp_max_(255).to(torch.uint8)
            return diff, diff_img

        hr_bl_diff, hr_bl_diff_img = get_diff_buffer(hr_feats, bl_hr_feats)

        hr_jit_img, hr_jit_valid_mask, jit_tx = self.extract_images(batch, 1)
        jit_img = F.interpolate(hr_jit_img, lr_res, mode='bilinear', align_corners=False)

        lr_jit_feats, hr_jit_feats = self(hr_jit_img)

        if self.predicted_uncertainty:
            scales, _ = self.scale_net(hr_feats)

        tx_0_to_1 = jit_tx
        grid = generate_homography_grid(tx_0_to_1, hr_feats.shape)
        hr_jit_feats = F.grid_sample(hr_feats, grid, mode='bilinear', align_corners=True, padding_mode='reflection')
        down_jit_feats = self.downsampler(hr_jit_feats, jit_img)

        img_grid = generate_homography_grid(tx_0_to_1, jit_img.shape)
        warp_img_to_jit = F.grid_sample(img, img_grid, mode='bilinear', align_corners=True, padding_mode='reflection')

        overlay_jit = (warp_img_to_jit + jit_img) / 2

        tile_levels, upsample_factors = create_tiles(
            hr_img,
            partial(self.get_lr_features, bias=False),
            self.input_size,
            self.upsample_factor,
        )

        hr_real = F.interpolate(hr_img, size=tuple(sz * upsample_factors[-1] for sz in (self.input_size, self.input_size)), mode='bilinear', align_corners=False)

        if getattr(self.model, 'allow_variable_resolution', True):
            hr_real_feats = self.get_lr_features(hr_real, bias=False)

            state.real_hr_feats.append(hr_real_feats)

        if self.use_cond_bias_transform:
            cond_deltas = tilize_deltas(self.upsampler, cond_deltas)

        for z in range(lr_feats.shape[0]):
            val_idx = batch_idx * lr_feats.shape[0] + z
            fit_pca = pcas.get(val_idx, None)

            if global_step == first_val_step:
                tiled_img_base = F.interpolate(self.inv_input_conditioner(img[z].unsqueeze(0)), (self.input_size, self.input_size), mode='bilinear', align_corners=False)[0]

                tl_range = range(len(tile_levels) - 1, -1, -1) if getattr(self.model, 'allow_variable_resolution', True) else range(len(tile_levels))
                for k in tl_range:
                    num_tiles = upsample_factors[k]
                    proc_size = tuple(sz * num_tiles for sz in (self.input_size, self.input_size))
                    hr_real = F.interpolate(hr_img[z].unsqueeze(0), proc_size, mode='bilinear', align_corners=False)

                    hr_real_feats = None
                    if getattr(self.model, 'allow_variable_resolution', True) or num_tiles == 1:
                        hr_real_feats = self.get_lr_features(hr_real, bias=False)
                        [red_hr_real_feats], fit_pca = pca([hr_real_feats], fit_pca=fit_pca)
                        viz.add_image(f'real_viz/image_{val_idx}/zoom_{num_tiles}x', red_hr_real_feats[0])

                        if val_idx not in pcas:
                            pcas[val_idx] = fit_pca

                    tile_level = tile_levels[k]
                    [red_tl], _ = pca([tile_level[z].unsqueeze(0)], fit_pca=fit_pca)
                    red_tl = red_tl[0]

                    tiled_img = tiled_img_base.clone()

                    px_line_offset = tiled_img_base.shape[-1] / num_tiles
                    for x in range(1, num_tiles):
                        rmin = int(round(x * px_line_offset - 1))
                        rmax = rmin + 3
                        tiled_img[..., rmin:rmax, :].fill_(1)
                        tiled_img[..., rmin:rmax].fill_(1)

                    viz.add_image(f"feat_tile/{val_idx}/zoom_{num_tiles}x", red_tl)
                    viz.add_image(f"image_tile/{val_idx}/raw_{num_tiles}x", tiled_img)

                    if hr_real_feats is not None:
                        real_tile_diff, real_tile_diff_img = get_diff_buffer(hr_real_feats, tile_level[z:z+1])
                        viz.add_image(f"feat_real_tile_diff/{val_idx}/zoom_{num_tiles}x", real_tile_diff_img)
                        real_tile_diff = real_tile_diff.mean()
                        state.real_tile_diff[num_tiles].append(real_tile_diff)

            [red_lr_feats], _ = pca([lr_feats[z].unsqueeze(0)], fit_pca=fit_pca)
            [red_hr_feats], _ = pca([hr_feats[z].unsqueeze(0)], fit_pca=fit_pca)
            [red_bl_hr_feats], _ = pca([bl_hr_feats[z].unsqueeze(0)], fit_pca=fit_pca)
            [red_lr_jit_feats], _ = pca([lr_jit_feats[z].unsqueeze(0)], fit_pca=fit_pca)
            [red_hr_jit_feats], _ = pca([hr_jit_feats[z].unsqueeze(0)], fit_pca=fit_pca)
            [red_down_jit_feats], _ = pca([down_jit_feats[z].unsqueeze(0)], fit_pca=fit_pca)

            viz.add_image(f"image/{val_idx}", self.inv_input_conditioner(img[z].unsqueeze(0))[0])
            viz.add_image(f"lr_viz/feats/{val_idx}", red_lr_feats[0])
            viz.add_image(f"hr_viz/feats/{val_idx}", red_hr_feats[0])
            viz.add_image(f"bl_hr_viz/feats/{val_idx}", red_bl_hr_feats[0])
            viz.add_image(f"jit_viz/jit_image/{val_idx}", self.inv_input_conditioner(jit_img[z].unsqueeze(0))[0])
            viz.add_image(f"jit_viz/lr_jit_feats/{val_idx}", red_lr_jit_feats[0])
            viz.add_image(f"jit_viz/hr_jit_feats/{val_idx}", red_hr_jit_feats[0])
            viz.add_image(f"jit_viz/down_jit_feats/{val_idx}", red_down_jit_feats[0])
            viz.add_image(f"jit_viz/overlay/{val_idx}", self.inv_input_conditioner(overlay_jit[z].unsqueeze(0))[0])
            viz.add_image(f"bl_diff/{val_idx}", hr_bl_diff_img[z])

            if self.use_cond_bias_transform:
                for k in range(len(cond_deltas[1])):
                    curr_delta = cond_deltas[1][k]
                    num_tiles = curr_delta.shape[-1] // lr_feats.shape[-1]
                    [red_cond_delta], _ = pca([curr_delta[z].unsqueeze(0)], fit_pca=fit_pca)
                    viz.add_image(f"cond_delta/{val_idx}/tile_{num_tiles}x", red_cond_delta[0])

        if self.predicted_uncertainty:
            for n, im_scales in scales.items():
                for i, scale in enumerate(im_scales):
                    unc = 1 / scale
                    unc = unc / unc.amax().clamp_min(1)  # For uncertainties < 1, don't rescale below that
                    cimg = self.inv_input_conditioner(img[i][None])[0]
                    rs_unc = F.interpolate(unc[None, None], cimg.shape[-2:], mode='bilinear', align_corners=False)[0]
                    cimg = cimg * rs_unc
                    cimg = torch.cat([cimg, rs_unc.expand_as(cimg)], dim=-1)
                    val_idx = lr_feats.shape[0] * batch_idx + i
                    viz.add_image(f"unc/{n}/{val_idx}", cimg)

        if isinstance(self.downsampler, SimpleDownsampler):
            viz.add_image(
                f"down/filter/{batch_idx}",
                prep_image(self.downsampler.get_kernel().squeeze(), subtract_min=False),
            )

        if self.use_bias_buffer:
            [pca_bias], _ = pca([self.bias_buffer.buffer])
            viz.add_image(f"bias/global", pca_bias[0])

        state.lr_feats.append(lr_feats.detach())
        state.hr_feats.append(hr_feats.detach())

        if self.use_cond_bias_transform:
            self.cb_deltas.clear()

        return state


    def finalize_validation(self, viz: VizWriter, global_step: int, state: Optional[ValState] = None):
        state.finish()

        pass


def wrap_model(model: nn.Module):
    model.cuda()
    if get_world_size() > 1:
        is_trainable = any(p.requires_grad for p in model.parameters())
        if is_trainable:
            dev = torch.cuda.current_device()
            nn.SyncBatchNorm.convert_sync_batchnorm(model)
            return DistributedDataParallel(model, [dev], dev)
    return model

first_val = True
first_val_step = -1

pcas = dict()

def tilize_deltas(upsampler: nn.Module, deltas: List[Tuple[torch.Tensor, torch.Tensor]]):
    if isinstance(upsampler, DistributedDataParallel):
        upsampler = upsampler.module

    ups_factors = [1] + getattr(upsampler, 'upsample_factors', [upsampler.upsample_factor])

    # zip(*val) is a transpose, converting a list of tuples into a tuple of lists
    deltas = [torch.cat(d) for d in zip(*deltas)]
    deltas = [untile(d, ups_factors) for d in deltas]
    return deltas
