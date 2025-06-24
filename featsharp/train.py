# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NSCLv2
# found in the LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
import collections
import gc
import os
from os.path import join
import sys
from tqdm import tqdm
from typing import Dict

import debugpy
import hydra
import torch
import torch.amp
from torch import distributed as dist
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torchvision.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import wandb
from PIL import Image

from featsharp.data.data_pipe_v2 import get_data_pipeline, PipelineConfig
from featsharp.data.utils import FewImageDataset
from featsharp.training_harness import TrainingHarness
from featsharp.util import (
    norm_of_tensors,
    RollingAvg,
    seed_everything, DistributedDataParallel,
    to_device,
)
from featsharp.visualization import append_kwargs, get_visualizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch._dynamo.config.optimize_ddp=False

DTYPE = torch.float32
Image.MAX_IMAGE_PIXELS = 258760000


@hydra.main(config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    hydra.main()

    if rank == 0:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(cfg.output_root)

    seed_everything(seed=local_rank)

    cfg.num_gpus = min(world_size, torch.cuda.device_count())

    unc_suff = '_unc' if cfg.outlier_detection else ''
    name = getattr(cfg, 'name', None)
    if name is None:
        name = (f"{cfg.model_type}_{cfg.upsampler_type}-{cfg.upsample_factor}x_"
                f"norm_{cfg.normalizer_mode}"
                f"{unc_suff}")

    if rank == 0:
        print(f'Job Name: {name}')

    dir_prefix = getattr(cfg, 'dir_prefix', '')

    log_dir = join(cfg.output_root, f"{dir_prefix}{name}/logs")
    chkpt_dir = join(cfg.output_root, f"{dir_prefix}{name}/checkpoints")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    if cfg.distance_fn == 'mse':
        distance_fn = F.mse_loss
    elif cfg.distance_fn == 'smooth_l1':
        distance_fn = F.smooth_l1_loss
    elif cfg.distance_fn == 'l1':
        distance_fn = F.l1_loss
    else:
        raise ValueError(f'Unknown distance function: {cfg.distance_fn}')

    trainer = TrainingHarness(
        model_type=cfg.model_type,
        model_extra_kwargs=getattr(cfg, 'model_args', dict()),
        activation_type=cfg.activation_type,
        upsampler=cfg.upsampler_type,
        upsample_factor=cfg.upsample_factor,
        downsampler=cfg.downsampler_type,
        input_size=getattr(cfg, 'input_size', None),
        ds_kernel_size=getattr(cfg, 'downsampler_kernel_size', 9),
        random_projection=cfg.random_projection,
        predicted_uncertainty=cfg.outlier_detection,
        crf_weight=cfg.crf_weight,
        tv_weight=cfg.tv_weight,
        mmd_weight=cfg.mmd_weight,
        normalizer_mode=cfg.normalizer_mode,
        distance_fn=distance_fn,
        n_jitters=cfg.n_jitters,
        bias_buffer=getattr(cfg, 'bias_buffer', True),
        grid_weight=getattr(cfg, 'grid_weight', 1.),
        spectral_reparam=getattr(cfg, 'spectral_reparam', False),
        cond_bias_weight=getattr(cfg, 'cond_bias_weight', 0.0),
    )
    trainer.to(DTYPE)

    load_size = getattr(trainer.model, 'input_size', getattr(cfg, 'input_size', None))
    cfg.__setattr__('input_size', load_size)

    num_steps = cfg.num_training_steps * cfg.epochs

    precision = getattr(cfg, 'precision', 32)
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 16)

    optim_params = trainer.trainable_params(lr=cfg.lr)
    trainable_params = []
    for group in optim_params:
        trainable_params.extend(group['params'])

    trainable_norm = norm_of_tensors(trainable_params)
    print(f'Model norm: {trainable_norm.item()}')

    optim = torch.optim.NAdam(optim_params, lr=cfg.lr)

    warmup = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_steps)
    cos_sched = CosineAnnealingLR(optim, num_steps - cfg.warmup_steps)
    lr_sched = SequentialLR(optim, [warmup, cos_sched], milestones=[cfg.warmup_steps])

    last_chk_path = os.path.join(chkpt_dir, 'last.pth.tar')
    if not getattr(cfg, 'restart', False) and os.path.exists(last_chk_path):
        last_chk = torch.load(last_chk_path, map_location='cpu')

        trainer.load_state_dict(last_chk['model_state'], strict=False)
        scaler.load_state_dict(last_chk['grad_scaler_state'])
        lr_sched.load_state_dict(last_chk['lr_sched_state'])
        optim.load_state_dict(last_chk['optim_state'])
        global_step = last_chk['global_step']
    else:
        last_chk = dict()
        global_step = 0

    trainer.cuda()
    viz = get_visualizer(cfg.viz, state_dict=last_chk.get('viz', dict()), log_dir=log_dir, config=OmegaConf.to_container(cfg))

    seed_everything(seed=rank)

    image_load_size = load_size * getattr(trainer.upsampler, 'input_upsample_factor', 1)

    normalizer_init = getattr(trainer.normalizer, 'initialize', None)
    if normalizer_init is not None:
        num_norm_steps = getattr(cfg, 'normalizer_args', dict()).get('num_batches', cfg.num_training_steps)

        if trainer.normalizer.requires_loader(trainer.model, cfg):
            norm_loader, _ = get_data_pipeline(
                PipelineConfig(steps_per_epoch=num_norm_steps, workers=cfg.num_workers),
                ds_listing=getattr(cfg, 'dist_train_data_dir', cfg.train_data_dir),
                input_sizes=[load_size],
                patch_sizes=[trainer.patch_size],
                batch_size=cfg.batch_size,
                is_train=[True],
                is_teacher=[False],
                prefetch=False,
                seed=71 + rank,
                full_equivariance=[False],
            )
        else:
            norm_loader = None

        normalizer_init(trainer.model, norm_loader, cfg)

    seed_everything(seed=rank, workers=True)

    val_frequency: int = cfg.val_frequency

    loader, _ = get_data_pipeline(
        PipelineConfig(steps_per_epoch=cfg.num_training_steps, workers=cfg.num_workers),
        ds_listing=cfg.train_data_dir,
        input_sizes=[image_load_size] * (cfg.n_jitters + 1),
        patch_sizes=[trainer.patch_size] * (cfg.n_jitters + 1),
        batch_size=cfg.batch_size,
        is_train=[True] * (cfg.n_jitters + 1),
        is_teacher=[False] * (cfg.n_jitters + 1),
        prefetch=False,
        seed=35 + rank + global_step,
        full_equivariance=[False] + [True] * cfg.n_jitters,
        epoch=global_step // val_frequency - 1,
    )

    val_loader, _ = get_data_pipeline(
        PipelineConfig(steps_per_epoch=cfg.num_training_steps, workers=cfg.num_workers),
        ds_listing=getattr(cfg, 'val_data_dir', cfg.train_data_dir),
        input_sizes=[image_load_size] * (cfg.n_jitters + 1),
        patch_sizes=[trainer.patch_size] * (cfg.n_jitters + 1),
        batch_size=cfg.batch_size,
        is_train=[True] * (cfg.n_jitters + 1),
        is_teacher=[False] * (cfg.n_jitters + 1),
        prefetch=False,
        seed=35 + rank,
        full_equivariance=[False] + [True] * cfg.n_jitters,
        epoch=-1,
        data_weight_mode=None,
    )
    val_loader = DataLoader(
        FewImageDataset(val_loader, 4),
        1,
        shuffle=False,
        num_workers=0,
    )

    num_stages = (num_steps - global_step) // val_frequency

    data_iter = iter(loader)

    metric_frequency = getattr(cfg, 'metric_frequency', 20)
    metrics = RollingAvg(metric_frequency)

    def backward(loss: torch.Tensor, retain_graph: bool = False):
        assert torch.all(torch.isfinite(loss)), "The loss wasn't finite!"

        scaled = scaler.scale(loss)
        scaled.backward(retain_graph=retain_graph)

    def run_validation(step: int):
        trainer.eval()
        with append_kwargs(viz, step=step):
            val_state = None
            for i, batch in enumerate(val_loader):
                batch = to_cuda(batch)
                val_state = trainer.forward_validation(batch, batch_idx=i, viz=viz, global_step=step, state=val_state)

            trainer.finalize_validation(viz=viz, global_step=step, state=val_state)

            viz.commit()

    if global_step == 0 and not getattr(cfg, 'skip_first_val', False):
        run_validation(global_step)

    progress = tqdm(total=num_steps, disable=rank > 0)
    progress.update(global_step)
    for stage in range(num_stages):
        trainer.train()

        for _ in range(global_step, (global_step + val_frequency) // val_frequency * val_frequency):
            try:
                batch = to_device(next(data_iter), device='cuda', dtype=DTYPE)
            except StopIteration:
                break

            optim.zero_grad(set_to_none=False)

            components = trainer.forward_train(
                batch, backward_fn=backward, autocast=precision == 16,
                global_step=global_step,
            )

            scaler.unscale_(optim)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, 0.0001 if global_step < 10 else 5.0)

            for key, value in components.items():
                metrics.add(key, value)

            global_step = global_step + 1

            if global_step % metric_frequency == 0:
                with append_kwargs(viz, step=global_step):
                    metrics.logall(viz.add_scalar)
                    viz.add_scalar("opt/lr", lr_sched.get_last_lr()[0])
                    viz.add_scalar("opt/grad_norm", gnorm.item())

                    if global_step % val_frequency != 0:
                        viz.commit()

            scaler.step(optim)
            scaler.update()
            lr_sched.step()
            progress.update()

        trainer.eval()

        run_validation(global_step)

        if rank == 0:
            chk = {
                'model_state': strip_featurizer((trainer.module if isinstance(trainer, DistributedDataParallel) else trainer).state_dict()),
                'grad_scaler_state': scaler.state_dict(),
                'lr_sched_state': lr_sched.state_dict(),
                'optim_state': optim.state_dict(),
                'viz': viz.state_dict(),
                'global_step': global_step,
                'upsample_args': {
                    'type': cfg.upsampler_type,
                    'factor': cfg.upsample_factor,
                    'normalizer': cfg.normalizer_mode,
                },
                'train_config': OmegaConf.to_container(cfg, resolve=True),
            }
            tmp_chk_path = f'{last_chk_path}.tmp'
            torch.save(chk, tmp_chk_path)
            os.system(f'mv {tmp_chk_path} {last_chk_path}')


def to_cuda(batch):
    if torch.is_tensor(batch):
        return batch.cuda()
    elif isinstance(batch, (list, tuple)):
        return [to_cuda(inner) for inner in batch]
    elif isinstance(batch, collections.Mapping):
        return {k: to_cuda(v) for k, v in batch}
    return batch


def strip_featurizer(state_dict: Dict[str, torch.Tensor]):
    key = 'model.'
    return {k: v for k, v in state_dict.items() if not k.startswith(key)}


if __name__ == "__main__":
    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--local-rank='):
            local_rank = int(arg.split('=')[1])
            del sys.argv[i]
            break

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    print(rank, world_size)

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size,
        )

        dist.barrier()
        print(rank, world_size)
        dist.barrier()

    # Prevent different nodes from stomping on each other
    ticache = os.environ.get('TORCHINDUCTOR_CACHE_DIR', None)
    if ticache:
        local_rank = rank // torch.cuda.device_count()
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = f'{ticache}/node_rank_{local_rank}'

    main()
