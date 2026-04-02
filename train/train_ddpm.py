
from ddpm import FineStage, Trainer
from ddpm.new_unet3d import Unet3D

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from train.get_dataset import get_dataset
import torch
import os
from ddpm.unet import UNet

# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    if cfg.model.denoising_fn == 'old_Unet3D':
        model = Unet3D(
            dim=64,  # 64
            dim_mults=cfg.model.dim_mults,  # [1,2,4,8]
            channels=8,  # 1
            out_dim=cfg.model.out_dim  # 1
        ).cuda()
        fine_model = Unet3D(
            dim=128,  # 64
            dim_mults=cfg.model.dim_mults,  # [1,2,4,8]
            channels=24,  # 1
            out_dim=cfg.model.out_dim  # 1
        ).cuda()


    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    diffusion = FineStage(
        fit_model=model,
        fine_model=fine_model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        vqgan_ckpt_bp=cfg.model.vqgan_ckpt_bp,
        vqgan_ckpt_bp025=cfg.model.vqgan_ckpt_bp025,
        image_size=cfg.model.diffusion_img_size,
        input_img_size=cfg.model.input_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        pose_timesteps=cfg.model.pose_timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).cuda()

    train_dataset, val_dataset, visual_dataset, real_dataset = get_dataset(cfg)
    print(train_dataset[0]["data"].shape)

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,
        real_dataset=real_dataset,
        visual_dataset=visual_dataset,
        train_batch_size=cfg.model.batch_size,
        test_batch_size=cfg.model.test_batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        val_every=cfg.model.val_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        # logger=cfg.model.logger
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    if cfg.model.coarse_model:
        trainer.load_coarse(cfg.model.coarse_model)

    trainer.val()


if __name__ == '__main__':
    run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
