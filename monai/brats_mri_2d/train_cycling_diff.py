from cycling_utils import Timer

timer = Timer()
timer.report('importing Timer')

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from cycling_utils import InterruptableDistributedSampler, Timer
# from loops import train_generator_one_epoch, evaluate_generator
from loops import train_diffusion_one_epoch, evaluate_diffusion
import utils

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Latent Diffusion Model Training", add_help=add_help)

    parser.add_argument("--resume", type=str, help="path of checkpoint", required=True) # for checkpointing
    # parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--data-path", default="/mnt/Datasets/Open-Datasets/MONAI", type=str, help="dataset path", dest="data_path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    # parser.add_argument("-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size", dest="batch_size")
    # parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    # parser.add_argument("--print-freq", default=1, type=int, help="print frequency")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
 
    return parser

timer.report('importing everything else')

def main(args, timer):

    # ## Distributed training prelims
    # if args.output_dir:
    #     utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args) # Sets args.distributed among other things
    assert args.distributed # don't support cycling when not distributed for simplicity

    device = torch.device(args.device)

    timer.report('preliminaries')

    # Maybe this will work?
    set_determinism(42)

    channel = 0  # 0 = Flair
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
    train_transforms = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False), # image_only current default will change soon, so including explicitly
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Lambdad(keys=["image"], func=lambda x: x[channel, None, :, :, :]),
            transforms.EnsureTyped(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 44)),
            transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 1), random_size=False),
            transforms.Lambdad(keys=["image", "label"], func=lambda x: x.squeeze(-1)),
            transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
            transforms.Lambdad(keys=["slice_label"], func=lambda x: 1.0 if x.sum() > 0 else 0.0),
    ])

    train_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="training", cache_rate=0.0,
        num_workers=4, download=False, seed=0, transform=train_transforms,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="validation", cache_rate=0.0,
        num_workers=4, download=False, seed=0, transform=train_transforms,
    )

    # ## SUBSET FOR TESTING
    # train_ds = torch.utils.data.Subset(train_ds, torch.arange(2*9*3)) # batch_size x nodes x iterations
    # val_ds = torch.utils.data.Subset(val_ds, torch.arange(1*9*2)) # batch_size x nodes x iterations

    timer.report('build datasets')

    train_sampler = InterruptableDistributedSampler(train_ds)
    val_sampler = InterruptableDistributedSampler(val_ds)

    timer.report('build samplers')

    train_loader = DataLoader(train_ds, batch_size=2, sampler=train_sampler, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=args.workers)
    # check_data = first(train_loader) # Used later

    timer.report('build dataloaders')

    # # Auto-encoder definition
    # generator = AutoencoderKL(
    #     spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128, 128, 256), latent_channels=3,
    #     num_res_blocks=2, attention_levels=(False, False, False),with_encoder_nonlocal_attn=False,
    #     with_decoder_nonlocal_attn=False,
    # )
    # generator = generator.to(device)

    # timer.report('generator to device')

    # # Discriminator definition
    # discriminator = PatchDiscriminator(
    #     spatial_dims=2, num_layers_d=3, num_channels=64, 
    #     in_channels=1, out_channels=1
    # )
    # discriminator = discriminator.to(device)

    # timer.report('discriminator to device')

    # Diffusion model (unet)
    unet = DiffusionModelUNet(
        spatial_dims=2, in_channels=3, out_channels=3, num_res_blocks=2, 
        num_channels=(128, 256, 512),attention_levels=(False, True, True), 
        num_head_channels=(0, 256, 512),
    )
    unet = unet.to(device)

    timer.report('unet to device')

    # # Autoencoder loss functions
    # adv_loss = PatchAdversarialLoss(criterion="least_squares")
    # perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
    # perceptual_loss.to(device)

    # timer.report('loss functions')

    # Prepare for distributed training
    # generator_without_ddp = generator
    # discriminator_without_ddp = discriminator
    unet_without_ddp = unet
    # perceptual_loss_without_ddp = perceptual_loss
    if args.distributed:
        # generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
        # discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
        unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[args.gpu])
        # perceptual_loss = torch.nn.parallel.DistributedDataParallel(perceptual_loss, device_ids=[args.gpu])
        # generator_without_ddp = generator.module
        # discriminator_without_ddp = discriminator.module
        unet_without_ddp = unet.module
        # perceptual_loss_without_ddp = perceptual_loss.module

    timer.report('unet prepped for distribution')

    # Optimizers
    # optimizer_g = torch.optim.Adam(generator_without_ddp.parameters(), lr=1e-4)
    # optimizer_d = torch.optim.Adam(discriminator_without_ddp.parameters(), lr=5e-4)
    optimizer_u = torch.optim.Adam(unet_without_ddp.parameters(), lr=1e-4)

    timer.report('optimizers')

    # For mixed precision training
    # scaler_g = GradScaler()
    # scaler_d = GradScaler()
    scaler_u = GradScaler()

    timer.report('grad scalers')

    # Init tracking metrics
    # ???

    # RETRIEVE CHECKPOINT
    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    if args.resume and os.path.isfile(args.resume): # If we're resuming...
        checkpoint = torch.load(args.resume, map_location="cpu")
        args.start_epoch = checkpoint["epoch"]
        unet_without_ddp.load_state_dict(checkpoint["unet"], strict=not args.test_only)
        optimizer_u.load_state_dict(checkpoint["optimizer_u"])
        scaler_u.load_state_dict(checkpoint["scaler_u"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        train_images_seen = checkpoint["train_images_seen"]
        val_images_seen = checkpoint["val_images_seen"]
        # Metrics
        # ???

    timer.report('checkpoint retrieval')

    # ## -- TRAINING THE AUTO-ENCODER - ##

    # n_gen_epochs = 100
    # gen_val_interval = 1

    # for epoch in range(n_gen_epochs):

    #     print('\n')
    #     print(f"EPOCH (gen) :: {epoch}")
    #     print('\n')

    #     with train_sampler.in_epoch(epoch):
    #         timer = Timer()
    #         generator, timer, _, _, _ = train_generator_one_epoch(
    #             epoch, generator, discriminator, optimizer_g, optimizer_d, 
    #             scaler_g, scaler_d, train_loader, perceptual_loss, adv_loss, device, timer
    #         )
    #         timer.report(f'training generator for epoch {epoch}')

    #         if epoch % gen_val_interval == 0: # Eval every epoch
    #             with val_sampler.in_epoch(epoch):
    #                 timer = Timer()
    #                 timer, _ = evaluate_generator(epoch, generator, val_loader, device, timer)
    #                 timer.report(f'evaluating generator for epoch {epoch}')


    ## -- TRAINING THE DIFFUSION MODEL - ##

    n_diff_epochs = 200
    diff_val_interval = 1

    # Prepare LatentDiffusionInferer
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
    with torch.no_grad():
        with autocast(enabled=True):
            z = generator.encode_stage_2_inputs(check_data["image"].to(device))
    scale_factor = 1 / torch.std(z)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    timer.report('building inferer')

    for epoch in range(n_diff_epochs):

        print('\n')
        print(f"EPOCH (diff) :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = Timer()
            unet, timer, _ = train_diffusion_one_epoch(
                epoch, unet, generator, optimizer_u, 
                inferer, scaler_u, train_loader, device, timer
            )
            timer.report(f'training unet for epoch {epoch}')

            if epoch % diff_val_interval == 0:
                with val_sampler.in_epoch(epoch):
                    timer = Timer()
                    _ = evaluate_diffusion(epoch, unet, generator, inferer, val_loader, device)
                    timer.report(f'evaluating unet for epoch {epoch}')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)