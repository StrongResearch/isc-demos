from cycling_utils import TimestampedTimer

timer = TimestampedTimer()
timer.report('importing Timer')

import os
import torch
# import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
# from monai.config import print_config
from monai.data import DataLoader #, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
# from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
# from generative.losses.adversarial_loss import PatchAdversarialLoss
# from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet # , PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from cycling_utils import InterruptableDistributedSampler, Timer, MetricsTracker
# from loops import train_generator_one_epoch, evaluate_generator
from loops import train_diffusion_one_epoch, evaluate_diffusion
import utils

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="Latent Diffusion Model Training", add_help=add_help)
    parser.add_argument("--resume", type=str, help="path of checkpoint", required=True) # for checkpointing
    parser.add_argument("--gen-load-path", type=str, help="path of checkpoint", dest="gen_load_path") # for checkpointing
    parser.add_argument("--prev-resume", default=None, help="path of previous job checkpoint for strong fail resume", dest="prev_resume") # for checkpointing
    parser.add_argument("--tboard-path", default=None, help="path for saving tensorboard logs", dest="tboard_path") # for checkpointing
    parser.add_argument("--data-path", default="/mnt/Datasets/Open-Datasets/MONAI", type=str, help="dataset path", dest="data_path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    return parser

def compute_scale_factor(autoencoder, train_loader, device):
    with torch.no_grad():
        check_data = first(train_loader)
        z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
    scale_factor = 1 / torch.std(z)
    return scale_factor.item()

timer.report('importing everything else')

def main(args, timer):

    utils.init_distributed_mode(args) # Sets args.distributed among other things
    assert args.distributed # don't support cycling when not distributed for simplicity

    device = torch.device(args.device)

    timer.report('preliminaries')

    # Maybe this will work?
    set_determinism(42)

    channel = 0  # 0 = "Flair" channel
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
    preprocessing_transform = transforms.Compose([
            transforms.LoadImaged(keys="image", image_only=False), # image_only current default will change soon, so including explicitly
            transforms.EnsureChannelFirstd(keys="image"),
            transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            transforms.AddChanneld(keys="image"),
            transforms.EnsureTyped(keys="image"),
            transforms.Orientationd(keys="image", axcodes="RAS"),
            transforms.CenterSpatialCropd(keys="image", roi_size=(240, 240, 100)),
            transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=100, b_min=0, b_max=1),
    ])
            
    crop_transform = transforms.Compose([
            transforms.DivisiblePadd(keys="image", k=[32,32,1]),
            transforms.RandSpatialCropd(keys="image", roi_size=(256, 256, 1), random_size=False), # Each of the 100 slices will be randomly sampled.
            transforms.SqueezeDimd(keys="image", dim=3),
            # transforms.RandFlipd(keys="image", prob=0.5, spatial_axis=0),
            # transforms.RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    ])

    preprocessing = transforms.Compose([preprocessing_transform, crop_transform])

    train_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="training", cache_rate=0.0,
        num_workers=8, download=False, seed=0, transform=preprocessing,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="validation", cache_rate=0.0,
        num_workers=8, download=False, seed=0, transform=preprocessing,
    )

    timer.report('build datasets')

    train_sampler = InterruptableDistributedSampler(train_ds)
    val_sampler = InterruptableDistributedSampler(val_ds)

    timer.report('build samplers')

    # Original trainer had batch size = 2 * 50. Using 11 nodes x 6 GPUs x batch size 2 => eff batch size = 132
    train_loader = DataLoader(train_ds, batch_size=2, sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=1)
    # check_data = first(train_loader) # Used later

    timer.report('build dataloaders')

    # Auto-encoder definition
    generator = AutoencoderKL(
        spatial_dims=2, in_channels=1, out_channels=1, num_channels=(64, 128, 256), 
        latent_channels=1, num_res_blocks=2, norm_num_groups=32, norm_eps=1e-06,
        attention_levels=(False, False, False), with_encoder_nonlocal_attn=True, 
        with_decoder_nonlocal_attn=True,
    )
    # saved_generator_checkpoint = torch.load("/output_brats_mri_2d_gen/exp_1645/checkpoint.isc", map_location="cpu")
    saved_generator_checkpoint = torch.load(args.gen_load_path, map_location="cpu")
    generator.load_state_dict(saved_generator_checkpoint["generator"])
    generator = generator.to(device)

    timer.report('generator to device')

    # Diffusion model (unet)
    unet = DiffusionModelUNet(
        spatial_dims=2, in_channels=1, out_channels=1, num_res_blocks=2, 
        num_channels=(32, 64, 128, 256), attention_levels=(False, True, True, True), 
        num_head_channels=(0, 32, 32, 32),
    )
    unet = unet.to(device)

    timer.report('unet to device')

    # Prepare for distributed training
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    unet_without_ddp = unet
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[args.gpu], find_unused_parameters=True)
        unet_without_ddp = unet.module

    timer.report('unet prepped for distribution')

    # Optimizers
    optimizer_u = torch.optim.Adam(unet_without_ddp.parameters(), lr=5e-5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_u, milestones=[1000], gamma=0.1)

    # For mixed precision training
    scaler_u = GradScaler()

    timer.report('optimizer, lr_scheduler and grad scaler')

    # Init metric tracker
    metrics = {'train': MetricsTracker(), 'val': MetricsTracker()}

    # RETRIEVE GENERATOR CHECKPOINT FROM PREVIOUS JOB
    
    # RETRIEVE CHECKPOINT
    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.resume and os.path.isfile(args.resume): # If we're resuming...
        checkpoint = torch.load(args.resume, map_location="cpu")
    elif args.prev_resume and os.path.isfile(args.prev_resume):
        checkpoint = torch.load(args.prev_resume, map_location="cpu")
    if checkpoint is not None:
        args.start_epoch = checkpoint["epoch"]
        unet_without_ddp.load_state_dict(checkpoint["unet"])
        optimizer_u.load_state_dict(checkpoint["optimizer_u"])
        scaler_u.load_state_dict(checkpoint["scaler_u"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # Metrics
        metrics = checkpoint["metrics"]

    timer.report('checkpoint retrieval')

    ## -- TRAINING THE DIFFUSION MODEL - ##

    n_diff_epochs = 200
    diff_val_interval = 1

    # Prepare LatentDiffusionInferer
    
    # with torch.no_grad():
    #     with autocast(enabled=True):
    #         z = generator.encode_stage_2_inputs(check_data["image"].to(device))
    # scale_factor = 1 / torch.std(z)

    scale_factor = compute_scale_factor(generator, train_loader, device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    timer.report('building inferer')

    for epoch in range(args.start_epoch, n_diff_epochs):

        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = TimestampedTimer("Start training")
            unet, timer, metrics = train_diffusion_one_epoch(
                args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
                train_sampler, val_sampler, lr_scheduler, device, timer, metrics
            )
            timer.report(f'training unet for epoch {epoch}')

            if epoch % diff_val_interval == 0:
                with val_sampler.in_epoch(epoch):
                    timer = TimestampedTimer("Start evaluation")
                    timer, metrics = evaluate_diffusion(
                        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
                        train_sampler, val_sampler, lr_scheduler, device, timer, metrics
                    )
                    timer.report(f'evaluating unet for epoch {epoch}')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
