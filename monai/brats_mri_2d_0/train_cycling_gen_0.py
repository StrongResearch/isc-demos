from cycling_utils import Timer

timer = Timer()
timer.report('importing Timer')

import os

# import matplotlib.pyplot as plt
# import numpy as np
import torch
import torch.distributed as dist
# import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
# from monai.config import print_config
from monai.data import DataLoader# , Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler# , autocast
from pathlib import Path
# from tqdm import tqdm

# from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator # , DiffusionModelUNet
# from generative.networks.schedulers import DDPMScheduler

from cycling_utils import InterruptableDistributedSampler, Timer
from loops_0 import train_generator_one_epoch, evaluate_generator
# from loops import train_diffusion_one_epoch, evaluate_diffusion
import utils

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Latent Diffusion Model Training", add_help=add_help)

    parser.add_argument("--resume", type=str, help="path of checkpoint", required=True) # for checkpointing
    parser.add_argument("--prev-resume", default=None, help="path of previous job checkpoint for strong fail resume", dest="prev_resume") # for checkpointing
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

class MetricsTracker:
    def __init__(self, metric_names):
        self.map = {n:i for i,n in enumerate(metric_names)}
        self.local = torch.zeros(len(metric_names), dtype=torch.float16, requires_grad=False, device='cuda')
        self.agg = torch.zeros(len(metric_names), dtype=torch.float16, requires_grad=False, device='cuda')
        self.epoch_reports = []

    def update(self, metrics: dict):
        for n,v in metrics.items():
            self.local[self.map[n]] += v
        
    def reduce_and_reset_local(self):
        # Reduce over all nodes, add that to local store, and reset local
        dist.all_reduce(self.local, op=dist.ReduceOp.SUM)
        self.agg += self.local
        self.local = torch.zeros(len(self.local), dtype=torch.float16, requires_grad=False, device='cuda')
    
    def end_epoch(self):
        self.epoch_reports.append(self.agg)
        self.local = torch.zeros(len(self.local), dtype=torch.float16, requires_grad=False, device='cuda')
        self.agg = torch.zeros(len(self.local), dtype=torch.float16, requires_grad=False, device='cuda')

    def to(self, device):
        self.local = self.local.to(device)
        self.agg = self.agg.to(device)


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
            transforms.DivisiblePadd(keys="image", k=[4,4,1]),
            # transforms.RandSpatialCropSamplesd(keys="image", roi_size=(240, 240, 1), random_size=False, num_samples=26),
            transforms.RandSpatialCropd(keys="image", roi_size=(240, 240, 1), random_size=False), # Each of the 100 slices will be randomly sampled.
            transforms.SqueezeDimd(keys="image", dim=3),
            transforms.RandFlipd(keys="image", prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys="image", prob=0.5, spatial_axis=1),
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

    # Original trainer had batch size = 26. Using 9 nodes x batch size 3 = eff batch size = 27
    train_loader = DataLoader(train_ds, batch_size=3, sampler=train_sampler, num_workers=1)
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
    generator = generator.to(device)

    timer.report('generator to device')

    # Discriminator definition
    discriminator = PatchDiscriminator(
        spatial_dims=2, num_layers_d=3, num_channels=32, 
        in_channels=1, out_channels=1, norm="INSTANCE"
    )
    discriminator = discriminator.to(device)

    timer.report('discriminator to device')

    # # Diffusion model (unet)
    # unet = DiffusionModelUNet(
    #     spatial_dims=2, in_channels=3, out_channels=3, num_res_blocks=2, 
    #     num_channels=(128, 256, 512),attention_levels=(False, True, True), 
    #     num_head_channels=(0, 256, 512),
    # )
    # unet = unet.to(device)

    # timer.report('unet to device')

    # Autoencoder loss functions
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = PerceptualLoss(
        spatial_dims=2, network_type="resnet50", pretrained=True, #ImageNet pretrained weights used
    )
    perceptual_loss.to(device)

    timer.report('loss functions')

    # Prepare for distributed training
    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    # unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    generator_without_ddp = generator
    discriminator_without_ddp = discriminator
    # unet_without_ddp = unet
    if args.distributed:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu], find_unused_parameters=True) # find_unused_parameters necessary for monai training
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu], find_unused_parameters=True) # find_unused_parameters necessary for monai training
        # unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[args.gpu], find_unused_parameters=True)
        generator_without_ddp = generator.module
        discriminator_without_ddp = discriminator.module
        # unet_without_ddp = unet.module

    timer.report('models prepped for distribution')

    # Optimizers
    optimizer_g = torch.optim.Adam(generator_without_ddp.parameters(), lr=5e-5)
    optimizer_d = torch.optim.Adam(discriminator_without_ddp.parameters(), lr=5e-5)
    # optimizer_u = torch.optim.Adam(unet_without_ddp.parameters(), lr=1e-4)

    timer.report('optimizers')

    # For mixed precision training
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    # scaler_u = GradScaler()

    timer.report('grad scalers')

    # Init metric tracker
    train_metrics = MetricsTracker(["train_images_seen", "epoch_loss", "gen_epoch_loss", "disc_epoch_loss"])
    val_metrics = MetricsTracker(["val_images_seen", "val_loss"])
    metrics = {'train': train_metrics, 'val': val_metrics}

    # RETRIEVE CHECKPOINT
    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.resume and os.path.isfile(args.resume): # If we're resuming...
        checkpoint = torch.load(args.resume, map_location="cpu")
    elif args.prev_resume and os.path.isfile(args.prev_resume):
        checkpoint = torch.load(args.prev_resume, map_location="cpu")
    if checkpoint is not None:
        args.start_epoch = checkpoint["epoch"]
        generator_without_ddp.load_state_dict(checkpoint["generator"])
        discriminator_without_ddp.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        scaler_g.load_state_dict(checkpoint["scaler_g"])
        scaler_d.load_state_dict(checkpoint["scaler_d"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        # Metrics
        metrics = checkpoint["metrics"]
        metrics["train"].to(device)
        metrics["val"].to(device)

    timer.report('checkpoint retrieval')

    ## -- TRAINING THE AUTO-ENCODER - ##

    n_gen_epochs = 200
    gen_val_interval = 1

    for epoch in range(args.start_epoch, n_gen_epochs):

        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = Timer("Start training")
            generator, timer, metrics = train_generator_one_epoch(
                args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
                scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer, metrics
            )
            timer.report(f'training generator for epoch {epoch}')

            if epoch % gen_val_interval == 0: # Eval every epoch
                with val_sampler.in_epoch(epoch):
                    timer = Timer("Start evaluation")
                    timer, metrics = evaluate_generator(
                        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
                        scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer, metrics
                    )
                    timer.report(f'evaluating generator for epoch {epoch}')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
