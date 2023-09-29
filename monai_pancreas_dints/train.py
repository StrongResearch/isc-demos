from cycling_utils import TimestampedTimer

timer = TimestampedTimer()
timer.report('importing Timer')

import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Sequence, Union
from scipy import ndimage

import monai
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from monai import transforms
from monai.bundle import ConfigParser
from monai.networks.nets import TopologyInstance, DiNTS
from monai.losses import DiceCELoss
from monai.data import ThreadDataLoader, partition_dataset, DataLoader, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
from monai.utils import set_determinism

from torch.nn.parallel import DistributedDataParallel


import argparse
from cycling_utils import InterruptableDistributedSampler, MetricsTracker
from loops import train_one_epoch, evaluate
from pathlib import Path
import utils

# def get_args_parser(add_help=True):
#     parser = argparse.ArgumentParser(description="DiNTS train", add_help=add_help)
#     parser.add_argument("--resume", type=str, help="path of checkpoint", required=True) # for checkpointing
#     parser.add_argument("--prev-resume", default=None, help="path of previous job checkpoint for strong fail resume", dest="prev_resume") # for checkpointing
#     parser.add_argument("--tboard-path", default=None, help="path for saving tensorboard logs", dest="tboard_path") # for checkpointing
#     parser.add_argument("--data_list_file_path", default=None, help="path for retreiving pre-prepared data list", dest="data_list_file_path")
#     return parser

def run(config_file: Union[str, Sequence[str]], resume=None, prev_resume=None, tboard_path=None):
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = ConfigParser()
    parser.read_config(config_file)

    args = {
        "start_epoch": 0,
        "resume": resume,
        "prev_resume": prev_resume,
        "tboard_path": tboard_path,
        "arch_ckpt_path": "/models/search_code.pt",
        "num_epochs_per_validation": 10,

        "learning_rate": 0.025,
        "data_list_file_path": "/configs/dataset_0.json",
        "dataset_dir": "/mnt/Datasets/Open-Datasets/MONAI/Task07_Pancreas",
    }

    utils.init_distributed_mode(args) # Sets args.distributed among other things
    assert args["distributed"] # don't support cycling when not distributed for simplicity
    device = torch.device(args["device"])

    train_datalist = load_decathlon_datalist(args["data_list_file_path"], data_list_key='training', base_dir=args["dataset_dir"])
    val_datalist = load_decathlon_datalist(args["data_list_file_path"], data_list_key='validation', base_dir=args["dataset_dir"])

    train_preprocessing = parser.get_parsed_content("train_preprocessing")
    val_preprocessing = parser.get_parsed_content("val_preprocessing")
    postprocessing = parser.get_parsed_content("postprocessing")

    n_workers = 1
    cache_rate = 0.0
    train_ds = monai.data.CacheDataset(data=train_datalist, transform=train_preprocessing, cache_rate=cache_rate, num_workers=n_workers)
    val_ds = monai.data.CacheDataset(data=val_datalist, transform=val_preprocessing, cache_rate=cache_rate, num_workers=n_workers)

    train_sampler = InterruptableDistributedSampler(train_ds)
    val_sampler = InterruptableDistributedSampler(val_ds)

    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=1)

    arch_ckpt = torch.load(args["arch_ckpt_path"], map_location=device)
    dints_space = TopologyInstance(arch_code=[arch_ckpt['arch_code_a'], arch_ckpt['arch_code_c']], channel_mul=1.0, num_blocks=12, num_depths=4, use_downsample=True, device=device)
    model = DiNTS(dints_space, in_channels=1, num_classes=3, use_downsample=True, node_a=arch_ckpt['node_a'])
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss_func = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, squared_pred=True, batch=True, smooth_nr=1e-05, smooth_dr=1e-05)

    post_pred = transforms.Compose([transforms.Activationsd(softmax=True), transforms.AsDiscrete(to_onehot=args["output_classes"], argmax=True)])
    post_label = transforms.Compose([transforms.Activationsd(softmax=False), transforms.AsDiscrete(to_onehot=args["output_classes"], argmax=False)])

    model_without_ddp = model
    if args["distributed"]:
        model = DistributedDataParallel(model, device_ids=[args["gpu"]], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = torch.optim.SGD(
        model_without_ddp.weight_parameters(), lr=args["learning_rate"] * args["world_size"], momentum=0.9, weight_decay=0.00004
    )
    dints_space.log_alpha_a.requires_grad = False
    dints_space.log_alpha_c.requires_grad = False

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, milestones=[80 * args["world_size"]], gamma=0.5)

    # amp
    if args["amp"]:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("[info] amp enabled")

    val_interval = args["num_epochs_per_validation"]

    # Init metric trackers
    train_metrics = MetricsTracker()
    val_metric = torch.zeros((args["output_classes"] - 1) * 2, dtype=torch.float, device=device)

    # RETRIEVE CHECKPOINT
    Path(args["resume"]).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    if args["resume"] and os.path.isfile(args["resume"]): # If we're resuming...
        checkpoint = torch.load(args["resume"], map_location="cpu")
    elif args["prev_resume"] and os.path.isfile(args["prev_resume"]):
        checkpoint = torch.load(args["prev_resume"], map_location="cpu")

    if checkpoint is not None:
        args["start_epoch"] = checkpoint["epoch"]
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        scaler.load_state_dict(checkpoint["scaler"])
        train_metrics = checkpoint["train_metrics"]
        val_metric = checkpoint["val_metric"]
        val_metric.to(device)

    for epoch in range(args["start_epoch"], args["num_epochs"]):

        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = TimestampedTimer("Start training")

            model, dints_space, timer, train_metrics, val_metric = train_one_epoch(
                model, optimizer, lr_scheduler,
                train_sampler, val_sampler, scaler, train_metrics, val_metric,
                epoch, train_loader, loss_func, args
            )
            timer.report(f'training for epoch {epoch}')

            if (epoch + 1) % val_interval == 0 or (epoch + 1) == args["num_epochs"]:

                with val_sampler.in_epoch(epoch):
                    timer = TimestampedTimer("Start evaluation")

                    timer = evaluate(
                        model, optimizer, lr_scheduler,
                        train_sampler, val_sampler, scaler, train_metrics, val_metric,
                        epoch, val_loader, post_pred, post_label, args,
                    )
                    timer.report(f'evaluating for epoch {epoch}')

