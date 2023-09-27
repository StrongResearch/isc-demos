# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

import monai
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from monai import transforms
from monai.bundle import ConfigParser
from monai.data import ThreadDataLoader, partition_dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from cycling_utils import InterruptableDistributedSampler, MetricsTracker
from loops import search_one_epoch, eval_search
from pathlib import Path
import utils


def run(config_file: Union[str, Sequence[str]]):
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = ConfigParser()
    parser.read_config(config_file)

    args = {
        "resume": parser["resume"],
        "arch_ckpt_path": parser["arch_ckpt_path"],
        "amp": parser["amp"],
        "data_file_base_dir": parser["data_file_base_dir"],
        "data_list_file_path": parser["data_list_file_path"],
        "determ": parser["determ"],
        "learning_rate": parser["learning_rate"],
        "learning_rate_arch": parser["learning_rate_arch"],
        "learning_rate_milestones": np.array(parser["learning_rate_milestones"]),
        "num_images_per_batch": parser["num_images_per_batch"],
        "num_epochs": parser["num_epochs"],  # around 20k iterations
        "num_epochs_per_validation": parser["num_epochs_per_validation"],
        "num_epochs_warmup": parser["num_epochs_warmup"],
        "num_sw_batch_size": parser["num_sw_batch_size"],
        "output_classes": parser["output_classes"],
        "overlap_ratio": parser["overlap_ratio"],
        "patch_size_valid": parser["patch_size_valid"],
        "ram_cost_factor": parser["ram_cost_factor"],

        "start_epoch": 0,
    }
    print("[info] GPU RAM cost factor:", args["ram_cost_factor"])

    utils.init_distributed_mode(args) # Sets args.distributed among other things
    assert args["distributed"] # don't support cycling when not distributed for simplicity
    device = torch.device(args["device"])

    train_transforms = parser.get_parsed_content("transform_train")
    val_transforms = parser.get_parsed_content("transform_validation")

    # network architecture
    if torch.cuda.device_count() > 1:
        device = torch.device(f"cuda:{dist.get_rank()}")
    else:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # deterministic training
    if args["determ"]:
        set_determinism(seed=0)

    print("[info] number of GPUs:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        # initialize the distributed training process, every GPU runs in a process
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        world_size = 1
    print("[info] world_size:", world_size)

    print("Loading json")
    with open(args["data_list_file_path"], "r") as f:
        json_data = json.load(f)

    print("Listing json")
    list_train = json_data["training"]
    list_valid = json_data["validation"]

    # training data
    print("Preparing train_files")
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(args["data_file_base_dir"], list_train[_i]["image"])
        str_seg = os.path.join(args["data_file_base_dir"], list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    train_files = files

    random.shuffle(train_files)

    # validation data
    print("Preparing val_files")
    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(args["data_file_base_dir"], list_valid[_i]["image"])
        str_seg = os.path.join(args["data_file_base_dir"], list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    val_files = files

    n_workers = 1
    cache_rate = 0.0
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=n_workers
    )
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=n_workers)

    train_sampler = InterruptableDistributedSampler(train_ds)
    val_sampler = InterruptableDistributedSampler(val_ds)

    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=1)

    # # TESTING
    # timer = TimestampedTimer("testing start")
    # for i, batch_data in enumerate(train_loader):
    #     inputs, labels = batch_data["image"], batch_data["label"]
    #     timer.report("batch")
    #     inputs.size == (1, 1, 96, 96, 96), labels.size == (1, 1, 96, 96, 96)

    model = parser.get_parsed_content("network")
    dints_space = parser.get_parsed_content("dints_space")
    loss_func = parser.get_parsed_content("loss")

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    post_pred = transforms.Compose(
        [transforms.EnsureType(), transforms.AsDiscrete(argmax=True, to_onehot=args["output_classes"])]
    )
    post_label = transforms.Compose([transforms.EnsureType(), transforms.AsDiscrete(to_onehot=args["output_classes"])])

    model_without_ddp = model
    if args["distributed"]:
        model = DistributedDataParallel(model, device_ids=[args["gpu"]], find_unused_parameters=True)
        model_without_ddp = model.module

    # optimizers
    optimizer = torch.optim.SGD(
        model_without_ddp.weight_parameters(), lr=args["learning_rate"] * world_size, momentum=0.9, weight_decay=0.00004
    )
    arch_optimizer_a = torch.optim.Adam(
        [dints_space.log_alpha_a], lr=args["learning_rate_arch"] * world_size, betas=(0.5, 0.999), weight_decay=0.0
    )
    arch_optimizer_c = torch.optim.Adam(
        [dints_space.log_alpha_c], lr=args["learning_rate_arch"] * world_size, betas=(0.5, 0.999), weight_decay=0.0
    )

    # amp
    if args["amp"]:
        from torch.cuda.amp import GradScaler, autocast
        model_scaler = GradScaler()
        space_scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("[info] amp enabled")

    # start a typical PyTorch training
    val_interval = args["num_epochs_per_validation"]
    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args["arch_ckpt_path"], "Events"))

        with open(os.path.join(args["arch_ckpt_path"], "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    # Init metric tracker
    metrics = {'train': MetricsTracker(), 'val': MetricsTracker()}

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
        dints_space.load_state_dict(checkpoint["dints"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        arch_optimizer_a.load_state_dict(checkpoint["arch_optimizer_a"])
        arch_optimizer_c.load_state_dict(checkpoint["arch_optimizer_c"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        model_scaler.load_state_dict(checkpoint["model_scaler"])
        space_scaler.load_state_dict(checkpoint["space_scaler"])
        metrics = checkpoint["metrics"]

    for epoch in range(args["start_epoch"], args["num_epochs"]):

        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = TimestampedTimer("Start training")
            model, dints_space, writer, metrics = search_one_epoch(...)
            timer.report(f'training generator for epoch {epoch}')

            if (epoch + 1) % val_interval == 0 or (epoch + 1) == args["num_epochs"]:

                with val_sampler.in_epoch(epoch):
                    timer = TimestampedTimer("Start evaluation")

                    timer, metrics = eval_search(...)
                    timer.report(f'evaluating generator for epoch {epoch}')

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer.close()

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
