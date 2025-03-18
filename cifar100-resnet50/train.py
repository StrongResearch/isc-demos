import os, sys, time, warnings, pickle, random
from pathlib import Path
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from torchvision import models, datasets, transforms
from cycling_utils import InterruptableDistributedSampler, AtomicDirectory, atomic_torch_save

warnings.filterwarnings("ignore")

def topk_accuracy(preds, targs, topk=1, normalize=True):
    topk_preds = preds.argsort(axis=1, descending=True)[:,:topk]
    topk_accurate = np.array([[t in p] for t,p in zip(targs,topk_preds)])
    if normalize:
        return topk_accurate.sum() / len(targs)
    else:
        return topk_accurate.sum()

def model_builder(model_parameters):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, model_parameters['output_size'])
    return model

def train_eval_ddp(device_id, rank, world_size, model_parameters, nepochs, batch_size, accumulate, train_dataset, valid_dataset, evaluate, learning_rate):
    # Config cuda rank and model
    torch.cuda.empty_cache()
    torch.cuda.set_device(device_id)
    model = model_builder(model_parameters)
    model.to(device_id)
    model.cuda()
    ddp_model = DDP(model, device_ids=[device_id])

    SEED = 308184653
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Use gradient compression to reduce communication
    ddp_model.register_comm_hook(None, default.fp16_compress_hook)

    loss_function = nn.CrossEntropyLoss(reduction='sum').to(device_id)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)


    # Init train and validation samplers and loaders
    train_sampler = InterruptableDistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=6)

    if evaluate:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False, num_workers=6)

    completed_epochs = 0

    # init checkpoint saver
    output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    saver = AtomicDirectory(output_directory=output_directory, is_master=rank==0)

    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        checkpoint_path = os.path.join(latest_checkpoint_path, "checkpoint.pt")
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        ddp_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        train_sampler.load_state_dict(checkpoint["sampler_state_dict"])
        completed_epochs = checkpoint["epoch"]

    # Training
    results = {}

    print(f"Training from epoch {completed_epochs+1} to epoch {nepochs}")
    for epoch in range(train_loader.sampler.epoch, nepochs):

        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)

        cumulative_train_loss = 0.0
        train_examples_seen = 0.0
        n_train_batches = len(train_loader)
        ddp_model.train(True)
        optimizer.zero_grad()
        start = time.perf_counter()

        for X_train, y_train in train_loader:
            i = train_sampler.progress // train_loader.batch_size

            if rank == 0 and i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{n_train_batches}")
            X_train, y_train = X_train.to(device_id), y_train.to(device_id)

            train_sampler.advance(len(X_train))

            if (i + 1) % accumulate == 0 or (i + 1) == n_train_batches: # Final loop in accumulation cycle, or last batch in dataset
                z_train = ddp_model(X_train)
                loss = loss_function(z_train, y_train)
                cumulative_train_loss += loss.item()
                train_examples_seen += len(y_train)
                loss.backward() # Sync gradients between devices
                optimizer.step() # Weight update
                optimizer.zero_grad() # Zero grad

                if i % 50 == 0:
                    checkpoint_directory = saver.prepare_checkpoint_directory()

                    if rank == 0:
                        print("saving checkpoint", train_sampler.state_dict())
                        atomic_torch_save(
                            {
                                "epoch": epoch,
                                "model_state_dict": ddp_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "sampler_state_dict": train_sampler.state_dict(),
                            }, os.path.join(checkpoint_directory, "checkpoint.pt")
                        )

                    saver.symlink_latest(checkpoint_directory)

            else: # Otherwise only accumulate gradients locally to save time.
                with ddp_model.no_sync():
                    z_train = ddp_model(X_train)
                    loss = loss_function(z_train, y_train)
                    cumulative_train_loss += loss.item()
                    train_examples_seen += len(y_train)
                    loss.backward()

        # Average training loss per batch and training time
        tloss = cumulative_train_loss / (train_examples_seen if train_examples_seen > 0 else 1)
        epoch_duration = time.perf_counter() - start

        # Evaluation
        if evaluate:
            cumulative_valid_loss = 0.0
            valid_examples_seen = 0.0
            top1acc = 0.0
            top5acc = 0.0
            
            ddp_model.eval()
            with torch.no_grad():
                for X_valid, y_valid in valid_loader:
                    X_valid, y_valid = X_valid.to(device_id), y_valid.to(device_id)
                    z_valid = model(X_valid)

                    loss_valid = loss_function(z_valid, y_valid)
                    valid_top1acc = topk_accuracy(z_valid, y_valid, topk=1, normalize=False)
                    valid_top5acc = topk_accuracy(z_valid, y_valid, topk=5, normalize=False)

                    cumulative_valid_loss += loss_valid.item()
                    top1acc += valid_top1acc.item()
                    top5acc += valid_top5acc.item()
                    valid_examples_seen += len(y_valid)

            vloss = cumulative_valid_loss / valid_examples_seen
            top1 = (top1acc / valid_examples_seen) * 100
            top5 = (top5acc / valid_examples_seen) * 100
        
        else:
            vloss = 0
            top1 = 0
            top5 = 0

        # Gather performance from all devices and average for reporting
        transmit_data = np.array([tloss, vloss, top1, top5, epoch_duration])
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, transmit_data)
        result = np.stack(outputs).mean(axis=0)
        tloss_, vloss_, top1_, top5_, epoch_duration_ = result

        results[epoch] = {
            "tloss": tloss_,
            "vloss": vloss_,
            "top1": top1_,
            "top5": top5_,
            "time": epoch_duration_
        }

        # Learning rate scheduler reducing by factor of 10 when training loss stops reducing. Likely to overfit first.
        # Must apply same operation for all devices to ensure optimizers remain in sync.
        scheduler.step(tloss_)
        
        # If main rank, save results and report.
        if rank == 0:
            print(f'EPOCH {epoch}, TLOSS {tloss_:.3f}, VLOSS {vloss_:.3f}, TOP1 {top1_:.2f}, TOP5 {top5_:.2f}, TIME {epoch_duration_:.3f}')

        checkpoint_directory = saver.prepare_checkpoint_directory()

        if rank == 0:
            print("saving checkpoint")
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": ddp_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "sampler_state_dict": train_sampler.state_dict(),
                }, os.path.join(checkpoint_directory, "checkpoint.pt")
            )

        saver.symlink_latest(checkpoint_directory)

        train_loader.sampler.reset_progress()
        valid_loader.sampler.reset_progress()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    print(args)

    # setup distributed training
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    # Set up train and validation datasets
    norm_stats = ((0.5071, 0.4866, 0.4409),(0.2009, 0.1984, 0.2023)) # CIFAR100 training set normalization constants
    R = 384
    train_transform = transforms.Compose([
        transforms.AutoAugment(policy = transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(R),
        transforms.ToTensor(), # Also standardizes to range [0,1]
        transforms.Normalize(*norm_stats),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(R),
        transforms.ToTensor(), # Also standardizes to range [0,1]
        transforms.Normalize(*norm_stats),
    ])

    data_path = os.path.join("/data", args.dataset_id)
    train_dataset = datasets.CIFAR100(root=data_path, train=True, transform=train_transform, download=True)

    # Hold-out this data for final evaluation
    valid_dataset = datasets.CIFAR100(root=data_path, train=False, transform=valid_transform, download=True)

    print(f'Train: {len(train_dataset):,.0f}, Valid: {len(valid_dataset):,.0f}')

    train_eval_ddp(device_id, rank, world_size, model_parameters={"output_size": 100}, nepochs=args.epochs, batch_size=args.batch_size, accumulate=1, train_dataset=train_dataset, valid_dataset=valid_dataset, evaluate=True, learning_rate=args.lr)

