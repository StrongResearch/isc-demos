from cycling_utils import Timer

timer = Timer()
timer.report('importing Timer')

import datetime
import os
import time
import warnings

from pathlib import Path
import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.transforms import functional as F, InterpolationMode
from cycling_utils import InterruptableDistributedSampler, atomic_torch_save, Timer

timer.report('importing everything else')

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)
    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train, args):
    if train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)
        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=520)

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
    if len(losses) == 1:
        return losses["out"]
    return losses["out"] + 0.5 * losses["aux"]

def evaluate(
        model, data_loader_test, num_classes, confmat,
        optimizer, lr_scheduler,
        train_sampler: InterruptableDistributedSampler,
        test_sampler: InterruptableDistributedSampler, 
        device, epoch, print_freq, scaler=None, timer=None
    ):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    timer.report(f'evaluation preliminaries')

    test_step = test_sampler.progress // data_loader_test.batch_size
    print(f'\nEvaluating / resuming epoch {epoch} from eval step {test_step}\n')
    timer.report('launch evaluation routine')

    with torch.inference_mode():

        for image, target in metric_logger.log_every(data_loader_test, test_step, print_freq, header):

            image, target = image.to(device), target.to(device)
            timer.report(f'Epoch {epoch} batch: {test_step} moving to device')

            output = model(image)
            output = output["out"]
            timer.report(f'Epoch {epoch} batch: {test_step} forward through model')

            confmat_temp = utils.ConfusionMatrix(num_classes)
            confmat_temp.update(target.flatten().detach().cpu(), output.argmax(1).flatten().detach().cpu())
            confmat_temp.reduce_from_all_processes()
            confmat.mat += confmat_temp.mat.cpu() # Add to the running eval matrix

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]
            timer.report(f'Epoch {epoch} batch: {test_step} confmat update')

            test_sampler.advance(len(image))

            test_step = test_sampler.progress // data_loader_test.batch_size
            if utils.is_main_process() and test_step % 1 == 0: # Checkpointing every batch

                print(f"Saving checkpoint at epoch {epoch} eval batch {test_step}")
                checkpoint = {
                    "args": args,
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_sampler": train_sampler.state_dict(),
                    "test_sampler": test_sampler.state_dict(),
                    "confmat": confmat.mat, # For storing eval metric
                }
                if args.amp:
                    checkpoint["scaler"] = scaler.state_dict()
                timer = atomic_torch_save(checkpoint, args.resume, timer)

    print(confmat)
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)

    if (
        hasattr(data_loader_test.dataset, "__len__")
        and len(data_loader_test.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader_test.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    # # Save final checkpoint once all eval ops have completed - MIGHT BE ABLE TO OMIT THIS ALTOGETHER
    # if utils.is_main_process():
    #     print(f"Saving checkpoint at end of epoch {epoch} evaluation")
    #     checkpoint = {
    #         "mode": "train",
    #         "model": model.module.state_dict(),
    #         "epoch": epoch,
    #         "args": args,
    #         "test_sampler": test_sampler.state_dict(),
    #         "conf_mat": confmat, # For storing eval metric
    #     }
    #     timer = atomic_torch_save(checkpoint, args.resume, timer)

    return confmat, timer


def train_one_epoch(
        model, criterion, optimizer, data_loader_train,
        train_sampler: InterruptableDistributedSampler, 
        test_sampler: InterruptableDistributedSampler, confmat,
        lr_scheduler, device, epoch, print_freq, scaler=None, timer=None
    ):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    timer.report('training preliminaries')

    # Running this before starting the training loop assists reporting on progress after resuming - train_step == batch count
    # Also means when resuming during evaluation, the training phase is skipped as train_sampler progress == 100%.
    train_step = train_sampler.progress // data_loader_train.batch_size
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for image, target in metric_logger.log_every(data_loader_train, train_step, print_freq, header):

        image, target = image.to(device), target.to(device)
        timer.report(f'Epoch: {epoch} batch {train_step}: moving batch data to device')

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
        timer.report(f'Epoch: {epoch} batch {train_step}: forward pass')

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        timer.report(f'Epoch: {epoch} batch {train_step}: backward pass')

        lr_scheduler.step()
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        train_sampler.advance(len(image))
        timer.report(f'Epoch: {epoch} batch {train_step}: updating metric logger')

        train_step = train_sampler.progress // data_loader_train.batch_size
        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch
            print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "test_sampler": test_sampler.state_dict(),
                "confmat": confmat, # For storing eval metric
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    # # Saving checkpoint at the end of training epoch - MIGHT BE ABLE TO OMIT THIS ALTOGETHER
    # if utils.is_main_process(): # Checkpointing every batch?
    #     print(f"Saving checkpoint at end of epoch {epoch} training")
    #     checkpoint = {
    #         "mode": "evaluate",
    #         "model": model.module.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "lr_scheduler": lr_scheduler.state_dict(),
    #         "epoch": epoch,
    #         "args": args,
    #         "train_sampler": train_sampler.state_dict(),
    #     }
    #     if args.amp:
    #         checkpoint["scaler"] = scaler.state_dict()
    #     timer = atomic_torch_save(checkpoint, args.resume, timer)

    return metric_logger, timer

timer.report('defined other functions')

def main(args, timer):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    assert args.distributed # don't support cycling when not distributed for simplicity

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    timer.report('main preliminaries')

    dataset_train, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(True, args))
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args))

    ## SUBSET FOR TESTING EPOCH ROLLOVER
    dataset_train = torch.utils.data.Subset(dataset_train, torch.arange(450))
    dataset_test = torch.utils.data.Subset(dataset_test, torch.arange(108))

    timer.report('loading data')

    # if args.distributed:
    train_sampler = InterruptableDistributedSampler(dataset_train)
    test_sampler = InterruptableDistributedSampler(dataset_test)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(dataset)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    timer.report('creating data samplers')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, collate_fn=utils.collate_fn, drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    timer.report('creating data loaders')

    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, aux_loss=args.aux_loss,
    )
    model.to(device)

    timer.report('creating model and .to(device)')

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    timer.report('preparing model for distributed training')

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    timer.report('optimizer and scaler')

    iters_per_epoch = len(data_loader_train)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    timer.report('learning rate schedulers')

    # Init global confmat for eval - eval accumulator
    confmat = utils.ConfusionMatrix(num_classes)
    timer.report('init confmat')

    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    if args.resume and os.path.isfile(args.resume): # If we're resuming...

        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        args.start_epoch = checkpoint["epoch"]
 
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        if args.amp: # Could align this syntactically...
            scaler.load_state_dict(checkpoint["scaler"])

        test_sampler.load_state_dict(checkpoint["test_sampler"])
        confmat.mat = checkpoint["confmat"]
            
    timer.report('retrieving checkpoint')

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat, timer = evaluate(model, data_loader_test, num_classes, confmat, test_sampler, device, 0, args.print_freq, timer)
        print(confmat)
        return

    for epoch in range(args.start_epoch, args.epochs):

        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = Timer() # Restarting timer, timed the preliminaries, now obtain time trial for each epoch
            metric_logger, timer = train_one_epoch(model, criterion, optimizer, data_loader_train, train_sampler, test_sampler, confmat, lr_scheduler, device, epoch, args.print_freq, scaler, timer)
            timer.report(f'training for epoch {epoch}')

        with test_sampler.in_epoch(epoch):
            timer = Timer() # Restarting timer, timed the preliminaries, now obtain time trial for each epoch
            confmat, timer = evaluate(model, data_loader_test, num_classes, confmat, optimizer, lr_scheduler, train_sampler, test_sampler, device, epoch, args.print_freq, scaler, timer)
            timer.report(f'evaluation for epoch {epoch}')


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size", dest="batch_size")
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=1, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", type=str, help="path of checkpoint", required=True)
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=9, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
