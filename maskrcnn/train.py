from cycling_utils import TimestampedTimer
timer = TimestampedTimer()

import os
from pathlib import Path
import argparse
import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection import MaskRCNN, RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, InterruptableDistributedGroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste

from cycling_utils import InterruptableDistributedSampler

timer.report('importing everything else')

def print_rank0(message):
    if int(os.environ["RANK"]) == 0:
        print(message)

def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))

def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes

def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)
    
def _get_iou_types(model): # intersection over union (iou) types
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def get_optim(model, args):
    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd-nesterov"):
        optimizer = torch.optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
        print(f"Optimizer: SGD-Nesterov, lr: {args.lr}, momentum: {args.momentum}, weight_decay: {args.weight_decay}")
    elif opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(parameters, lr=args.lr)
        print(f"Optimizer: SGD, lr: {args.lr}")
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        print(f"Optimizer: AdamW, lr: {args.lr}, weight_decay: {args.weight_decay}")
    return optimizer

timer.report('defined other functions')

def main(args, timer):

    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp"):
        raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    timer.report('main preliminaries')

    dataset_train, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)
    timer.report('loading data')

    group_ids = create_aspect_ratio_groups(dataset_train, k=args.aspect_ratio_group_factor)
    train_sampler = InterruptableDistributedGroupedBatchSampler(dataset_train, group_ids, args.batch_size)
    test_sampler = InterruptableDistributedSampler(dataset_test)
    timer.report('data samplers')

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")
        print_rank0("Using copypaste_collate_fn for train_collate_fn")
        train_collate_fn = copypaste_collate_fn

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=train_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    timer.report('data loaders')

    if args.model.endswith("resnet50_fpn"):
        backbone = resnet_fpn_backbone(backbone_name="resnet50", weights="ResNet50_Weights.IMAGENET1K_V1", trainable_layers=args.trainable_backbone_layers)
    elif args.model.endswith("resnet101_fpn"):
        backbone = resnet_fpn_backbone(backbone_name='resnet101', weights='ResNet101_Weights.IMAGENET1K_V1', trainable_layers=args.trainable_backbone_layers)
    else:
        raise Exception(f"Backbone must be resnet50_fpn or resnet101_fpn, received {args.model}")
    total_backbone_params = sum(p.numel() for p in backbone.parameters())/1e6
    trainable_backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)/1e6
    timer.report(f'backbone learnable params {trainable_backbone_params:,.2f}M / {total_backbone_params:,.2f}M ({100*trainable_backbone_params/total_backbone_params:.1f}%)')

    if args.model.startswith("maskrcnn"):
        model = MaskRCNN(backbone=backbone, num_classes=num_classes)
    elif args.model.startswith("retinanet"):
        model = RetinaNet(backbone=backbone, num_classes=num_classes)
    else:
        raise Exception(f"Model must be maskrcnn or retinanet, received {args.model}")
    total_model_params = sum(p.numel() for p in model.parameters())/1e6
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    timer.report(f'model learnable params {trainable_model_params:,.2f}M / {total_model_params:,.2f}M ({100*trainable_model_params/total_model_params:.1f}%)')
        
    model.to(device)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    timer.report('preparing model for distributed training')
    
    optimizer = get_optim(model, args)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    steps_per_epoch = len(dataloader_train)
    lr_step_size = steps_per_epoch*args.lr_step_size

    ## WARMUP LR_SCHEDULER
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=steps_per_epoch*args.warmup_epochs
    )

    ## ONGOING LR_SCHEDULER
    long_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=args.lr_gamma)

    ## MASTER LR_SCHEDULER
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, long_lr_scheduler], milestones=[steps_per_epoch*args.warmup_epochs]
    )
    timer.report('optimizer, scaler, learning rate schedulers')

    coco = get_coco_api_from_dataset(dataloader_test.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    timer.report('init coco evaluator')

    # RETRIEVE CHECKPOINT
    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.resume and os.path.isfile(args.resume): # If we're resuming...
        print_rank0("RESUMING FROM CURRENT JOB")
        checkpoint = torch.load(args.resume, map_location="cpu")
    elif args.prev_resume and os.path.isfile(args.prev_resume):
        print_rank0(f"RESUMING FROM PREVIOUS JOB {args.prev_resume}")
        checkpoint = torch.load(args.prev_resume, map_location="cpu")
    if checkpoint is not None:
        args.start_epoch = checkpoint["epoch"]
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        test_sampler.load_state_dict(checkpoint["test_sampler"])
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        # Evaluator
        coco_evaluator.img_ids = checkpoint["img_ids"]
        coco_evaluator.eval_imgs = checkpoint["eval_imgs"]

    timer.report('retrieving checkpoint')

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        epoch = 0
        coco_evaluator, timer = evaluate(
            model, dataloader_test, epoch, test_sampler, args, coco_evaluator, optimizer, 
            lr_scheduler, train_sampler, device, scaler, timer
        )
        return

    for epoch in range(args.start_epoch, args.epochs):

        print_rank0(f"\nEPOCH :: {epoch}\n")

        with train_sampler.in_epoch(epoch):
            model, timer = train_one_epoch(
                model, optimizer, dataloader_train, dataloader_test,
                lr_scheduler, args, device, coco_evaluator, 
                epoch, scaler, timer
            )

            with test_sampler.in_epoch(epoch):
                coco_evaluator, timer = evaluate(
                    model, optimizer, dataloader_train, dataloader_test,
                    lr_scheduler, args, device, coco_evaluator, 
                    epoch, scaler, timer
                )
                
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("--data-path", default=None, type=str, help="dataset path")
    parser.add_argument("--dataset",default="coco",type=str, help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",)
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--warmup-epochs", default=1, type=int, metavar="N", help="We're going to try training on maskrcnn loss for N epochs first, then focal loss.")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.02, type=float,help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay",)
    parser.add_argument("--norm-weight-decay",default=None,type=float,help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--lr-step-size", default=4, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.9, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)")
    parser.add_argument("--grad-clip-norm", default=0.1, type=float, help="gradient clipping norm (using 'inf' norm)")
    parser.add_argument("--accumulation-steps", default=1, type=int, help="training gradient accumulation steps")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--prev-resume", default=None, help="path of previous job checkpoint for strong fail resume", dest="prev_resume") # for checkpointing
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--tboard-path", default=None, help="path for saving tensorboard logs", dest="tboard_path") # for checkpointing
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--trainable-backbone-layers", default=3, type=int, help="number of trainable layers of backbone")
    parser.add_argument("--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)")
    parser.add_argument("--sync-bn",dest="sync_bn",help="Use sync batch norm",action="store_true")
    parser.add_argument("--test-only",dest="test_only",help="Only test the model",action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    # Use CopyPaste augmentation training parameter
    parser.add_argument("--use-copypaste", action="store_true",help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",)
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
