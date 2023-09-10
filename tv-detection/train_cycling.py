r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
from cycling_utils import Timer

timer = Timer()
timer.report('importing Timer')

import datetime
import os
import time
# import warnings

from pathlib import Path
import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco

import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste

from cycling_utils import InterruptableDistributedSampler, InterruptableDistributedGroupedBatchSampler, atomic_torch_save

timer.report('importing everything else')

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

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    timer.report('main preliminaries')

    # Data loading code
    dataset_train, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    # ## SUBSET FOR TESTING
    # dataset_train = torch.utils.data.Subset(dataset_train, torch.arange(500))
    # dataset_test = torch.utils.data.Subset(dataset_test, torch.arange(200))

    timer.report('loading data')

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(dataset)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # if args.aspect_ratio_group_factor >= 0: # default == 3
    #     group_ids = create_aspect_ratio_groups(dataset_train, k=args.aspect_ratio_group_factor)
    #     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    # else:
    #     train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    group_ids = create_aspect_ratio_groups(dataset_train, k=args.aspect_ratio_group_factor)
    train_sampler = InterruptableDistributedGroupedBatchSampler(dataset_train, group_ids, args.batch_size)
    test_sampler = InterruptableDistributedSampler(dataset_test)

    timer.report('creating data samplers')

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")
        print("Using copypaste_collate_fn for train_collate_fn")
        train_collate_fn = copypaste_collate_fn

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=train_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    timer.report('creating data loaders')

    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    )
    model.to(device)

    timer.report('creating model and .to(device)')

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    timer.report('preparing model for distributed training')

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    timer.report('optimizer and scaler')

    ## OUTER LR_SCHEDULER
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    
    ## WARMUP LR_SCHEDULER
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(data_loader_train) - 1)
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    timer.report('learning rate schedulers')

    from coco_eval import CocoEvaluator
    from coco_utils import get_coco_api_from_dataset
    coco = get_coco_api_from_dataset(data_loader_test.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    timer.report('init coco evaluator')

    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    if args.resume and os.path.isfile(args.resume):

        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        args.start_epoch = checkpoint["epoch"]

        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        warmup_lr_scheduler.load_state_dict(checkpoint["warmup_lr_scheduler"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

        test_sampler.load_state_dict(checkpoint["test_sampler"])

        # Evaluator state variables
        coco_evaluator.img_ids = checkpoint["img_ids"]
        coco_evaluator.eval_imgs = checkpoint["eval_imgs"]

    timer.report('retrieving checkpoint')

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        coco_evaluator, timer = evaluate(model, data_loader_test, device, timer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        
        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = Timer() # Restarting timer, timed the preliminaries, now obtain time trial for each epoch
            metric_logger, timer = train_one_epoch(model, optimizer, data_loader_train, train_sampler, test_sampler, lr_scheduler, warmup_lr_scheduler, args, device, coco_evaluator, epoch, scaler, timer)

            # NEST THE TEST SAMPLER IN TRAIN SAMPLER CONTEXT TO AVOID EPOCH RESTART?
            with test_sampler.in_epoch(epoch):
                timer = Timer() # Restarting timer, timed the preliminaries, now obtain time trial for each epoch
                coco_evaluator, timer = evaluate(model, data_loader_test, epoch, test_sampler, args, coco_evaluator, optimizer, lr_scheduler, warmup_lr_scheduler, train_sampler, device, scaler, timer)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset",default="coco",type=str,help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",)
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr",default=0.02,type=float,help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay",)
    parser.add_argument("--norm-weight-decay",default=None,type=float,help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)")
    parser.add_argument("--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr-steps",default=[16, 22],nargs="+",type=int,help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)")
    parser.add_argument("--print-freq", default=1, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument("--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone")
    parser.add_argument("--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)")
    parser.add_argument("--sync-bn",dest="sync_bn",help="Use sync batch norm",action="store_true")
    parser.add_argument("--test-only",dest="test_only",help="Only test the model",action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument("--use-copypaste",action="store_true",help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",)

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
