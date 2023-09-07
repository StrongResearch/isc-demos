import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from cycling_utils import InterruptableDistributedSampler, atomic_torch_save

def train_one_epoch(model, optimizer, data_loader, train_batch_sampler, lr_scheduler, warmup_lr_scheduler, args, device, epoch, print_freq, scaler=None, timer=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    timer.report('training preliminaries')

    # Running this before starting the training loop assists reporting on progress after resuming - step == batch count
    step = train_batch_sampler.sampler.progress // args.batch_size

    for images, targets in metric_logger.log_every(data_loader, train_batch_sampler.sampler.progress // args.batch_size, print_freq, header): ## EDITED THIS - ARGS.BATCH_SIZE == DATALOADER.BATCH_SIZE? GROUPEDBATCHSAMPLER AT PLAY

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        timer.report(f'Epoch: {epoch} Step {step}: moving batch data to device')

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        timer.report(f'Epoch: {epoch} Step {step}: forward pass')

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        timer.report(f'Epoch: {epoch} Step {step}: computing loss')

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        timer.report(f'Epoch: {epoch} Step {step}: backward pass')

        ## Always update warmup_lr_scheduler - once progressed past epoch 0, this will make no difference.
        warmup_lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        timer.report(f'Epoch: {epoch} Step {step}: updating metric logger')

        # ADDED THE FOLLOWING - INC NECESSARY ARGS TO TRAIN
        train_batch_sampler.sampler.advance(len(images))
        step = train_batch_sampler.sampler.progress // args.batch_size

        timer.report(f'Epoch: {epoch} Step {step}: advancing sampler and computing step')

        if utils.is_main_process() and step % 5 == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "warmup_lr_scheduler": warmup_lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "sampler": train_batch_sampler.sampler.state_dict(),
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            timer = atomic_torch_save(checkpoint, args.resume, timer)

        # # Simulating end of epoch
        # if step >= 10:
        #     print("Simulating end of epoch")
        #     return metric_logger, timer
        
    #     # END ADD

    return metric_logger, timer


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, timer):

    timer.report('starting evaluation routine')

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    timer.report(f'preliminaries')

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    timer.report(f'preparing coco evaluator')

    eval_batch = 1
    for images, targets in metric_logger.log_every(data_loader, 100, header):

        images = list(img.to(device) for img in images)

        timer.report(f'eval batch: {eval_batch} moving to device')

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)

        timer.report(f'eval batch: {eval_batch} forward through model')

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        timer.report(f'eval batch: {eval_batch} outputs back to cpu')

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        timer.report(f'eval batch: {eval_batch} update evaluator')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    timer.report(f'evaluator accumulation and summarization')

    return coco_evaluator, timer
