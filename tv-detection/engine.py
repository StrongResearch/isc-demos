import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from cycling_utils import InterruptableDistributedSampler, atomic_torch_save

def train_one_epoch(
        model, optimizer, data_loader_train, train_sampler, test_sampler,
        lr_scheduler, warmup_lr_scheduler, args, device,
        epoch, scaler=None, timer=None
    ):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    timer.report('training preliminaries')

    # Running this before starting the training loop assists reporting on progress after resuming - step == batch count
    # train_step = train_sampler.progress // args.batch_size
    print(f'\nTraining / resuming epoch {epoch} from training step {train_sampler.progress}\n')

    for images, targets in metric_logger.log_every(data_loader_train, train_sampler.progress, args.print_freq, header): ## EDITED THIS - ARGS.BATCH_SIZE == DATALOADER.BATCH_SIZE? GROUPEDBATCHSAMPLER AT PLAY

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: moving batch data to device')

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: forward pass')

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: computing loss')

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
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: backward pass')

        ## Always update warmup_lr_scheduler - once progressed past epoch 0, this will make no difference.
        warmup_lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: updating metric logger')

        # train_step = train_sampler.progress
        train_sampler.advance() # counted in batches, no args to pass
        if utils.is_main_process() and train_sampler.progress % 1 == 0: # Checkpointing every batch
            print(f"Saving checkpoint at epoch {epoch} train batch {train_sampler.progress}")
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "warmup_lr_scheduler": warmup_lr_scheduler.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "test_sampler": test_sampler.state_dict(),
                
                # # Evaluator state variables
                # "coco_gt": coco_evaluator.coco_gt,
                # "iou_types": coco_evaluator.iou_types,
                # "coco_eval": coco_evaluator.coco_eval,
                # "img_ids": coco_evaluator.img_ids,
                # "eval_imgs": coco_evaluator.eval_imgs,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    lr_scheduler.step() # OUTER LR_SCHEDULER STEP EACH EPOCH
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
def evaluate(
    model, data_loader_test, epoch, test_sampler, args, coco_evaluator,
    optimizer, lr_scheduler, warmup_lr_scheduler, train_sampler,
    device, scaler=None, timer=None
):

    timer.report('starting evaluation routine')

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    timer.report(f'evaluation preliminaries')

    # coco = get_coco_api_from_dataset(data_loader_test.dataset)
    # iou_types = _get_iou_types(model)
    # coco_evaluator = CocoEvaluator(coco, iou_types)

    test_step = test_sampler.progress // data_loader_test.batch_size
    print(f'\nEvaluating / resuming epoch {epoch} from eval step {test_step}\n')
    timer.report('launch evaluation routine')

    for images, targets in metric_logger.log_every(data_loader_test, test_sampler.progress, args.print_freq, header):

        images = list(img.to(device) for img in images)
        timer.report(f'Epoch {epoch} batch: {test_step} moving to device')

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        timer.report(f'Epoch {epoch} batch: {test_step} forward through model')

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        timer.report(f'Epoch {epoch} batch: {test_step} outputs back to cpu')

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        timer.report(f'Epoch {epoch} batch: {test_step} update evaluator')

        test_sampler.advance(len(images))

        test_step = test_sampler.progress // data_loader_test.batch_size
        if utils.is_main_process() and test_step % 1 == 0: # Checkpointing every batch
            print(f"Saving checkpoint at epoch {epoch} eval batch {test_step}")
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "warmup_lr_scheduler": warmup_lr_scheduler.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "test_sampler": test_sampler.state_dict(),

                # Evaluator state variables
                "coco_gt": coco_evaluator.coco_gt,
                "iou_types": coco_evaluator.iou_types,
                "coco_eval": coco_evaluator.coco_eval,
                "img_ids": coco_evaluator.img_ids,
                "eval_imgs": coco_evaluator.eval_imgs,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            timer = atomic_torch_save(checkpoint, args.resume, timer)

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
