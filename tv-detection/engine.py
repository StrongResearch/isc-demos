import math
import sys
from itertools import product

import torch
import torchvision.models.detection.mask_rcnn
import torch.distributed as dist 
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from cycling_utils import atomic_torch_save

from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(
        model, optimizer, data_loader_train, train_sampler, test_sampler,
        lr_scheduler, warmup_lr_scheduler, args, device, coco_evaluator,
        epoch, scaler=None, timer=None, metrics=None,
    ):

    model.train()

    timer.report('training preliminaries')

    print(f'\nTraining / resuming epoch {epoch} from training step {train_sampler.progress}\n')

    for images, targets in data_loader_train:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: moving batch data to device')
        # print(f"First 2 image shapes: {images[0].shape}, {images[1].shape}")

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            # CHECK IF NUMERIC ERROR HAS OCCURRED AND IF SO, SKIP THIS BATCH
            check_0 = 1 if torch.tensor([torch.isnan(v) for v in loss_dict.values()]).any() else 0
            check_1 = 1 if not all([math.isfinite(v) for v in loss_dict.values()]) else 0
            check_tensor = torch.tensor([check_0, check_1], requires_grad=False, device=device)
            dist.all_reduce(check_tensor, op=dist.ReduceOp.SUM)
            if check_tensor.sum() > 0:
                print(f"CONTINUE CONDITION: {[e for e in check_tensor]}")
                train_sampler.advance() # Advance sampler to try next batch
                continue

            losses = sum(loss for loss in loss_dict.values())
            timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: forward pass')

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        warmup_lr_scheduler.step()

        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: backward pass')

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: computing loss')

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metrics["train"].update({"images_seen": len(images) ,"loss": loss_value})
        metrics["train"].update({k:v.item() for k,v in loss_dict_reduced.items()})
        metrics["train"].reduce() # Gather results from all nodes
        
        report_metrics = [m for m in metrics["train"].local if m != "images_seen"]
        images_seen = metrics["train"].local["images_seen"]
        vals = [metrics["train"].local[m]/images_seen for m in report_metrics]
        rpt = ", ".join([f"{m}: {v:,.3f}" for m,v in zip(report_metrics, vals)])
        print(f"EPOCH: [{epoch}], BATCH: [{train_sampler.progress}/{len(train_sampler)}], "+rpt)

        metrics["train"].reset_local()

        print(f"Saving checkpoint at epoch {epoch} train batch {train_sampler.progress}")
        train_sampler.advance()

        if train_sampler.progress == len(train_sampler):
            metrics["train"].end_epoch()

        if utils.is_main_process() and train_sampler.progress % 1 == 0: # Checkpointing every batch

            writer = SummaryWriter(log_dir=args.tboard_path)
            for metric,val in zip(report_metrics, vals):
                writer.add_scalar("Train/"+metric, val, train_sampler.progress + epoch * len(train_sampler))
            writer.flush()
            writer.close()
                
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
                "img_ids": coco_evaluator.img_ids, # catalogue of images seen already
                "eval_imgs": coco_evaluator.eval_imgs, # image evaluations
                "metrics": metrics,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    lr_scheduler.step() # OUTER LR_SCHEDULER STEP EACH EPOCH
    return model, timer, metrics

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
    device, scaler=None, timer=None, metrics=None,
):

    timer.report('starting evaluation routine')

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    timer.report('evaluation preliminaries')

    test_step = test_sampler.progress // data_loader_test.batch_size
    print(f'\nEvaluating / resuming epoch {epoch} from eval step {test_step}\n')

    for images, targets in data_loader_test:

        images = list(img.to(device) for img in images)
        timer.report(f'Epoch {epoch} batch: {test_step} moving to device')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs = model(images)
        timer.report(f'Epoch {epoch} batch: {test_step} forward through model')

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        timer.report(f'Epoch {epoch} batch: {test_step} outputs back to cpu')

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        # res = {img_id: {'boxes': T, 'labels': T, 'scores': T, 'masks': T}, ...}
        coco_evaluator.update(res)
        timer.report(f'Epoch {epoch} batch: {test_step} update evaluator')

        print(f"Saving checkpoint at epoch {epoch} eval batch {test_step}")
        test_sampler.advance(len(images))
        test_step = test_sampler.progress // data_loader_test.batch_size

        if utils.is_main_process() and test_step % 1 == 0: # Checkpointing every batch
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
                "img_ids": coco_evaluator.img_ids, # catalogue of images seen already
                "eval_imgs": coco_evaluator.eval_imgs, # image evaluations
                "metrics": metrics,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    results = coco_evaluator.summarize()

    metric_A = ["bbox-", "segm-"]
    metric_B = ["AP", "AR"]
    metric_C = ["", "50", "75", "-S", "-M", "-L"]
    metric_names = ["".join(t) for t in product(metric_A, metric_B, metric_C)]
    metrics["val"].update({name: val for name,val in zip(metric_names, results)})
    metrics["val"].reduce()
    metrics["val"].end_epoch()

    if utils.is_main_process():
        writer = SummaryWriter(log_dir=args.tboard_path)
        for name,val in metrics["val"].epoch_reports[-1].items():
            writer.add_scalar("Val/"+name, val, epoch)
        writer.flush()
        writer.close()

    torch.set_num_threads(n_threads)

    # Reset the coco evaluator at the end of the epoch
    coco = get_coco_api_from_dataset(data_loader_test.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    timer.report('evaluator accumulation, summarization, and reset')

    return coco_evaluator, timer, metrics
