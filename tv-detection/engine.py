import math
import sys
import socket
import os
import subprocess

# from itertools import product

import torch
import torchvision.models.detection.mask_rcnn
import torch.distributed as dist
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from cycling_utils import atomic_torch_save

from torch.utils.tensorboard import SummaryWriter

def print_rank0(message):
    if int(os.environ["RANK"]) == 0:
        print(message)

def get_grad_norm_stats(model):
    norms = []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        nm = p.grad.data.norm(2).item()
        norms.append(nm)
    norms = torch.tensor(norms)
    return [torch.mean(norms), torch.min(norms)] + [torch.quantile(norms, q) for q in [0.05, 0.25, 0.50, 0.75, 0.95]] + [torch.max(norms)]

def get_grad_stats(model):

    mins, maxs, means = [], [], []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        abs_grad_data = p.grad.data.abs()
        mn, mx, mean = abs_grad_data.min(), abs_grad_data.max(), abs_grad_data.mean()
        mins.append(mn)
        maxs.append(mx)
        means.append(mean)
    mins, maxs, means = torch.tensor(mins), torch.tensor(maxs), torch.tensor(means)

    min_stats = [torch.quantile(mins, q) for q in [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]]
    max_stats = [torch.quantile(maxs, q) for q in [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]]
    mean_stats = [torch.quantile(means, q) for q in [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]]

    return min_stats, max_stats, mean_stats

def get_mdl():
    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
    line = line_as_bytes.decode("ascii")
    return line

'''
def grad_clip(model, norm_type=2.0, scale=1.0):
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0, norm_type='inf')
'''

def train_one_epoch(
        model, optimizer, data_loader_train, train_sampler, test_sampler, 
        lr_scheduler, args, device, coco_evaluator, 
        epoch, scaler, timer, metrics
    ):

    model.train()
    timer.report('training preliminaries')
    if args.model.startswith("retinanet"):
        use_focal = epoch >= 0

    accumulation_steps = 1
    accumulated = 0

    cache_model_state = model.module.state_dict()
    
    print_rank0(f'\nTraining / resuming epoch {epoch} from training step {train_sampler.progress}\n')

    for images, targets in data_loader_train:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: moving batch data to device')

        with torch.cuda.amp.autocast(enabled=scaler is not None):

            if args.model.startswith("retinanet"):
                 loss_dict = model(images, targets, use_focal)
            else:
                loss_dict = model(images, targets)
            timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: forward pass')

            losses = sum(loss for loss in loss_dict.values())
            losses = losses / accumulation_steps
            timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: calculate loss')

            if losses > 1000 or not math.isfinite(losses):
              print(f"Losses value {losses} occurred on machine {socket.gethostname()} GPU {args.gpu}")
              gpu_report = get_mdl()
              print(gpu_report)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if loss_value > 1000 or not math.isfinite(loss_value):
            print_rank0(f"Loss is {loss_value}, WINDING BACK")
            print_rank0(loss_dict_reduced)
            # Load the model state from before this error occurred
            model.module.load_state_dict(cache_model_state)
            # Zero any accumulated gradients
            optimizer.zero_grad()
            # Reset accumulation counter
            accumulated = 0
            # sys.exit(1)
            continue

        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()
        accumulated += 1

        if accumulated == accumulation_steps or train_sampler.progress + 1 == len(data_loader_train):
            # Cache the model state right before the update...
            cache_model_state = model.module.state_dict()
            if scaler is not None:
                pre_min_stats, pre_max_stats, pre_mean_stats = get_grad_stats(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type='inf')
                scaler.step(optimizer)
                scaler.update()
            else:
                pre_min_stats, pre_max_stats, pre_mean_stats = get_grad_stats(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type='inf')
                optimizer.step()

            # trying gradient clipping to prevent gradient issues with retinanet...
            pst_min_stats, pst_max_stats, pst_mean_stats = get_grad_stats(model)
            print_rank0("PRE-CLIP MAX STATS: Q00: {:.5f}, Q05: {:.5f}, Q25: {:.5f}, Q50: {:.5f}, Q75: {:.5f}, Q95: {:.5f}, Q100: {:.5f}".format(*pre_max_stats))
            print_rank0("POST-CLIP MAX STATS: Q00: {:.5f}, Q05: {:.5f}, Q25: {:.5f}, Q50: {:.5f}, Q75: {:.5f}, Q95: {:.5f}, Q100: {:.5f}".format(*pst_max_stats))

            optimizer.zero_grad()
            accumulated = 0
            timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: graient clipping')

        lr_scheduler.step()
        timer.report(f'Epoch: {epoch} batch {train_sampler.progress}: backward pass')

        metrics["train"].update({"images_seen": len(images) ,"loss": loss_value})
        metrics["train"].update({k:v.item() for k,v in loss_dict_reduced.items()})
        metrics["train"].reduce() # Gather results from all nodes
        
        images_seen = metrics["train"].local["images_seen"]
        report_metrics = [m for m in metrics["train"].local if m != "images_seen"]
        vals = [metrics["train"].local[m]/images_seen for m in report_metrics]
        rpt = ", ".join([f"{m}: {v:,.3f}" for m,v in zip(report_metrics, vals)])
        print_rank0(f"EPOCH: [{epoch}], BATCH: [{train_sampler.progress}/{len(train_sampler)}], "+rpt)
        metrics["train"].reset_local()

        print_rank0(f"Saving checkpoint at epoch {epoch} train batch {train_sampler.progress}")
        train_sampler.advance()

        if train_sampler.progress == len(data_loader_train):
            metrics["train"].end_epoch()

        if utils.is_main_process() and train_sampler.progress % 1 == 0: # Checkpointing every batch

            total_progress = train_sampler.progress + epoch * len(train_sampler)
            writer = SummaryWriter(log_dir=args.tboard_path)
            for metric,val in zip(report_metrics, vals):
                writer.add_scalar("Train/"+metric, val, total_progress)
            writer.add_scalar("Train/learn_rate", lr_scheduler.get_last_lr()[0], total_progress)
            writer.flush()
            writer.close()
                
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "test_sampler": test_sampler.state_dict(),
                # Evaluator state variables
                "img_ids": coco_evaluator.img_ids, # catalogue of images seen already
                "eval_imgs": coco_evaluator.eval_imgs, # image evaluations
                "metrics": metrics,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            atomic_torch_save(checkpoint, args.resume)

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
    optimizer, lr_scheduler, train_sampler, device, 
    scaler, timer, metrics
):

    timer.report('starting evaluation routine')

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    timer.report('evaluation preliminaries')

    test_step = test_sampler.progress // data_loader_test.batch_size
    print_rank0(f'\nEvaluating / resuming epoch {epoch} from eval step {test_step}\n')

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

        print_rank0(f"Saving checkpoint at epoch {epoch} eval batch {test_step}")
        test_sampler.advance(len(images))
        test_step = test_sampler.progress // data_loader_test.batch_size

        if test_sampler.progress == len(data_loader_test):

            # gather the stats from all processes
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            results = coco_evaluator.summarize()

            metric_names = [
                "bbox/AP", "bbox/AP-50", "bbox/AP-75", "bbox/AP-S", "bbox/AP-M", "bbox/AP-L", 
                "bbox/AR-MD1", "bbox/AR-MD10", "bbox/AR-MD100", "bbox/AR-S", "bbox/AR-M", "bbox/AR-L",
                "segm/AP", "segm/AP-50", "segm/AP-75", "segm/AP-S", "segm/AP-M", "segm/AP-L", 
                "segm/AR-MD1", "segm/AR-MD10", "segm/AR-MD100", "segm/AR-S", "segm/AR-M", "segm/AR-L"
            ]

            metrics["val"].update({name: val for name,val in zip(metric_names, results)})
            metrics["val"].reduce()
            # Normalise validation metrics by world_size
            ngpus = dist.get_world_size()
            metrics["val"].agg = {k:v/ngpus for k,v in metrics["val"].agg.items()}
            metrics["val"].end_epoch()

            if utils.is_main_process():
                writer = SummaryWriter(log_dir=args.tboard_path)
                for name,val in metrics["val"].epoch_reports[-1].items():
                    writer.add_scalar("Val/"+name, val/ngpus, epoch)
                writer.flush()
                writer.close()

            torch.set_num_threads(n_threads)

            # Reset the coco evaluator at the end of the epoch
            coco = get_coco_api_from_dataset(data_loader_test.dataset)
            iou_types = _get_iou_types(model)
            coco_evaluator = CocoEvaluator(coco, iou_types)

            timer.report('evaluator accumulation, summarization, and reset')

        if utils.is_main_process() and test_step % 1 == 0: # Checkpointing every batch
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "test_sampler": test_sampler.state_dict(),
                # Evaluator state variables
                "img_ids": coco_evaluator.img_ids, # catalogue of images seen already
                "eval_imgs": coco_evaluator.eval_imgs, # image evaluations
                "metrics": metrics,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            atomic_torch_save(checkpoint, args.resume)

    return coco_evaluator, timer, metrics
