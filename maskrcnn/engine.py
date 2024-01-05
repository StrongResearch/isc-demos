import os
import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from cycling_utils import atomic_torch_save

from torch.utils.tensorboard import SummaryWriter

def print_rank0(message):
    if int(os.environ["RANK"]) == 0:
        print(message)

def train_one_epoch(
        model, optimizer, dataloader_train, dataloader_test,
        lr_scheduler, args, device, coco_evaluator, 
        epoch, scaler, timer
    ):

    model.train()

    accumulated = 0
    total_batches = len(dataloader_train.batch_sampler)
    
    print_rank0(f'\nTraining / resuming epoch {epoch} from training step {dataloader_train.batch_sampler.progress}\n')

    for images, targets in dataloader_train:

        batch = dataloader_train.batch_sampler.progress + 1

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        timer.report(f'Epoch: {epoch} batch [{batch}/{total_batches}]: batch to device')

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            timer.report(f'Epoch: {epoch} batch [{batch}/{total_batches}]: forward')

            losses = sum(loss for loss in loss_dict.values())
            losses = losses / args.accumulation_steps
            timer.report(f'Epoch: {epoch} batch [{batch}/{total_batches}]: loss')

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        loss_dict_reduced["total_loss"] = loss_value

        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()
        accumulated += 1
        timer.report(f'Epoch: {epoch} batch [{batch}/{total_batches}]: backward')

        if accumulated == args.accumulation_steps or batch == len(dataloader_train):
            if scaler is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm, norm_type='inf')
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm, norm_type='inf')
                optimizer.step()

            optimizer.zero_grad()
            accumulated = 0
        lr_scheduler.step()
        timer.report(f'Epoch: {epoch} batch [{batch}/{total_batches}]: update')
        
        rpt = ", ".join([f"{m}: {v:,.3f}" for m,v in loss_dict_reduced.items()])
        print_rank0(f"EPOCH: [{epoch}], BATCH: [{batch}/{total_batches}], "+rpt)

        dataloader_train.batch_sampler.advance()

        if utils.is_main_process(): # Checkpointing every batch
            total_progress = batch + epoch * total_batches
            writer = SummaryWriter(log_dir=args.tboard_path)
            for metric,val in loss_dict_reduced.items():
                writer.add_scalar("Train/"+metric, val, total_progress)
            writer.add_scalar("Train/learn_rate", lr_scheduler.get_last_lr()[0], total_progress)
            writer.flush()
            writer.close()
                
        checkpointing = batch % 10 == 0 or batch == total_batches
        if utils.is_main_process() and checkpointing:
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_sampler": dataloader_train.batch_sampler.state_dict(),
                "test_sampler": dataloader_test.sampler.state_dict(),
                # Evaluator state variables
                "img_ids": coco_evaluator.img_ids, # catalogue of images seen already
                "eval_imgs": coco_evaluator.eval_imgs, # image evaluations
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            atomic_torch_save(checkpoint, args.resume)

    return model, timer

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
    model, optimizer, dataloader_train, dataloader_test,
        lr_scheduler, args, device, coco_evaluator, 
        epoch, scaler, timer
):

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    timer.report('evaluation preliminaries')

    batch = (dataloader_test.sampler.progress // dataloader_test.batch_size) + 1
    total_batches = len(dataloader_test)
    print_rank0(f'\nEvaluating / resuming epoch {epoch} from eval batch [{batch}]\n')

    for images, targets in dataloader_test:

        batch = (dataloader_test.sampler.progress // dataloader_test.batch_size) + 1
        images = list(img.to(device) for img in images)
        timer.report(f'Epoch {epoch} batch [{batch}/{total_batches}]: batch to device')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs = model(images)
        timer.report(f'Epoch {epoch} batch [{batch}/{total_batches}]: inference')

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        timer.report(f'Epoch {epoch} batch [{batch}/{total_batches}]: outputs to cpu')

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        # res -> {img_id: {'boxes': T, 'labels': T, 'scores': T, 'masks': T}, ...}
        coco_evaluator.update(res)
        timer.report(f'Epoch {epoch} batch [{batch}/{total_batches}]: update evaluator')

        dataloader_test.sampler.advance(len(images))

        if batch == total_batches:

            # gather the stats from all processes
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            results = coco_evaluator.summarize()
            torch.set_num_threads(n_threads)

            if args.model.startswith("maskrcnn"):
                metric_names = [
                    "bbox/AP", "bbox/AP-50", "bbox/AP-75", "bbox/AP-S", "bbox/AP-M", "bbox/AP-L", 
                    "bbox/AR-MD1", "bbox/AR-MD10", "bbox/AR-MD100", "bbox/AR-S", "bbox/AR-M", "bbox/AR-L",
                    "segm/AP", "segm/AP-50", "segm/AP-75", "segm/AP-S", "segm/AP-M", "segm/AP-L", 
                    "segm/AR-MD1", "segm/AR-MD10", "segm/AR-MD100", "segm/AR-S", "segm/AR-M", "segm/AR-L"
                ]
            elif args.model.startswith("retinanet"):
                metric_names = [
                    "bbox/AP", "bbox/AP-50", "bbox/AP-75", "bbox/AP-S", "bbox/AP-M", "bbox/AP-L", 
                    "bbox/AR-MD1", "bbox/AR-MD10", "bbox/AR-MD100", "bbox/AR-S", "bbox/AR-M", "bbox/AR-L"
                ]

            if utils.is_main_process():
                writer = SummaryWriter(log_dir=args.tboard_path)
                for name,val in zip(metric_names, results):
                    writer.add_scalar("Val/"+name, val, epoch)
                writer.flush()
                writer.close()

            # Reset the coco evaluator at the end of the epoch
            coco = get_coco_api_from_dataset(dataloader_test.dataset)
            iou_types = _get_iou_types(model)
            coco_evaluator = CocoEvaluator(coco, iou_types)

            timer.report('evaluator accumulation, summarization, and reset')

        checkpointing = batch % 1 == 0 or batch == total_batches
        if utils.is_main_process() and checkpointing: # Checkpointing every batch
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_sampler": dataloader_train.batch_sampler.state_dict(),
                "test_sampler": dataloader_test.sampler.state_dict(),
                # Evaluator state variables
                "img_ids": coco_evaluator.img_ids, # catalogue of images seen already
                "eval_imgs": coco_evaluator.eval_imgs, # image evaluations
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            atomic_torch_save(checkpoint, args.resume)

    return coco_evaluator, timer
