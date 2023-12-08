import json
import logging
import math
import os
import time

import socket

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed import get_world_size
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

from cycling_utils import atomic_torch_save

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

# This function saves the inner-epoch state of the run so it can be easily resumed
def save_train_checkpoint(epoch, iteration, model, optimizer, sampler, val_sampler, scaler, path):
    scaler_dict = None
    if scaler is not None:
        scaler_dict = scaler.state_dict()
    checkpoint_dict = {
                "epoch": epoch,
                "iteration": iteration,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                "scaler": scaler_dict
            }
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    atomic_torch_save(checkpoint_dict, path)

def train_one_epoch(model, data, loss, epoch, iters, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    metric_data = []
    torch.cuda.cudart().cudaProfilerStart()
    # No need to enumerate since we pass in iterations
    for batch in dataloader:
        if iters >= 200:
            torch.cuda.cudart().cudaProfilerStop()
            return
        # Advance sampler by batch size before checkpointing
        dataloader.sampler.advance(args.batch_size)
        iters += 1
        if is_master(args) and iters % 1 == 0:
            logging.info(f"Training - {iters}/{len(dataloader)}")
        i_accum = (iters - 1) // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)
                if torch.isnan(losses["contrastive_loss"]).any():
                    print(f"Nan on {device} + {socket.gethostname()}")
                if torch.isinf(losses["contrastive_loss"]).any():
                    print(f"Inf on {device} + {socket.gethostname()}")

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
            
            for param in model.parameters():
                if torch.isinf(param.grad).any():
                    print(f"inf at {device} + {socket.gethostname()}")
                if torch.isnan(param.grad).any():
                    print(f"nan at {device} + {socket.gethostname()}")
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((iters) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    if torch.isnan(losses["contrastive_loss"]).any():
                        print(f"Nan on {device} + {socket.gethostname()}")
                    if torch.isinf(losses["contrastive_loss"]).any():
                        print(f"Inf on {device} + {socket.gethostname()}")

                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)
                for param in model.parameters():
                    if torch.isinf(param.grad).any():
                        print(f"inf at {device} + {socket.gethostname()}")
                    if torch.isnan(param.grad).any():
                        print(f"nan at {device} + {socket.gethostname()}")

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # Gather metric data from each iteration but only write when checkpointing to avoid writing twice to the same iteration
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        batch_loss = losses["loss"]
        dist.all_reduce(batch_loss, op=dist.ReduceOp.SUM)
        batch_loss = torch.div(batch_loss, dist.get_world_size())
        metric_data.append((batch_loss, (len(dataloader) * epoch) + iters))
        
        # Save checkpoint if at frequency or end of dataloader
        # Also writes to tensorboard
        if is_master(args):
            if iters >= len(dataloader):
                #  save checkpoint and break
                logging.info("Reached the end of the dataloader - saving checkpoint")
                save_train_checkpoint(epoch, iters, model, optimizer, dataloader.sampler, data["val"].dataloader.sampler, scaler, args.resume)
                if tb_writer is not None:
                    for scalar in metric_data:
                        tb_writer.add_scalar("Train/avg_loss", scalar[0], scalar[1])
                        
                    tb_writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], (len(dataloader) * epoch) + iters)
                    
                    metric_data = []
                    logging.info("Finished writing log data")
                #break
            elif iters % args.save_frequency == 0:  # Save checkpoint every n iterations
                logging.info(f"Saving checkpoint at epoch {epoch} and iteration {iters}/{len(dataloader)}")
                save_train_checkpoint(epoch, iters, model, optimizer, dataloader.sampler, data["val"].dataloader.sampler, scaler, args.resume)
                if tb_writer is not None:
                    for scalar in metric_data:
                        tb_writer.add_scalar("Train/avg_loss", scalar[0], scalar[1])
                    tb_writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], iters)
                    metric_data = []
                    logging.info("Finished writing log data")

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = len(dataloader) * get_world_size() * batch_size

            percent_complete = ((iters - 1) * batch_size * get_world_size() * 100) / samples_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            
            logging.info(
                f"Train Epoch: {epoch} [{(iters - 1) * batch_size * get_world_size()}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}
            
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

    # end for


# def save_eval_checkpoint(sampler, iteration, cum_loss, cum_gen_loss, img_features, txt_features, path):
#     checkpoint_dict = {
#                 "sampler": sampler.state_dict(),
#                 "iteration": iteration,
#                 "cum_loss": cum_loss,
#                 "cum_gen_loss": cum_gen_loss,
#                 "img_features": img_features,
#                 "txt_features": txt_features,
#             }
#     atomic_torch_save(checkpoint_dict, path)

# def load_eval_checkpoint(path):
#     if os.path.isfile(path):
#         checkpoint = torch.load(path, map_location="cpu")
#         logging.info(f"Checkpoint loaded with iters = {checkpoint['iteration']}")
#         return ((checkpoint["epoch"], checkpoint["iteration"], checkpoint["cum_loss"], 
#                 checkpoint["cum_gen_loss"], checkpoint["img_features"], checkpoint["txt_features"]), checkpoint["sampler"])
#     logging.info("No eval checkpoint found, starting eval from scratch.")
#     return False

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    # if not is_master(args):
    #     return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data:
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print(f"Eval - {i}/{len(dataloader)}")
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                
                # Advance the sampler by the batch size - ignore when it steps too far on the last step
                try:
                    dataloader.sampler.advance(args.batch_size)
                except Exception:
                    logging.info("Sampler stepped too far at the end of batch - ignoring")
                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2
                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if (i % 100) == 0:
                    if args.distributed_evaluation:
                        total_num_samples = torch.Tensor([num_samples]).to(device)
                        print(7)
                        torch.distributed.all_reduce(total_num_samples)
                        print(8)
                        total_num_samples = total_num_samples.item()

                        total_cumulative_loss = cumulative_loss.clone()
                        torch.distributed.all_reduce(total_cumulative_loss)

                        loss = total_cumulative_loss / total_num_samples
                    else:
                        loss = cumulative_loss / num_samples
                    if is_master(args):
                        logging.info(
                            f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                            f"Loss: {loss:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
                args=args,
            )
            
            total_num_samples = torch.Tensor([num_samples]).to(device)
            torch.distributed.all_reduce(total_num_samples)
            total_num_samples = int(total_num_samples.item())

            total_cumulative_loss = cumulative_loss.clone()
            torch.distributed.all_reduce(total_cumulative_loss)

            loss = total_cumulative_loss / total_num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": total_num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"Val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)
        os.makedirs(args.checkpoint_path, exist_ok=True)
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale, args):
    metrics = {}
    
    image_features = varsize_tensor_all_gather(image_features.to(args.device)).cpu()
    text_features = varsize_tensor_all_gather(text_features.to(args.device)).cpu()
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
def varsize_tensor_all_gather(tensor: torch.Tensor):
    # https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4
    # thanks to @mranzinger
    device = tensor.device
    size_tens = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=device)
    size_tens = tensor_all_gather(size_tens).cpu()
    max_size = size_tens.max()

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=device)
    padded[:tensor.shape[0]] = tensor

    ag = tensor_all_gather(padded)

    slices = []
    for i, sz in enumerate(size_tens):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)

    return ret.to(tensor)

def tensor_all_gather(tensor):
    world_size = torch.distributed.get_world_size()
    tensor_list = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)