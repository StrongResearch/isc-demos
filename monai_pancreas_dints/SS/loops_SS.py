
import torch
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast
# from datetime import datetime
# from scipy import ndimage
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
# import yaml, time, 
import os
import utils
from cycling_utils import atomic_torch_save
from torch.utils.tensorboard import SummaryWriter

def search_one_epoch(
    model, optimizer, dints_space, arch_optimizer_a, arch_optimizer_c, 
    train_sampler, val_sampler, scaler, train_metrics, val_metric,
    epoch, train_loader, loss_func, args, timer
):
    device = args["device"] # for convenience

    decay = 0.5 ** np.sum(
        [(epoch - args["num_epochs_warmup"]) / (args["num_epochs"] - args["num_epochs_warmup"]) > args["learning_rate_milestones"]]
    )
    lr = args["learning_rate"] * decay * args["world_size"]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    model.train()

    timer.report('model.train()')

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for batch_data in train_loader:

        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        inputs_search, labels_search = inputs.detach().clone(), labels.detach().clone() # added, will this work?

        timer.report('data to device')

        # UPDATE MODEL

        for p in model.module.weight_parameters():
            p.requires_grad=True
        dints_space.log_alpha_a.requires_grad = False
        dints_space.log_alpha_c.requires_grad = False

        optimizer.zero_grad()

        timer.report('config model to train')

        if args["amp"]:
            with autocast():
                outputs = model(inputs)
                timer.report('model forward')
                if args["output_classes"] == 2:
                    loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                else:
                    loss = loss_func(outputs, labels)
                timer.report('model loss')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            timer.report('model backward')
        else:
            outputs = model(inputs)
            timer.report('model forward')
            if args["output_classes"] == 2:
                loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
            else:
                loss = loss_func(outputs, labels)
            timer.report('model loss')
            loss.backward()
            optimizer.step()
            timer.report('model backward')

        # Reporting and stuff
        train_metrics.update({"model_loss": loss.item(), "inputs_seen": len(inputs)})

        timer.report('model update')

        # Only update space after number of warmup epochs
        if epoch >= args["num_epochs_warmup"]:

            # UPDATE SPACE

            for p in model.module.weight_parameters():
                p.requires_grad=False
            dints_space.log_alpha_a.requires_grad = True
            dints_space.log_alpha_c.requires_grad = True

            # linear increase topology and RAM loss
            entropy_alpha_c = torch.tensor(0.0,).to(device)
            entropy_alpha_a = torch.tensor(0.0).to(device)
            ram_cost_full = torch.tensor(0.0).to(device)
            ram_cost_usage = torch.tensor(0.0).to(device)
            ram_cost_loss = torch.tensor(0.0).to(device)
            topology_loss = torch.tensor(0.0).to(device)

            probs_a, arch_code_prob_a = dints_space.get_prob_a(child=True)
            entropy_alpha_a = -((probs_a) * torch.log(probs_a + 1e-5)).mean()
            sm = F.softmax(dints_space.log_alpha_c, dim=-1)
            lsm = F.log_softmax(dints_space.log_alpha_c, dim=-1)
            entropy_alpha_c = -(sm * lsm).mean()
            topology_loss = dints_space.get_topology_entropy(probs_a)

            ram_cost_full = dints_space.get_ram_cost_usage(inputs.shape, full=True)
            ram_cost_usage = dints_space.get_ram_cost_usage(inputs.shape)
            ram_cost_loss = torch.abs(args["ram_cost_factor"] - ram_cost_usage / ram_cost_full)

            arch_optimizer_a.zero_grad()
            arch_optimizer_c.zero_grad()

            combination_weights = (epoch - args["num_epochs_warmup"]) / (args["num_epochs"] - args["num_epochs_warmup"])

            timer.report('space combination_weights')

            if args["amp"]:
                with autocast():
                    outputs_search = model(inputs_search)
                    timer.report('space forward')
                    if args["output_classes"] == 2:
                        loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                    else:
                        loss = loss_func(outputs_search, labels_search)

                    loss += combination_weights * (
                        (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                    )
                    timer.report('space loss')

                scaler.scale(loss).backward()
                scaler.step(arch_optimizer_a)
                scaler.step(arch_optimizer_c)
                scaler.update()
                timer.report('space backward')
            else:
                outputs_search = model(inputs_search)
                timer.report('space forward')
                if args["output_classes"] == 2:
                    loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                else:
                    loss = loss_func(outputs_search, labels_search)

                loss += 1.0 * (
                    combination_weights * (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                )
                timer.report('space loss')

                loss.backward()
                arch_optimizer_a.step()
                arch_optimizer_c.step()
                timer.report('space backward')

            # Reporting and stuff
            train_metrics.update({"space_loss": loss.item()})

            timer.report('space update')

        # Batch reporting
        train_metrics.reduce()
        batch_model_loss = train_metrics.local["model_loss"] / train_metrics.local["inputs_seen"]
        if "space_loss" in train_metrics.local:
            batch_space_loss = train_metrics.local["space_loss"] / train_metrics.local["inputs_seen"]
        else:
            batch_space_loss = 0.0
        print(f"EPOCH [{epoch}], BATCH [{train_step}], MODEL LOSS [{batch_model_loss:,.3f}, SPACE LOSS: [{batch_space_loss:,.3f}]")
        train_metrics.reset_local()

        timer.report('metrics reduce')

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(inputs))
        train_step = train_sampler.progress // train_loader.batch_size

        if train_step == total_steps:
            train_metrics.end_epoch()

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch

            writer = SummaryWriter(log_dir=args["tboard_path"])
            writer.add_scalar("Train/model_loss", batch_model_loss, train_step + epoch * total_steps)
            if batch_space_loss != "NONE":
                writer.add_scalar("Train/space_loss", batch_space_loss, train_step + epoch * total_steps)
            writer.flush()
            writer.close()

            checkpoint = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "dints": dints_space.state_dict(),
                "optimizer": optimizer.state_dict(),
                "arch_optimizer_a": arch_optimizer_a.state_dict(),
                "arch_optimizer_c": arch_optimizer_c.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                "scaler": scaler.state_dict(),
                "train_metrics": train_metrics,
                "val_metric": val_metric
            }
            timer = atomic_torch_save(checkpoint, args["resume"], timer)

    timer.report(f'EPOCH {epoch}')

    return model, dints_space, timer, train_metrics


def eval_search(
        model, optimizer, dints_space, arch_optimizer_a, arch_optimizer_c, 
        train_sampler, val_sampler, scaler, train_metrics, val_metric,
        epoch, val_loader, post_pred, post_label, args, timer,
):
    device = args["device"] # for convenience

    torch.cuda.empty_cache()
    model.eval()

    timer.report('model ready to eval')

    with torch.no_grad():

        val_step = val_sampler.progress // val_loader.batch_size
        total_steps = int(len(val_sampler) / val_loader.batch_size)
        print(f'\nEvaluating / resuming epoch {epoch} from eval step {val_step}\n')

        for val_data in val_loader:

            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            roi_size = args["patch_size_valid"]
            sw_batch_size = args["num_sw_batch_size"]

            if args["amp"]:
                with torch.cuda.amp.autocast():
                    pred = sliding_window_inference(
                        val_images, roi_size, sw_batch_size,
                        lambda x: model(x), mode="gaussian",
                        overlap=args["overlap_ratio"],
                    )
            else:
                pred = sliding_window_inference(
                    val_images, roi_size, sw_batch_size,
                    lambda x: model(x), mode="gaussian",
                    overlap=args["overlap_ratio"],
                )

            val_outputs = post_pred(pred[0, ...])
            val_outputs = val_outputs[None, ...]
            val_labels = post_label(val_labels[0, ...])
            val_labels = val_labels[None, ...]

            value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=False)

            for _c in range(args["output_classes"] - 1):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, 0]).float()
                val_metric[2 * _c] += val0 * val1
                val_metric[2 * _c + 1] += val1

            ## Checkpointing
            print(f"Saving checkpoint at epoch {epoch} eval batch {val_step}")
            val_sampler.advance(len(val_images))
            val_step = val_sampler.progress // val_loader.batch_size

            if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch

                checkpoint = {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "dints": dints_space.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "arch_optimizer_a": arch_optimizer_a.state_dict(),
                    "arch_optimizer_c": arch_optimizer_c.state_dict(),
                    "train_sampler": train_sampler.state_dict(),
                    "val_sampler": val_sampler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metric": val_metric
                }
                timer = atomic_torch_save(checkpoint, args["resume"], timer)

            timer.report(f'eval step {val_step}')

        # synchronizes all processes and reduce results
        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(val_metric, op=torch.distributed.ReduceOp.SUM)

        val_metric = val_metric.tolist()
        if utils.is_main_process():

            for _c in range(args["output_classes"] - 1):
                print("evaluation metric - class {0:d}:".format(_c + 1), val_metric[2 * _c] / val_metric[2 * _c + 1])
            avg_metric = 0
            for _c in range(args["output_classes"] - 1):
                avg_metric += val_metric[2 * _c] / val_metric[2 * _c + 1]
            avg_metric = avg_metric / float(args["output_classes"] - 1)
            print("avg_metric", avg_metric)

            if avg_metric > best_metric:
                best_metric = avg_metric
                # best_metric_epoch = epoch + 1
                # best_metric_iterations = idx_iter

                (node_a_d, arch_code_a_d, arch_code_c_d, arch_code_a_max_d) = dints_space.decode()
                torch.save(
                    {
                        "node_a": node_a_d,
                        "arch_code_a": arch_code_a_d,
                        "arch_code_a_max": arch_code_a_max_d,
                        "arch_code_c": arch_code_c_d,
                        # "iter_num": idx_iter,
                        "epochs": epoch + 1,
                        "best_dsc": best_metric,
                        # "best_path": best_metric_iterations,
                    },
                    os.path.join(args["arch_ckpt_path"], "search_code.pt"),
                )
    
    timer.report(f'EVAL EPOCH {epoch}')

    return timer


def train_one_epoch(
    model, optimizer, lr_scheduler,
    train_sampler, val_sampler, scaler, train_metrics, val_metric,
    epoch, train_loader, loss_func, args
):
    device = args["device"] # for convenience

    model.train()

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for batch_data in train_loader:

        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()

        if args["amp"]:
            with autocast():
                outputs = model(inputs)
                if args["output_classes"] == 2:
                    loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                else:
                    loss = loss_func(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if args["output_classes"] == 2:
                loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
            else:
                loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

        # Reporting and stuff
        train_metrics.update({"model_loss": loss.item(), "inputs_seen": len(inputs)})
        train_metrics.reduce()
        batch_model_loss = train_metrics.local["model_loss"] / train_metrics.local["inputs_seen"]
        print(f"EPOCH [{epoch}], BATCH [{train_step}], MODEL LOSS [{batch_model_loss:,.3f}]")
        train_metrics.reset_local()

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(inputs))
        train_step = train_sampler.progress // train_loader.batch_size

        if train_step == total_steps:
            train_metrics.end_epoch()
            lr_scheduler.step()

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch

            writer = SummaryWriter(log_dir=args["tboard_path"])
            writer.add_scalar("Train/model_loss", batch_model_loss, train_step + epoch * total_steps)
            writer.flush()
            writer.close()

            checkpoint = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                "scaler": scaler.state_dict(),
                "train_metrics": train_metrics,
                "val_metric": val_metric,
            }
            timer = atomic_torch_save(checkpoint, args["resume"], timer)

    return model, timer, train_metrics


def evaluate(
        model, optimizer, lr_scheduler,
        train_sampler, val_sampler, scaler, train_metrics, val_metric,
        epoch, val_loader, post_pred, post_label, args,
):
    device = args["device"] # for convenience

    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad():

        val_step = val_sampler.progress // val_loader.batch_size
        print(f'\nEvaluating / resuming epoch {epoch} from eval step {val_step}\n')

        for val_data in val_loader:

            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            roi_size = args["patch_size_valid"]
            sw_batch_size = args["num_sw_batch_size"]

            if args["amp"]:
                with torch.cuda.amp.autocast():
                    pred = sliding_window_inference(
                        val_images, roi_size, sw_batch_size,
                        lambda x: model(x), mode="gaussian",
                        overlap=args["overlap_ratio"],
                    )
            else:
                pred = sliding_window_inference(
                    val_images, roi_size, sw_batch_size,
                    lambda x: model(x), mode="gaussian",
                    overlap=args["overlap_ratio"],
                )

            val_outputs = post_pred(pred[0, ...])
            val_outputs = val_outputs[None, ...]
            val_labels = post_label(val_labels[0, ...])
            val_labels = val_labels[None, ...]

            value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=False)

            for _c in range(args["output_classes"] - 1):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, 0]).float()
                val_metric[2 * _c] += val0 * val1
                val_metric[2 * _c + 1] += val1

            ## Checkpointing
            print(f"Saving checkpoint at epoch {epoch} eval batch {val_step}")
            val_sampler.advance(len(val_images))
            val_step = val_sampler.progress // val_loader.batch_size

            if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch

                checkpoint = {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_sampler": train_sampler.state_dict(),
                    "val_sampler": val_sampler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metric": val_metric,
                }
                timer = atomic_torch_save(checkpoint, args["resume"], timer)

        # synchronizes all processes and reduce results
        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(val_metric, op=torch.distributed.ReduceOp.SUM)

        val_metric = val_metric.tolist()
        if utils.is_main_process():

            for _c in range(args["output_classes"] - 1):
                print("evaluation metric - class {0:d}:".format(_c + 1), val_metric[2 * _c] / val_metric[2 * _c + 1])
            avg_metric = 0
            for _c in range(args["output_classes"] - 1):
                avg_metric += val_metric[2 * _c] / val_metric[2 * _c + 1]
            avg_metric = avg_metric / float(args["output_classes"] - 1)
            print("avg_metric", avg_metric)

            writer = SummaryWriter(log_dir=args["tboard_path"])
            writer.add_scalar("Val/avg_metric", avg_metric, epoch)
            writer.flush()
            writer.close()