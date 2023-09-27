
import torch
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast
from datetime import datetime
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
import yaml, time, os, utils
from cycling_utils import atomic_torch_save

# args = {
#         "resume": parser["resume"],
#         "arch_ckpt_path": parser["arch_ckpt_path"],
#         "amp": parser["amp"],
#         "data_file_base_dir": parser["data_file_base_dir"],
#         "data_list_file_path": parser["data_list_file_path"],
#         "determ": parser["determ"],
#         "learning_rate": parser["learning_rate"],
#         "learning_rate_arch": parser["learning_rate_arch"],
#         "learning_rate_milestones": np.array(parser["learning_rate_milestones"]),
#         "num_images_per_batch": parser["num_images_per_batch"],
#         "num_epochs": parser["num_epochs"],  # around 20k iterations
#         "num_epochs_per_validation": parser["num_epochs_per_validation"],
#         "num_epochs_warmup": parser["num_epochs_warmup"],
#         "num_sw_batch_size": parser["num_sw_batch_size"],
#         "output_classes": parser["output_classes"],
#         "overlap_ratio": parser["overlap_ratio"],
#         "patch_size_valid": parser["patch_size_valid"],
#         "ram_cost_factor": parser["ram_cost_factor"],

#         "start_epoch": 0,
#     }

def search_one_epoch(
    # Stateful objs that will need to be checkpointed
    model, optimizer, dints_space, arch_optimizer_a, arch_optimizer_c, train_sampler, val_sampler, model_scaler, space_scaler, metrics,
    # Stateless callables
    train_loader, loss_func, writer,
    # Mutable constants
    epoch, args
):

    decay = 0.5 ** np.sum(
        [(epoch - args["num_epochs_warmup"]) / (args["num_epochs"] - args["num_epochs_warmup"]) > args["learning_rate_milestones"]]
    )
    lr = args["learning_rate"] * decay * args["world_size"]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    device = args["device"] # for convenience
    model.train()

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for batch_data in train_loader:

        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        inputs_search, labels_search = inputs.detach().clone(), labels.detach().clone() # added, will this work?

        # UPDATE MODEL

        # if args["world_size"] == 1:
        #     for _ in model.weight_parameters():
        #         _.requires_grad = True
        # else:
        #     for _ in model.module.weight_parameters():
        #         _.requires_grad = True

        for p in model.module.weight_parameters():
            p.requires_grad=True
        dints_space.log_alpha_a.requires_grad = False
        dints_space.log_alpha_c.requires_grad = False

        optimizer.zero_grad()

        if args["amp"]:
            with autocast():
                outputs = model(inputs)
                if args["output_classes"] == 2:
                    loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                else:
                    loss = loss_func(outputs, labels)

            model_scaler.scale(loss).backward()
            model_scaler.step(optimizer)
            model_scaler.update()
        else:
            outputs = model(inputs)
            if args["output_classes"] == 2:
                loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
            else:
                loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

        # Reporting and stuff
        metrics.update({"model_loss": loss.item(), "inputs_seen": len(inputs)})

        # Only update space after number of warmup epochs
        if epoch >= args["num_epochs_warmup"]:

            # UPDATE SPACE

            # if args["world_size"] == 1:
            #     for _ in model.weight_parameters():
            #         _.requires_grad = False
            # else:
            #     for _ in model.module.weight_parameters():
            #         _.requires_grad = False

            for p in model.module.weight_parameters():
                p.requires_grad=False
            dints_space.log_alpha_a.requires_grad = True
            dints_space.log_alpha_c.requires_grad = True

            # linear increase topology and RAM loss
            entropy_alpha_c = torch.tensor(0.0).to(device)
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

            if args["amp"]:
                with autocast():
                    outputs_search = model(inputs_search)
                    if args["output_classes"] == 2:
                        loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                    else:
                        loss = loss_func(outputs_search, labels_search)

                    loss += combination_weights * (
                        (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                    )

                space_scaler.scale(loss).backward()
                space_scaler.step(arch_optimizer_a)
                space_scaler.step(arch_optimizer_c)
                space_scaler.update()
            else:
                outputs_search = model(inputs_search)
                if args["output_classes"] == 2:
                    loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                else:
                    loss = loss_func(outputs_search, labels_search)

                loss += 1.0 * (
                    combination_weights * (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                )

                loss.backward()
                arch_optimizer_a.step()
                arch_optimizer_c.step()

            # Reporting and stuff
            metrics.update({"space_loss": loss.item()})

        # Batch reporting
        metrics.reduce()

        batch_model_loss = metrics.local["model_loss"] / metrics.local["inputs_seen"]
        batch_space_loss = metrics.local["space_loss"] / metrics.local["inputs_seen"]
        metrics.reset_local()
        print(f"EPOCH [{epoch}], BATCH [{train_step}], MODEL LOSS [{batch_model_loss:,.3f}, SPACE LOSS: [{batch_space_loss:,.3f}]")

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(inputs))
        train_step = train_sampler.progress // train_loader.batch_size

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch

            writer.add_scalar("Train/model_loss", batch_model_loss, train_step + epoch * total_steps)
            writer.add_scalar("Train/space_loss", batch_space_loss, train_step + epoch * total_steps)

            checkpoint = {
                "model": model.module.state_dict(),
                "dints": dints_space.state_dict(),
                "optimizer": optimizer.state_dict(),
                "arch_optimizer_a": arch_optimizer_a.state_dict(),
                "arch_optimizer_c": arch_optimizer_c.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                "model_scaler": model_scaler.state_dict(),
                "space_scaler": space_scaler.state_dict(),
                "metrics": metrics
            }
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    return model, dints_space


def eval_search(
        model, output_classes, device, val_loader, patch_size_valid, num_sw_batch_size, overlap_ratio, 
        post_pred, post_label, epoch, idx_iter
):

    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad():

        metric = torch.zeros((output_classes - 1) * 2, dtype=torch.float, device=device)
        metric_sum = 0.0
        metric_count = 0
        metric_mat = []
        val_images = None
        val_labels = None
        val_outputs = None

        _index = 0
        for val_data in val_loader:

            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            roi_size = patch_size_valid
            sw_batch_size = num_sw_batch_size

            if amp:
                with torch.cuda.amp.autocast():
                    pred = sliding_window_inference(
                        val_images,
                        roi_size,
                        sw_batch_size,
                        lambda x: model(x),
                        mode="gaussian",
                        overlap=overlap_ratio,
                    )
            else:
                pred = sliding_window_inference(
                    val_images,
                    roi_size,
                    sw_batch_size,
                    lambda x: model(x),
                    mode="gaussian",
                    overlap=overlap_ratio,
                )

            val_outputs = pred
            val_outputs = post_pred(val_outputs[0, ...])
            val_outputs = val_outputs[None, ...]
            val_labels = post_label(val_labels[0, ...])
            val_labels = val_labels[None, ...]

            value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=False)

            print(_index + 1, "/", len(val_loader), value)

            metric_count += len(value)
            metric_sum += value.sum().item()
            metric_vals = value.cpu().numpy()
            if len(metric_mat) == 0:
                metric_mat = metric_vals
            else:
                metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

            for _c in range(output_classes - 1):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, 0]).float()
                metric[2 * _c] += val0 * val1
                metric[2 * _c + 1] += val1

            _index += 1

            ## SAVE CHECKPOINT

        # synchronizes all processes and reduce results
        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

        metric = metric.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            for _c in range(output_classes - 1):
                print("evaluation metric - class {0:d}:".format(_c + 1), metric[2 * _c] / metric[2 * _c + 1])
            avg_metric = 0
            for _c in range(output_classes - 1):
                avg_metric += metric[2 * _c] / metric[2 * _c + 1]
            avg_metric = avg_metric / float(output_classes - 1)
            print("avg_metric", avg_metric)

            if avg_metric > best_metric:
                best_metric = avg_metric
                best_metric_epoch = epoch + 1
                best_metric_iterations = idx_iter

    return best_metric_epoch, best_metric_iterations
