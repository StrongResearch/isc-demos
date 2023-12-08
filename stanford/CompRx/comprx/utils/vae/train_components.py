import os
from time import time
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from rich import print
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torch.utils.tensorboard import SummaryWriter
import torchvision
import shutil

from comprx.utils.extras import get_weight_dtype
from comprx.utils.transforms import to_dict

from cycling_utils import AtomicDirectory

__all__ = ["training_epoch", "validation_epoch"]


def training_epoch(
    epoch: int,
    global_step: int,
    local_step: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    discriminator_iter_start: int,
    default_metrics: List[Metric],
    rec_metrics: List[Metric],
    optimizer_ae: Optimizer,
    optimizer_disc: Optimizer,
    options: Dict[str, Any],
):
    """Train a single epoch of a VAE model."""
    
    saver = AtomicDirectory(options["ckpt_dir"], symlink_name="latest.pt", chk_dir_prefix="checkpoint_")
    
    
    for metric in default_metrics:
        metric.reset()

    for _, metric in rec_metrics:
        metric.reset()

    metric_aeloss, metric_discloss, metric_ae_recloss, metric_data, metric_batch = default_metrics

    model.train()
    epoch_start, batch_start = time(), time()
    dtype = get_weight_dtype(accelerator)
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir='./checkpoints/tb')

    for batch in dataloader:
        
        save_dir = saver.prepare_checkpoint_directory()
        
        if batch is None:
            return global_step

        data_time = time() - batch_start

        with accelerator.accumulate(model):
            batch = to_dict(batch)

            # Train the encoder and decoder
            images = batch["img"].to(dtype=dtype)
            reconstructions, posterior = model(images)
            dataloader.sampler.advance(len(images))

            aeloss, log_dict_ae = criterion(
                inputs=images,
                reconstructions=reconstructions,
                posteriors=posterior,
                optimizer_idx=0,
                global_step=global_step,
                weight_dtype=dtype,
                last_layer=accelerator.unwrap_model(model).get_last_layer(),
                split="train",
            )

            optimizer_ae.zero_grad()
            accelerator.backward(aeloss)
            optimizer_ae.step()

            # Train the discriminator
            discloss = torch.tensor(0.0)
            if global_step >= discriminator_iter_start:
                with torch.no_grad():
                    reconstructions, posterior = model(images)
                discloss, _log_dict_disc = criterion(
                    inputs=images,
                    reconstructions=reconstructions,
                    posteriors=posterior,
                    optimizer_idx=1,
                    global_step=global_step,
                    weight_dtype=dtype,
                    last_layer=accelerator.unwrap_model(model).get_last_layer(),
                    split="train",
                )

                optimizer_disc.zero_grad()
                accelerator.backward(discloss)
                optimizer_disc.step()

        # Update metrics
        batch_time = time() - batch_start
        metric_aeloss.update(aeloss)
        metric_ae_recloss.update(log_dict_ae["train/rec_loss"])
        metric_discloss.update(discloss)
        metric_data.update(data_time)
        metric_batch.update(batch_time)

        images, reconstructions = accelerator.gather_for_metrics((images, reconstructions))
        for _, metric in rec_metrics:
            metric.update(images, reconstructions)

        # Logging values
        print(
            f"\r[Epoch <{epoch:03}/{options['max_epoch']}>: Step <{(local_step):03}/{len(dataloader)}>] - "
            + f"Data(s): {data_time:.3f} ({metric_data.compute():.3f}) - "
            + f"Batch(s): {batch_time:.3f} ({metric_batch.compute():.3f}) - "
            + f"AE Loss: {aeloss.item():.3f} ({metric_aeloss.compute():.3f}) - "
            + f"AE Rec Loss: {log_dict_ae['train/rec_loss'].item():.3f} ({metric_ae_recloss.compute():.3f}) - "
            + f"Disc Loss: {discloss.item():.3f} ({metric_discloss.compute():.3f}) - "
            + f"{(((time() - epoch_start) / (local_step + 1)) * (len(dataloader) - local_step)) / 60:.2f} m remaining\n"
        )

        if options["is_logging"] and global_step % options["log_every_n_steps"] == 0:
            log_data = {
                "epoch": epoch,
                "mean_aeloss": metric_aeloss.compute(),
                "mean_ae_recloss": metric_ae_recloss.compute(),
                "mean_discloss": metric_discloss.compute(),
                "mean_data": metric_data.compute(),
                "mean_batch": metric_batch.compute(),
                "step": local_step,
                "step_global": global_step,
                "step_aeloss": aeloss,
                "step_ae_recloss": log_dict_ae["train/rec_loss"],
                "step_discloss": discloss,
                "step_data": data_time,
                "step_batch": batch_time,
            }

            for name, metric in rec_metrics:
                log_data[name] = metric.compute()

            accelerator.log(log_data)
        
        if accelerator.is_main_process and global_step % 10 == 0:
            batch_images = list(torch.cat([images, reconstructions], dim=0))
            grid = torchvision.utils.make_grid(batch_images, nrow=len(images))
            writer.add_image(f"examples rank{os.environ['RANK']}", grid, global_step)
            writer.add_scalar("Train/reconstruction_loss", metric_ae_recloss.compute(), global_step)
            writer.add_scalar("Train/kldivergence_loss", log_dict_ae["train/kl_loss"], global_step)
            writer.add_scalar("Train/discriminator_loss", metric_discloss.compute(), global_step)
            writer.add_scalar("Train/nll_loss", log_dict_ae["train/nll_loss"], global_step)
            writer.add_scalar("Train/d_weight", log_dict_ae["train/d_weight"], global_step)
            writer.add_scalar("Train/g_loss", log_dict_ae["train/g_loss"], global_step)
            writer.add_scalar("Train/og_rec", log_dict_ae["train/og_rec"], global_step)
            writer.add_scalar("Train/total_loss", metric_aeloss.compute(), global_step)
            writer.close()

        if (global_step % options["ckpt_every_n_steps"] == 0 or (local_step >= len(dataloader) - 1)) and local_step > 0 and accelerator.is_main_process:
            accelerator.save_state(save_dir)
            torch.save({"train_sampler": dataloader.sampler.state_dict(),
                             "step": local_step,
                             "epoch": epoch}, os.path.join(save_dir, "train_sampler.bin"))
            
            saver.atomic_symlink(save_dir)

            # print("attempting to save")
            # save_dir_1 = os.path.join(options["ckpt_dir"], f"state_1.pt")
            # save_dir_2 = os.path.join(options["ckpt_dir"], f"state_2.pt")
            # if os.path.isdir(os.path.join(options["ckpt_dir"], "latest.pt")):
            #     old = os.path.realpath(os.path.join(options['ckpt_dir'], 'latest.pt'))
            #     new = save_dir_1 if save_dir_1 != old else save_dir_2
            #     shutil.rmtree(new, ignore_errors=True)
            #     accelerator.save_state(new)
            #     print("done saving state")
            #     torch.save({"train_sampler": dataloader.sampler.state_dict(),
            #                 "step": local_step,
            #                 "epoch": epoch}, os.path.join(new, "train_sampler.bin"))
            #     os.symlink(new, os.path.join(options["ckpt_dir"], "temp.pt"))
            #     os.replace(os.path.join(options["ckpt_dir"], "temp.pt"), os.path.join(options["ckpt_dir"], "latest.pt"))
            # else:
            #     new = os.path.join(options["ckpt_dir"], f"state_1.pt")
            #     shutil.rmtree(new, ignore_errors=True)
            #     accelerator.save_state(new)
            #     print("done saving state")
            #     torch.save({"train_sampler": dataloader.sampler.state_dict(),
            #                 "step": local_step,
            #                 "epoch": epoch}, os.path.join(new, "train_sampler.bin"))
            #     os.symlink(new, os.path.join(options["ckpt_dir"], "latest.pt"))
                
            # if os.path.isdir(os.path.join(options["ckpt_dir"], "latest.pt")):
            #     accelerator.save_state(save_dir)
            #     print("done saving state")
            #     torch.save({"train_sampler": dataloader.sampler.state_dict(),
            #                 "step": global_step,
            #                 "epoch": epoch}, os.path.join(save_dir, "train_sampler.bin"))
            #     os.symlink(save_dir, os.path.join(options["ckpt_dir"], "temp.pt"))
            #     os.replace(os.path.join(options["ckpt_dir"], "temp.pt"), os.path.join(options["ckpt_dir"], "latest.pt"))
            #     print(f"Saved checkpoint to {os.path.realpath(os.path.join(options['ckpt_dir'], 'latest.pt'))}")
            # else:
            #     accelerator.save_state(save_dir)
            #     print("done saving state")
            #     torch.save({"train_sampler": dataloader.sampler.state_dict(),
            #                 "step": global_step,
            #                 "epoch": epoch}, os.path.join(save_dir, "train_sampler.bin"))
            #     os.symlink(save_dir, os.path.join(options["ckpt_dir"], "latest.pt"))
    
        batch_start = time()

        if options["fast_dev_run"]:
            break
        
        global_step += 1
        local_step += 1
    print("returning for some reason")
    return global_step


def validation_epoch(
    options: Dict[str, Any],
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    default_metrics: List[Metric],
    rec_metrics: List[Metric],
    global_step: int = 0,
    postfix: str = "",
):
    """Validate one epoch for the VAE model."""
    for metric in default_metrics:
        metric.reset()

    for _, metric in rec_metrics:
        metric.reset()

    metric_aeloss = default_metrics[0]
    metric_discloss = default_metrics[1]
    metric_ae_recloss = default_metrics[2]
    metric_aeloss.reset()
    metric_discloss.reset()
    metric_ae_recloss.reset()

    model.eval()
    criterion.eval()
    epoch_start = time()
    dtype = get_weight_dtype(accelerator)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = to_dict(batch)

            images = batch["img"].to(dtype)
            reconstructions, posterior = model(images)

            aeloss, log_dict_ae = criterion(
                inputs=images,
                reconstructions=reconstructions,
                posteriors=posterior,
                optimizer_idx=0,
                global_step=global_step,
                weight_dtype=dtype,
                last_layer=accelerator.unwrap_model(model).get_last_layer(),
                split="valid",
            )

            discloss, _log_dict_disc = criterion(
                inputs=images,
                reconstructions=reconstructions,
                posteriors=posterior,
                optimizer_idx=1,
                global_step=global_step,
                weight_dtype=dtype,
                last_layer=accelerator.unwrap_model(model).get_last_layer(),
                split="valid",
            )

            metric_aeloss.update(aeloss)
            metric_ae_recloss.update(log_dict_ae["valid/rec_loss"])
            metric_discloss.update(discloss)
            images, reconstructions = accelerator.gather_for_metrics((images, reconstructions))
            for _, metric in rec_metrics:
                metric.update(images, reconstructions)

            # Logging values
            print(
                f"\r Validation{postfix}: "
                + f"\r[Epoch <{epoch:03}/{options['max_epoch']}>: Step <{i:03}/{len(dataloader)}>] - "
                + f"AE Loss: {aeloss.item():.3f} ({metric_aeloss.compute():.3f}) - "
                + f"AE Rec Loss: {log_dict_ae['valid/rec_loss'].item():.3f} ({metric_ae_recloss.compute():.3f}) - "
                + f"Disc Loss: {discloss.item():.3f} ({metric_discloss.compute():.3f}) - "
                + f"{(((time() - epoch_start) / (i + 1)) * (len(dataloader) - i)) / 60:.2f} m remaining\n"
            )

            if options["fast_dev_run"]:
                break

        if options["is_logging"]:
            log_data = {
                f"valid{postfix}/epoch": epoch,
                f"valid{postfix}/mean_aeloss": metric_aeloss.compute(),
                f"valid{postfix}/mean_ae_recloss": metric_ae_recloss.compute(),
                f"valid{postfix}/mean_discloss": metric_discloss.compute(),
            }
            for name, metric in rec_metrics:
                log_data[f"valid{postfix}/{name}"] = metric.compute()

            accelerator.log(log_data)

    return metric_ae_recloss.compute()
