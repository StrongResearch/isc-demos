# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
# import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
# from torch.distributed.fsdp import StateDictType
# from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
# from tqdm import tqdm
# from transformers import LlamaTokenizer


# from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen, bfSixteen, get_llama_wrapper
# from llama_recipes.utils.memory_utils import MemoryTrace

## ADDED
import json
import shutil
import uuid
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)
from torch.utils.tensorboard import SummaryWriter

## -- DISUSED -- ##
# def set_tokenizer_params(tokenizer: LlamaTokenizer):
#     tokenizer.pad_token_id = 0
#     tokenizer.padding_side = "left"

## -- DISUSED -- ##
# # Converting Bytes to Megabytes
# def byte2mb(x):
#     return int(x / 2**20)

## -- ADDED -- ##
def check_peft_save_successful(temp_path):
    try:
        adapter_config_path = os.path.join(temp_path, "adapter_config.json")
        _ = json.loads(open(adapter_config_path).read())
        return True
    except:
        return False
    
## -- ADDED -- ##
def save_checkpoint(save_path, model, optimizer, scaler, train_dataloader, eval_dataloader, lr_scheduler, metrics, timer, saver):

    timer.report("started checkpoint saving")
    peft_path = os.path.join(save_path, "peft")
    peft_model_saved = False
    while peft_model_saved == False:
        dist.barrier()
        # Remove any pre-existing peft_path
        if int(os.environ["RANK"]) == 0:
            if os.path.isdir(peft_path):
                shutil.rmtree(peft_path)
        dist.barrier()
        timer.report("prepared peft_path")

        # Attempt to save the model
        model.save_pretrained(peft_path)
        dist.barrier()
        timer.report("saved peft model")

        # check to confirm the peft model has saved successfully
        peft_model_saved = check_peft_save_successful(peft_path)
        dist.barrier()
        timer.report("checked peft model saved")

    full_osd = FSDP.full_optim_state_dict(model, optimizer)
    if int(os.environ["RANK"]) == 0:
        torch.save(full_osd, os.path.join(save_path, "optimizer.pt"))
    timer.report("saved optimizer")

    if int(os.environ["RANK"]) == 0:
        torch.save({
            "scaler": scaler.state_dict(),
            "train_sampler": train_dataloader.sampler.state_dict(),
            "eval_sampler": eval_dataloader.sampler.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "metrics": metrics,
        }, os.path.join(save_path, "other_checkpoint.pt"))
        timer.report("saved other checkpoint")

        saver.atomic_symlink(save_path)
        timer.report("re-assigned symlink, done.")

    dist.barrier()


def train(epoch, model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, scaler, metrics, kwargs, timer, gradient_accumulation_steps, train_config, saver):

    ## -- MOVED OUTSIDE TRAINING LOOP -- ##
    # # Create a gradient scaler for fp16
    # if train_config.use_fp16 and train_config.enable_fsdp:
    #     scaler = ShardedGradScaler()
    # elif train_config.use_fp16 and not train_config.enable_fsdp:
    #     scaler = torch.cuda.amp.GradScaler()

    ## -- MOVED OUTSIDE TRAINING LOOP -- ##
    # for epoch in range(train_config.num_epochs):
    #     epoch_start_time = time.perf_counter()

    model.train()
    writer = SummaryWriter(log_dir=os.path.join(kwargs["dist_checkpoint_root_folder"], "tb"))
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    step = train_dataloader.sampler.progress // train_dataloader.batch_size
    total_steps = len(train_dataloader) // gradient_accumulation_steps

    timer.report(f"training epoch {epoch + 1} start from step {step + 1}")

    for batch in train_dataloader:

        # Prep save path first for robustness under time constraint
        save_path = saver.prepare_checkpoint_directory()

        timer.report(f"training epoch {epoch + 1} prepped save path")

        for key in batch.keys():
            if train_config.enable_fsdp:
                batch[key] = batch[key].to(int(os.environ["LOCAL_RANK"]))
            else:
                batch[key] = batch[key].to('cuda:0')
        
        timer.report(f"batch {step + 1} data to device")

        with autocast():
            loss = model(**batch).loss
        loss = loss / gradient_accumulation_steps

        timer.report(f"batch {step + 1} rank 0 loss: {loss.item():.4f}")
        
        if train_config.use_fp16:
            # if fp16 is enabled, use gradient scaler to handle gradient update
            scaler.scale(loss).backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            # regular backpropagation when fp16 is not used
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        timer.report(f"batch {step + 1} backward")

        metrics["train"].update({"processed": train_dataloader.batch_size, "loss": loss.detach().float()})
        metrics["train"].reduce()
        batch_loss = metrics["train"].local["loss"] / metrics["train"].local["processed"]
        timer.report(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step + 1}/{len(train_dataloader)} batch (loss: {batch_loss:.4f})")
        metrics["train"].reset_local()

        # Advance sampler and measure progress
        train_dataloader.sampler.advance(train_dataloader.batch_size)
        step = train_dataloader.sampler.progress // train_dataloader.batch_size

        if int(os.environ["RANK"]) == 0:
            total_progress = epoch * total_steps + step
            writer.add_scalar('Train/Loss', batch_loss, total_progress)

        if step == total_steps:
            lr_scheduler.step()
            metrics["train"].end_epoch()

        save_checkpoint(save_path, model, optimizer, scaler, train_dataloader, eval_dataloader, lr_scheduler, metrics, timer, saver)


def evaluation(epoch, model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, scaler, metrics, kwargs, timer, train_config, saver):

    model.eval()
    writer = SummaryWriter(log_dir=os.path.join(kwargs["dist_checkpoint_root_folder"], "tb"))

    step = eval_dataloader.sampler.progress // eval_dataloader.batch_size
    total_steps = len(eval_dataloader)

    timer.report(f"evaluation epoch {epoch + 1} start from step {step + 1}")

    for batch in eval_dataloader:

        # Prep save path first for robustness under time constraint
        save_path = saver.prepare_checkpoint_directory()

        for key in batch.keys():
            if train_config.enable_fsdp:
                batch[key] = batch[key].to(int(os.environ["LOCAL_RANK"]))
            else:
                batch[key] = batch[key].to('cuda:0')

        timer.report(f"batch {step + 1} data to device")

        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            outputs = model(**batch)
            loss = outputs.loss
        
        timer.report(f"batch {step + 1} loss")

        metrics["eval"].update({"processed": eval_dataloader.batch_size, "loss": loss.detach().float()})
        metrics["eval"].reduce()

        # Advance sampler and measure progress
        eval_dataloader.sampler.advance(eval_dataloader.batch_size)
        step = eval_dataloader.sampler.progress // eval_dataloader.batch_size

        if step == total_steps:
            epoch_loss = metrics["eval"].local["loss"] / metrics["eval"].local["processed"]
            timer.report(f"Evaluation Epoch: {epoch+1}/{train_config.num_epochs}, epoch loss: {epoch_loss:.4f}")
            metrics["eval"].end_epoch()

            if int(os.environ["RANK"]) == 0:
                writer.add_scalar('Eval/Loss', epoch_loss, epoch)

        save_checkpoint(save_path, model, optimizer, scaler, train_dataloader, eval_dataloader, lr_scheduler, metrics, timer, saver)


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")
