from cycling_utils import TimestampedTimer

timer = TimestampedTimer()

from cycling_utils import InterruptableDistributedSampler

import os
from pkg_resources import packaging
from contextlib import nullcontext

import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset

from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.inference.model_utils import load_peft_model

from llama_recipes.utils.train_utils import (
    # train,
    train_epoch,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

def print_rank_0(s):
    if int(os.environ["RANK"]) == 0:
        print(s)

def est_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_gb = (param_size + buffer_size) / (1024**3)
    return size_all_gb

import torch.distributed as dist

import warnings
warnings.filterwarnings("ignore")

timer.report("imports done")

def main(**kwargs):

    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    print_rank_0(F"\nPRINT train_config: \n{train_config}\n")
    print_rank_0(F"\nPRINT fsdp_config: \n{fsdp_config}\n")

    timer.report("obtained configs")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    timer.report("set random seed")

    if train_config.enable_fsdp:
        print_rank_0("enable_fsdp TRUE")
        # setup() # Runs dist.init_process_group('nccl')
        dist.init_process_group("nccl")
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        timer.report(f"local rank {local_rank}, rank {rank}, world size {world_size}")

    if torch.distributed.is_initialized():
        print_rank_0("torch.distributed.is_initialized TRUE")
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
        timer.report("set device, clear gpu cache, setup environ flags")

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    print_rank_0(f"use_cache: {use_cache}")
    print_rank_0(f"model path: {train_config.model_name}")
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        print_rank_0("train_config.enable_fsdp and train_config.low_cpu_fsdp TRUE")
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        print_rank_0(f"verify_latest_nightly: {verify_latest_nightly}")
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            print("Hello from rank 0")
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
            timer.report("constructed model (on cpu)")
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        print_rank_0("train_config.enable_fsdp and train_config.low_cpu_fsdp FALSE")
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            use_flash_attention_2=True ## ADDED
        )
        timer.report("constructed model (on cpu)")

    print_rank_0('\n')
    print_rank_0(model) # gosh isn't it simple...
    print_rank_0('\n')
    model_size_gb = est_model_size(model)
    print(f"Rank {rank} model size {model_size_gb:,.2f} GB")
    print(f"Rank {rank} torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/(1024**3):,.2f} GB")

    # if train_config.enable_fsdp and train_config.use_fast_kernels:
    #     """
    #     For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
    #     using of Flash Attention or Xformer memory-efficient kernels
    #     based on the hardware being used. This would speed up fine-tuning.
    #     """
    #     try:
    #         from optimum.bettertransformer import BetterTransformer
    #         model = BetterTransformer.transform(model)
    #     except ImportError:
    #         print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # # Prepare the model for int8 training if quantization is enabled
    # if train_config.quantization:
    #     print_rank_0("quantization TRUE")
    #     model = prepare_model_for_int8_training(model)
    #     timer.report("set quantization")
# ______
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        print_rank_0("train_config.enable_fsdp and fsdp_config.pure_bf16 TRUE")
        model.to(torch.bfloat16)
        timer.report("model.to(torch.bfloat16)")
#_______

    ## HERE IF WE'RE RESUMING WE NEED TO LOAD THE PREVIOUSLY SAVED PEFT MODULES
    
    dist.barrier()
    if train_config.use_peft:
        resume_path = os.path.join(kwargs["resume"], "resume")
        if os.path.isdir(resume_path):
            print_rank_0("RESUMING SAVED PEFT MODULES")
            model = load_peft_model(model, resume_path, trainable=True)
            if rank == 0:
                model.print_trainable_parameters()
            # if train_config.enable_fsdp and fsdp_config.pure_bf16:
            #     model.to(torch.bfloat16) # to coerce peft loaded weights to bf16?
        else:
            print_rank_0("BUILDING NEW PEFT MODULES")
            peft_config = generate_peft_config(train_config, kwargs)
            print_rank_0("use_peft TRUE:")
            model = get_peft_model(model, peft_config)
            if rank == 0:
                model.print_trainable_parameters()
        timer.report("peft_config, peft_model")

    ## AND NOW WE CAN CONTINUE FROM HERE

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        timer.report("mixed precision")
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        timer.report(f"my_auto_wrapping_policy, fsdp_config.sharding_strategy: {fsdp_config.sharding_strategy}")

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        timer.report(f"FSDP(model), fsdp_cpu_offload: {fsdp_config.fsdp_cpu_offload}")
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)

    # elif not train_config.quantization and not train_config.enable_fsdp:
    #     print_rank_0("not train_config.quantization and not train_config.enable_fsdp")
    #     model.to("cuda")
    #     timer.report("model.to(cuda)")

    print_rank_0(f"model on device")
    print(f"Rank {rank} torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/(1024**3):,.2f} GB")
    model_size_gb = est_model_size(model)
    print(f"Rank {rank} model size {model_size_gb:,.2f} GB")

    dataset_config = generate_dataset_config(train_config, kwargs)
    timer.report("dataset_config")

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer, dataset_config, split="train",
    )
    dataset_val = get_preprocessed_dataset(
        tokenizer, dataset_config, split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
        timer.report(f"--> Training Set Length = {len(dataset_train)}, Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        print_rank_0("Preprocessing dataset_train with ConcatDataset...")
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)
        timer.report("dataset_train batching_strategy == packing")

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
    train_sampler = InterruptableDistributedSampler(dataset_train)
    train_dl_kwargs["sampler"] = train_sampler
    timer.report("train_dl_kwargs")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    timer.report("train_dataloader")

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            print_rank_0("Preprocessing dataset_val with ConcatDataset...")
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
        val_sampler = InterruptableDistributedSampler(dataset_val)
        val_dl_kwargs["sampler"] = val_sampler        

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        timer.report("eval_dataloader")

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        print_rank_0("fsdp_config.pure_bf16 and fsdp_config.optimizer == 'anyprecision'")
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        print_rank_0("fsdp_config.pure_bf16 and fsdp_config.optimizer == 'anyprecision' FALSE")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    timer.report("optimizer")

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    timer.report("scheduler")

    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    ## LOAD SAVED OPTIMIZER AND SCHEDULER IF SAVED PREVIOUSLY
    checkpoint_path = os.path.join(kwargs["resume"], "resume", "checkpoint.pt")
    if os.path.isfile(checkpoint_path):

        print_rank_0("RESUMING SAVED OPTIMIZER AND SCHEDULER")
        checkpoint = torch.load(checkpoint_path)
        sharded_osd = FSDP.scatter_full_optim_state_dict(checkpoint["optimizer"], model)
        optimizer.load_state_dict(sharded_osd)

        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["eval_sampler"])
        scaler.load_state_dict(checkpoint["scaler"])

    timer.report("retrieved checkpoint (if there)")

    # Start the training process
    for epoch in range(train_config.num_epochs):
        with train_sampler.in_epoch(epoch):
            print_rank_0(f"TRAINING EPOCH {epoch}")
            train_epoch(
                model, 
                optimizer, 
                train_dataloader, 
                eval_dataloader, 
                train_config, 
                scaler, 
                autocast, 
                timer, 
                local_rank, 
                rank, 
                kwargs
            )

            with val_sampler.in_epoch(epoch):
                print("EVAL GOES HERE")
                pass

if __name__ == "__main__":
    fire.Fire(main)