import os
import functools
import logging
import warnings

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig

from datasets import load_dataset

from cycling_utils import atomic_torch_save, AtomicDirectory, TimestampedTimer, InterruptableDistributedSampler
from fsdp_utils import bfSixteen_ready, bfSixteen_policy, count_trainable_parameters, AppState, get_args_parser

timer = TimestampedTimer("Start")

# suppressing warnings about missing modules in state_dict
logger = logging.getLogger("torch.distributed.fsdp._state_dict_utils")
logger.setLevel(logging.ERROR)
# suppress warnings about "UserWarning: `_get_pg_default_device` will be deprecated" while saving and loading
warnings.filterwarnings("ignore", category=UserWarning)

ADAPTER_NAME = "ExampleLora"
SHARD_STRATEGY = ShardingStrategy.FULL_SHARD

'''
For Hybrid shard we need to:
1. Initialize the device_mesh as an (NNODES, NPROC) array.
2. Set SHARD_STRATEGY = ShardingStrategy.HYBRID_SHARD
3. Enumerate processes within one model shard group to participate in saving - HOW??
4. Gate saving on being member of that group
5. Pass the saving process group to the dcp.save function
'''

print("Finished imports")

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    rank = int(os.environ["RANK"]) # Global rank
    local_device = int(os.environ["LOCAL_RANK"]) # Rank on local node
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of global ranks
    model_path = os.path.join("/data", args.dataset_id)
    torch.cuda.set_device(local_device)

    timer.report(f"Init process group for world size: {world_size}")

    # creating a device mesh enables FSDP to use DTensor instead of ShardedTensor
    device_mesh = init_device_mesh("cuda", (world_size,))
    saving_group = device_mesh.get_group() # modify this for HYBRID_SHARD
    assert bfSixteen_ready(), "ERROR: System not BF16 ready."

    # pre-trained model weights should be mounted at /data
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # save CPU RAM by loading non-main rank models to 'meta' device
    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            use_cache=False, 
            torch_dtype=torch.bfloat16
        )
        print(f"Main rank {rank} model params on device: {set([p.data.device for p in model.parameters()])}")
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                use_cache=False, 
                torch_dtype=torch.bfloat16
            )
            print(f"Non-main rank {rank} model params on device: {set([p.data.device for p in model.parameters()])}")

    timer.report(f"Loaded model: {count_trainable_parameters(model)}")

    # # inject PEFT modules
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_dropout=0, # set to zero to see identical loss on all ranks
    # )

    # model = LoraModel(model, lora_config, ADAPTER_NAME)

    # timer.report(f"PEFT model: {count_trainable_parameters(model)}")

    # wrap model in FSDP
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000
    )

    model = FSDP(model, 
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=SHARD_STRATEGY,
        mixed_precision=bfSixteen_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda mod: mod.to_empty(device=torch.cuda.current_device(), recurse=False), # for init from 'meta' device
        sync_module_states=True, # broadcast model weights from main rank
        device_mesh=device_mesh
    )

    timer.report("FSDP wrapped model and broadcast to GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    batch_size = 4
    sequence_length = 384

    # prepare dataset and utilities
    dataset = load_dataset("arrow", 
                        data_files=f"/data/82b2949e-1167-42df-a654-ea74a6877a53/filtered_reasoning_deepseek-train.arrow",  
                        split="train").select(range(90000))

    def preprocess_function(examples):
        # Access the messages and answers directly from the batch
        texts = [f"Question: {msg[0]['content']}\nAnswer: {ans}" 
                for msg, ans in zip(examples['messages'], examples['answer'])]
        
        # Tokenize with padding and truncation
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=sequence_length,  # Using the specified sequence_length
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels for causal language modeling
        encodings["labels"] = encodings["input_ids"].copy()
        
        return encodings

    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing and preprocessing dataset",
    )

    if rank == 0:
        print(f"Processed dataset with {len(tokenized_dataset)} examples")
        print(f"Sequence length: {sequence_length}")
        
    # Create dataloader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in batch])
        }

    train_sampler = InterruptableDistributedSampler(tokenized_dataset)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Effective batch size: {batch_size} (across {world_size} GPUs)")
    # load checkpoint if found
    saver = AtomicDirectory(output_directory=args.chk_path, is_master=rank==0)
    latest_sym = os.path.join(args.chk_path, saver.symlink_name)
    if os.path.exists(latest_sym):
        latest_path = os.readlink(latest_sym)
        state_dict = { "app": AppState(model, optimizer)}
        dcp.load(state_dict=state_dict, checkpoint_id=latest_path)

        train_state = torch.load(os.path.join(latest_path, "train_state.pt"))
        dataloader.sampler.load_state_dict(train_state["sampler"])

        timer.report("Loaded checkpoint")

    # training
    num_epochs = 5
    save_every = 20
    model.train()

    for epoch in range(dataloader.sampler.epoch, num_epochs):

        dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):

            is_save_step = (step + 1) % save_every == 0
            if is_save_step:
                checkpoint_directory = saver.prepare_checkpoint_directory()

                timer.report("Prepared checkpoint directory")

            timer.report("Moving batch to device!")
            # Move batch to device
            input_ids = batch["input_ids"].to(torch.cuda.current_device())
            attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
            labels = batch["labels"].to(torch.cuda.current_device())

            timer.report("Calculating loss!")
            # forward, backward, update
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            timer.report("Stepping optimizer!")
            optimizer.step()

            timer.report("Advancing dataloader!")
            dataloader.sampler.advance(len(input_ids))

            timer.report("Zeroing optimizer gradient!")
            optimizer.zero_grad()

            timer.report(f"Step {step} Loss: {loss.item()}")

            if is_save_step:
                timer.report(f"=== SAVING CHECKPOINT! ===")
                state_dict = { "app": AppState(model, optimizer) }

                timer.report(f"Grabbed state dict. Saving model state with dcp.save...")
                dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_directory, process_group=saving_group)

                timer.report(f"Called! Saving dataloader sampler with pt using torch.save...")
                torch.save({
                    "sampler": dataloader.sampler.state_dict()
                }, os.path.join(checkpoint_directory, "train_state.pt"))

                timer.report(f"Saved! Symlinking checkpoint directory...")

                saver.atomic_symlink(checkpoint_directory)
                timer.report(f"Symlink success!")

                timer.report(f"=== Saved checkpoint ===")
                dist.barrier()

        timer.report(f"=== RESETTING DATALOADER PROGRESS! ===")
        dataloader.sampler.reset_progress()

    timer.report("Done.")

    dist.barrier()
    dist.destroy_process_group()
