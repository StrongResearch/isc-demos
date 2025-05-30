### INFO: This is a helper script to allow participants to confirm their model is working!
import os
import functools
import logging
import warnings

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

from cycling_utils import AtomicDirectory, atomic_torch_save, TimestampedTimer, InterruptableDistributedSampler
from fsdp_utils import bfSixteen_ready, bfSixteen_policy, count_trainable_parameters, AppState, get_args_parser

from datasets import load_dataset, disable_progress_bars

timer = TimestampedTimer("Start")

# suppressing warnings about missing modules in state_dict
logger = logging.getLogger("torch.distributed.fsdp._state_dict_utils")
logger.setLevel(logging.ERROR)
# suppress warnings about "UserWarning: `_get_pg_default_device` will be deprecated" while saving and loading
warnings.filterwarnings("ignore", category=UserWarning)
# suppress huggingface datasets progress bars
disable_progress_bars()

ADAPTER_NAME = "ExampleLora"
SHARD_STRATEGY = ShardingStrategy.FULL_SHARD

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    rank = int(os.environ["RANK"]) # Global rank
    local_device = int(os.environ["LOCAL_RANK"]) # Rank on local node
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of global ranks
    model_path = os.path.join("/data", args.model_dataset_id)
    torch.cuda.set_device(local_device)

    timer.report(f"Init process group for world size: {world_size}")

    # creating a device mesh enables FSDP to use DTensor instead of ShardedTensor
    device_mesh = init_device_mesh("cuda", (world_size,))
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

        # inject PEFT modules
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, # set to zero to see identical loss on all ranks
    )

    model = LoraModel(model, lora_config, ADAPTER_NAME)

    timer.report(f"PEFT model: {count_trainable_parameters(model)}")

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    # prepare validation dataset and utilities
    validation_data_path = f"/data/{args.dataset_id}/validation-00000-of-00001.parquet"
    dataset = load_dataset("parquet", data_files={"valid": validation_data_path}, split="valid", cache_dir="/tmp/wiki_qa")

    def preprocess_function(examples):
        # Combine question and answer into a single text
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        
        # Tokenize with padding and truncation
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels for causal language modeling (shift input_ids right)
        encodings["labels"] = encodings["input_ids"].copy()
        
        return encodings

    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function, batched=True,
        remove_columns=dataset.column_names, desc="Tokenizing and preprocessing train dataset"
    )

    # Create dataloader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in batch])
        }

    validation_sampler = InterruptableDistributedSampler(tokenized_dataset)
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=validation_sampler)

    # load checkpoint if found
    try:
        output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    except KeyError as error:
        print("Must set env var CHECKPOINT_ARTIFACT_PATH so we know where to save checkpoints!")
        exit(1)
    saver = AtomicDirectory(output_directory=output_directory, is_master=rank==0)

    # performance metric to evaluate
    validation_loss = 0.0

    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        state_dict = { "app": AppState(model, optimizer)}
        dcp.load(state_dict=state_dict, checkpoint_id=latest_checkpoint_path)
        validation_state = torch.load(os.path.join(latest_checkpoint_path, "validation_state.pt"))
        dataloader.sampler.load_state_dict(validation_state["sampler"])
        validation_loss = validation_state["validation_loss"]
        timer.report("Loaded checkpoint")

    state_dict = { "app": AppState(model, optimizer) }

    save_every_steps = 20

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            step = dataloader.sampler.progress // dataloader.batch_size
            is_last_step = (step + 1) == len(dataloader)
            is_save_step = ((step + 1) % save_every_steps == 0) or is_last_step

            # Move batch to device
            input_ids = batch["input_ids"].to(torch.cuda.current_device())
            attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
            labels = batch["labels"].to(torch.cuda.current_device())

            # forward, backward, update
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            dataloader.sampler.advance(len(input_ids))

            # synchronize loss for reporting
            sync_loss = loss
            dist.all_reduce(sync_loss)
            validation_loss += sync_loss.item()

            timer.report(f"Validation step [{step} / {len(dataloader)}]")

            if is_save_step:
                checkpoint_directory = saver.prepare_checkpoint_directory()
                checkpoint_writer = dcp.FileSystemWriter(checkpoint_directory)

                if rank == 0:
                    atomic_torch_save(
                        {
                            "sampler": dataloader.sampler.state_dict(),
                            "validation_loss" : validation_loss
                        }, 
                        os.path.join(checkpoint_directory, "validation_state.pt")
                    )

                saver.symlink_latest(checkpoint_directory)

                timer.report("Saved checkpoint")

            if is_last_step:
                timer.report(f"Validation average loss: [{validation_loss / len(dataloader):,.3f}]")