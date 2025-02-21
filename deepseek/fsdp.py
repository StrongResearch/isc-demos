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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig

from cycling_utils import AtomicDirectory, TimestampedTimer
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # load checkpoint if found
    saver = AtomicDirectory(output_directory=args.chk_path, is_master=rank==0)
    latest_sym = os.path.join(args.chk_path, saver.symlink_name)
    if os.path.exists(latest_sym):
        latest_path = os.readlink(latest_sym)
        state_dict = { "app": AppState(model, optimizer)}
        dcp.load(state_dict=state_dict, checkpoint_id=latest_path)

        timer.report("Loaded checkpoint")

    # simulate a batch of data
    batch = ["Hello how are you?", "I'm good thanks, and you?", "Great!"]
    encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

    # train in autoregressive mode (next-token prediction)
    input_ids = encoding['input_ids'][:,:-1]
    attention_mask = encoding['attention_mask'][:,:-1]
    labels = encoding['input_ids'][:,1:]

    input_ids = input_ids.to(torch.cuda.current_device())
    attention_mask = attention_mask.to(torch.cuda.current_device())
    labels = labels.to(torch.cuda.current_device())

    # simulate training
    for step in range(100):
        is_save_step = (step + 1) % 5 == 0
        if is_save_step:
            checkpoint_directory = saver.prepare_checkpoint_directory()

            timer.report("Prepared checkpoint directory")

        # forward, backward, update
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        timer.report(f"Step {step} Loss: {loss.item()}")

        if is_save_step:
            state_dict = { "app": AppState(model, optimizer) }
            dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_directory, process_group=saving_group)
            saver.atomic_symlink(checkpoint_directory)

            timer.report("Saved checkpoint")

    timer.report("Done.")

    dist.barrier()
    dist.destroy_process_group()
  
