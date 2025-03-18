
# Initialize WandB for logging (only on rank 0)
#   if wandb_apikey:
import os
import torch
import wandb
from datetime import datetime

def init_wandb_logging(batch_size, seq_len, device_mesh_1d, world_size, save_every):
   sc_name = os.getenv("STRONG_EXPERIMENT_NAME", "ds-hacking")
   being_tested = os.getenv("BEING_TESTED", None)
   
   # Injects relevant test params into wandb name for ease of identification
   wandb_name = "{}{}.{}.{}:{}:{}".format(
         f"{being_tested}:" if being_tested else "",
         world_size,
         batch_size,
         seq_len,
         save_every,
         sc_name
   )

   # wandb_name = f"{being_tested}:{world_size}.{args.batch_size}.{args.grad_accum_steps}:{args.sharding_strategy}:{sc_name}"
   wandb.init(
         project="fsdp-ds-reasoning",
         # dir=exp_dir,
         name=wandb_name,
         id=os.getenv("STRONG_EXPERIMENT_ID", str(datetime.now().timestamp())),
         save_code=True,
         config={
            # "args": vars(args),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device_mesh": device_mesh_1d,
            "world_size": world_size,
         },
   )


# Function to get detailed memory stats
def get_detailed_memory_stats():
    """Get detailed memory statistics for all GPUs"""
    stats = {}
    for i in range(torch.cuda.device_count()):
        stats[f"gpu{i}_mem_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
        stats[f"gpu{i}_mem_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9
        stats[f"gpu{i}_max_mem_allocated_gb"] = torch.cuda.max_memory_allocated(i) / 1e9
        stats[f"gpu{i}_max_mem_reserved_gb"] = torch.cuda.max_memory_reserved(i) / 1e9
    return stats

def get_mem_stats(device=None):
    """Get memory statistics for the given device."""
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }
