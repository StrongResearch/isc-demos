project/
├── train.py                # Main training script
├── ds_config.json          # ZeRO-3 DeepSpeed config
├── launch_node.sh          # Per-node launch wrapper (called by scheduler)
├── host_settings.env       # Environment variables for master_addr, port, etc.
└── checkpoints/            # Output directory

# lauch with this on all nodes

`bash launch.sh`

# deepspeed vs torchrun

The important line in `launch.sh` is --no_local_rank, which puts DeepSpeed into torchrun-mode (no SSH launching).
Otherwise deepspeed expects the main rank to be able to ssh into other hosts nominated in a hostfile in 
passwordless manner with `ssh hostname` to be able to setup distributed training.

# About DeepSpeed Distributed Checkpointing

With zero_optimization.stage = 3, model states are sharded across processes.

DeepSpeed's save_checkpoint():

- Saves each shard per-rank under:

- checkpoints/step_500/global_step500/partition_<rank>.pt

- Automatically stores:
- - LoRA adapter weights (trainable params)
- - Optimizer state shards
- - Scheduler state
- - RNG states
- - ZeRO-3 partition metadata

This means saving is distributed across all GPUs, maximizing speed.
Loading is also distributed.

Your extra metadata is saved separately in training_state.pt.

# resume training with 

`bash launch_node.sh --resume_tag step_X`

Where X is the index of the checkpoint to resume from i.e. 0, 1, 2, ...