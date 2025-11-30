#!/bin/bash
set -e

# Load environment variables for multi-node config
source host_settings.env

# The scheduler must set:
#   NODE_RANK (0-based)
#   LOCAL_WORLD_SIZE (num GPUs per node)
#   NNODES (total nodes)

# Example:
#   export NODE_RANK=0
#   export LOCAL_WORLD_SIZE=8
#   export NNODES=2

echo "Launching DeepSpeed on node rank $NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

deepspeed \
  --no_local_rank \
  --num_gpus $LOCAL_WORLD_SIZE \
  --num_nodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  train.py \
  --deepspeed ds_config.json \
  --model_path /data/model-weights \
  --output_dir ./checkpoints \
  "$@"
