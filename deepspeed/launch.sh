#!/bin/bash
set -e

# Load environment variables for multi-node config
# source host_settings.env

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

# deepspeed \
#   --no_local_rank \
#   --no_ssh_check \
#   --num_gpus $N_PROC \
#   --num_nodes $NNODES \
#   --node_rank $NODE_RANK \
#   --master_addr $MASTER_ADDR \
#   --master_port $MASTER_PORT \
#   train.py \
#   --deepspeed ds_config.json \
#   --model_path /data/uds-warp-pattern-lifter-250527 \
#   --output_dir $CHECKPOINT_ARTIFACT_PATH \
#   "$@"

# Verify required environment variables
: "${N_PROC:?Need N_PROC}"
: "${NNODES:?Need NNODES}"
: "${NODE_RANK:?Need NODE_RANK}"
: "${MASTER_ADDR:?Need MASTER_ADDR}"
: "${MASTER_PORT:?Need MASTER_PORT}"

WORLD_SIZE=$(( NNODES * N_PROC ))
export WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK

echo "[Run Node] NODE_RANK=$NODE_RANK"
echo "[Run Node] NNODES=$NNODES"
echo "[Run Node] N_PROC=$N_PROC"
echo "[Run Node] WORLD_SIZE=$WORLD_SIZE"
echo "[Run Node] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Launch 1 DeepSpeed process per GPU (torchrun-style)
for LOCAL_RANK in $(seq 0 $((N_PROC - 1))); do
    RANK=$(( NODE_RANK * N_PROC + LOCAL_RANK ))

    echo "Launching RANK=$RANK LOCAL_RANK=$LOCAL_RANK"

    WORLD_SIZE=$WORLD_SIZE \
    RANK=$RANK \
    LOCAL_RANK=$LOCAL_RANK \
    NODE_RANK=$NODE_RANK \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
        deepspeed --no_local_rank --no_ssh_check /root/isc-demos/deepspeed/train.py \
          --deepspeed /root/isc-demos/deepspeed/ds_config.json \
          --model_path /data/uds-warp-pattern-lifter-250527 \
          --output_dir $CHECKPOINT_ARTIFACT_PATH \
          &
done

# Keep node alive until all GPU processes finish
wait
