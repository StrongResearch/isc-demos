#!/bin/bash
set -e

echo "Launching DeepSpeed on node rank $NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

## Dynamically generate a hostfile for deepspeed
# the hostname is not important because we're 
# launching without passwordless SSH torchrun style
# but the file need to have as many hosts listed
# as there are nodes in the cluster and each
# host must have a number of GPUs specified

# Ensure directory exists
mkdir -p /tmp/deepspeed
# define path to hostfile
HOSTFILE=/tmp/deepspeed/hostfile
# Empty or create the file
: > "$HOSTFILE"
# Generate node entries
for i in $(seq 1 "$NNODES"); do
    echo "node$i slots=$N_PROC" >> "$HOSTFILE"
done

## Launch training with deepspeed
deepspeed --hostfile /tmp/deepspeed/hostfile --no_ssh --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    /root/isc-demos/deepspeed/train.py \
    --model_path /data/uds-brazen-meowing-munchkin-250513 \
    --data_path /data/uds-visual-water-soup-250513 \
    --output_dir $CHECKPOINT_ARTIFACT_PATH \
    --deepspeed --deepspeed_config /root/isc-demos/deepspeed/ds_config.json