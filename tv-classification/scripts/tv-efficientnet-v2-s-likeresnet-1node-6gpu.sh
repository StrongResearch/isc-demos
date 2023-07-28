# this script copies tv-resnet50-1node-6gpu but trains an efficientnet_v2_s model
# hopefully this simple script works better than the more complicated tv-efficientnet-v2-s-1node-6gpu.sh
torchrun --nproc_per_node=6 train.py \
  --data-path /workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/ \
  --model efficientnet_v2_s -b 42 --output-dir runs/`date +%s`