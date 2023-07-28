torchrun --nproc_per_node=6 train.py \
  --data-path /workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/ \
  --model resnet50 -b 360 --output-dir runs/`date +%s`
