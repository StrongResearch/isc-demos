set -e

MODEL=efficientnet_v2_s
TRAIN_SIZE=300
EVAL_SIZE=384


torchrun --nproc_per_node=6 train.py \
  --model $MODEL -b 41 --lr 0.245 --lr-scheduler cosineannealinglr \
  --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 \
  --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --weight-decay 0.00002 --norm-weight-decay 0.0 \
  --train-crop-size $TRAIN_SIZE --model-ema --val-crop-size $EVAL_SIZE --val-resize-size $EVAL_SIZE \
  --ra-sampler --ra-reps 4 --output-dir runs/`date +%s` \
  --data-path /workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/ 
