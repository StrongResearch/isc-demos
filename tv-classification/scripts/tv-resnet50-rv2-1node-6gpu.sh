# https://github.com/pytorch/vision/issues/3995#issuecomment-1259420720 adapted for 1 node 6 gpu
torchrun --nproc_per_node=6 train.py --model resnet50 --batch-size 170 --lr 0.5 \
    --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
    --auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
    --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
    --train-crop-size 176 --model-ema --val-resize-size 232 \
    --ra-sampler --ra-reps=4 \
    --data-path /workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/ \
    --output-dir runs/`date +%s`