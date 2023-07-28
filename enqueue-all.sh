set -e
make do_tiny_training
make do_tv-classification
make do_tv-segmentation
make do_tv-detection
make do_nerf
make do_bert-qa