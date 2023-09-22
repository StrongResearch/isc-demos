# Semantic segmentation reference training scripts

This folder contains reference training scripts for semantic segmentation.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

You must ensure all dependencies in "requirements.txt" are installed, and
run "prep.py" to download pretrained model weights before launching your
training job.

You can run the training routines for the following models using cli.

### RetinaNet
```
isc train ./retinanet_resnet101_fpn.isc
```

### Mask R-CNN
```
isc train ./maskrcnn_resnet101_fpn.isc
```
