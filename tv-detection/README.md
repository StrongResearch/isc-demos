# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must also run "prep.py" to download pretrained model weights before 
launching your training job.

You can then run the training routines for the following models using cli.

### RetinaNet
```
isc train ./retinanet_resnet101_fpn.isc
```

### Mask R-CNN
```
isc train ./maskrcnn_resnet101_fpn.isc
```