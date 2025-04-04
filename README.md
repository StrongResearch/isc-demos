# ISC Demos
Welcome to the Strong Compute Instant Super Computer (ISC) Demos repo. 

Before diving into these demos, it is recommended that Strong Compute users complete the Getting Started section of the 
[Developer Docs](https://strong-compute.gitbook.io/developer-docs/getting-started).

### Recent Updates

Please note: Some old unmaintained demos have recently been deleted. 

### Demos <a name="more-examples"></a>

The following examples demonstrate use of the ISC for training a variety of models, including how to implement 
interruptibility in distributed training scripts using checkpointing, atomic saving, and stateful samplers.

These examples are being actively developed to achieve [1] interruptibility in distributed training, [2] verified 
completion of a full training run, and [3] achievement of benchmark performance published by others (where applicable). 
Each example published below is annotated with its degree of completion. Examples annotated with [0] are "coming soon".

#### Hello World

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| Fashion MNIST | Image classification | CNN | [3] | [isc-demos/fashion_mnist](fashion_mnist) |
| ImageNet | Image classification | ResNet50 | [2] | [isc-demos/imagenet-resnet50](imagenet-resnet50) |
| DeepSeek | Language Modelling | DeepSeek-R1 | [2] | [isc-demos/deepseek](deepseek) |


#### Torchvision segmentation

(from https://github.com/pytorch/vision/tree/main/references/segmentation)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| fcn_resnet101 | Image segmentation | ResNet101 | [2] | [isc-demos/tv-segmentation](tv-segmentation) |
| deeplabv3_mobilenet_v3_large | Image segmentation | MobileNetV3 Large | [2] | [isc-demos/tv-segmentation](tv-segmentation) |

#### Mask RCNN

(from https://github.com/pytorch/vision/tree/main/references/detection)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| maskrcnn_resnet101_fpn | Object detection | Mask RCNN (ResNet101 FPN) | [2] | [isc-demos/maskrcnn](maskrcnn) |


#### DeepSeek

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| DeepSeek | Language Modelling | DeepSeek-R1 | [2] | [isc-demos/deepseek](deepseek) |
