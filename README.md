# ISC Demos
Welcome to the Strong Compute Instant Super Computer (ISC) Demos repo. Before diving into these demos, it is recommended 
that Strong Compute users complete the Getting Started section of the 
[Developer Docs](https://strong-compute.gitbook.io/developer-docs/getting-started).

### Recent Updates
Please note: This repo has recently been updated to reflect the imminent changes to the Artifacts system in the Strong Compute ISC. These changes are scheduled for general release on Tuesday, 18 March 2025 at 09:30:00 pm PST.

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

#### pytorch-image-models (timm)

(from https://github.com/huggingface/pytorch-image-models)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| resnet50 | Image classification | ResNet50 | [3] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| resnet152 | Image classification | ResNet152 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnet_b0 | Image classification | EfficientNet B0 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnet_b7 | Image classification | EfficientNet B7 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnetv2_s | Image classification | EfficientNetV2 S | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| efficientnetv2_xl | Image classification | EfficientNetV2 XL | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| vit_base_patch16_224 | Image classification | VIT Base Patch16 224 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |
| vit_large_patch16_224 | Image classification | VIT Large Patch16 224 | [2] | [isc-demos/pytorch-image-models](pytorch-image-models) |

#### Torchvision segmentation

(from https://github.com/pytorch/vision/tree/main/references/segmentation)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| fcn_resnet101 | Image segmentation | ResNet101 | [2] | [isc-demos/tv-segmentation](tv-segmentation) |
| deeplabv3_mobilenet_v3_large | Image segmentation | MobileNetV3 Large | [2] | [isc-demos/tv-segmentation](tv-segmentation) |

#### Torchvision detection

(from https://github.com/pytorch/vision/tree/main/references/detection)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| maskrcnn_resnet101_fpn | Object detection | Mask RCNN (ResNet101 FPN) | [2] | [isc-demos/tv-detection](tv-detection) |
| retinanet_resnet101_fpn | Object detection | RetinaNet (ResNet101 FPN) | [2] | [isc-demos/tv-detection](tv-detection) |

#### Detectron2

(from https://github.com/facebookresearch/detectron2)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| detectron2 | TBC | Detectron2 | [2] | [isc-demos/detectron2](detectron2) |
| detectron2_densepose | TBC | Detectron2 | [2] | [isc-demos/detectron2/projects/densepose](detectron2/projects/densepose) |

#### Large Language Models (LLM)

| Title | Description | Model | Status | Link |
| :--- | :--- | :--- | :----: | :--- |
| DeepSeek | Language Modelling | DeepSeek-R1 | [2] | [isc-demos/deepseek](deepseek) |
| Llama2 | LoRA | Llama2 | [0] | [isc-demos/llama2](llama2) |
