# ISC Demos

Demo training runs for the ISC

## Quickstart

Firstly, set up `~/credentials.isc` with the following contents:

```toml
username="YOUR_USERNAME"
api_key="YOUR_API_KEY"
```

Then run `isc ping` to check that everything is working.

You can now run the following commands to setup the demo environment
and train various models.

```bash
# install demos
cd ~
python3 -m virtualenv ~/.venv
source ~/.venv/bin/activate

git clone https://github.com/StrongResearch/isc-demos.git
cd ~/isc-demos
pip install -r requirements.txt
pip install -e cycling_utils # supports suspending and resuming a distributed sampler
# Note: cycling_utils will soon be moved to a separate repository.

# fashion mnist
cd ~/isc-demos/fashion_mnist
isc train fashion_mnist.isc
```

You can also use the following commands to view the status
of your experiments and clusters.

```bash
isc experiments # view a list of your experiments
isc clusters # view the status of the clusters
```

## Example Configurations

The following examples demonstrate how to implement interruptibility in
distributed training scripts using checkpointing, atomic saving, and
stateful samplers. Tools for these purposes are provided in the `cycling_utils`
package.

These examples are being actively developed to achieve (1) interruptibility
in distributed training, (2) verified completion of a full training run, and
(3) achievement of benchmark performance published by others (where applicable). 
Each example published below is annotated with its degree of completion. Examples
annotated with [0] are "coming soon".

### Hello World

- [cifar100-resnet50.isc](./cifar100-resnet50/cifar100-resnet50.isc) [3]
- [fashion_mnist.isc](./fashion_mnist/fashion_mnist.isc) [3]
- WIP [dist_model_parallel.isc](./dist_model_parallel.isc) [0]

### pytorch-image-models (timm)

(from https://github.com/huggingface/pytorch-image-models)

- [resnet50.isc](./pytorch-image-models/resnet50.isc) [2]
- [resnet152.isc](./pytorch-image-models/resnet152.isc) [2]
- [efficientnet_b0.isc](./pytorch-image-models/efficientnet_b0.isc) [2]
- [efficientnet_b7.isc](./pytorch-image-models/efficientnet_b7.isc) [2]
- [efficientnetv2_s.isc](./pytorch-image-models/efficientnetv2_s.isc) [2]
- WIP [efficientnetv2_xl.isc](./pytorch-image-models/efficientnetv2_xl.isc) [2]
- [vit_base_patch16_224.isc](./pytorch-image-models/vit_base_patch16_224.isc) [2]
- WIP [vit_large_patch16_224.isc](./pytorch-image-models/vit_large_patch16_224.isc) [2]

### tv-segmentation

(from https://github.com/pytorch/vision/tree/main/references/segmentation)

- WIP [fcn_resnet101.isc](./tv-segmentation/fcn_resnet101.isc) [1]
- WIP [deeplabv3_mobilenet_v3_large.isc](./tv-segmentation/deeplabv3_mobilenet_v3_large.isc) [1]

### tv-detection

(from https://github.com/pytorch/vision/tree/main/references/detection)

- WIP [maskrcnn_resnet50_fpn.isc](./tv-detection/fasterrcnn_resnet50_fpn.isc) [0]
- WIP [retinanet_resnet50_fpn.isc](./tv-detection/retinanet_resnet50_fpn.isc) [0]

## Detectron2

(from https://github.com/facebookresearch/detectron2)

- WIP [detectron2.isc](./detectron2.isc) [0]
- WIP [detectron2_densepose.isc](./detectron2_densepose.isc) [0]

## Large Language Models

- WIP [llama2.isc](./llama2.isc) [0]
- WIP [mistral.isc](./mistral.isc) [0]
