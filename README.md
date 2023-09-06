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

## Officially Validated Configurations

### Hello World

- [cifar100-resnet50.isc](./cifar100-resnet50/cifar100-resnet50.isc)
- [fashion_mnist.isc](./fashion_mnist/fashion_mnist.isc)

### pytorch-image-models

(from https://github.com/huggingface/pytorch-image-models)

- [resnet50.isc](./pytorch-image-models/resnet50.isc)
- [resnet152.isc](./pytorch-image-models/resnet152.isc)
- [efficientnet_b0.isc](./pytorch-image-models/efficientnet_b0.isc)
- [efficientnet_b7.isc](./pytorch-image-models/efficientnet_b7.isc)
- [efficientnetv2_s.isc](./pytorch-image-models/efficientnetv2_s.isc)
- WIP [efficientnetv2_xl.isc](./pytorch-image-models/efficientnetv2_xl.isc)
- [vit_base_patch16_224.isc](./pytorch-image-models/vit_base_patch16_224.isc)
- WIP [vit_large_patch16_224.isc](./pytorch-image-models/vit_large_patch16_224.isc)

### tv-segmentation

(from https://github.com/pytorch/vision/tree/main/references/segmentation)

- [fcn_resnet101.isc](./tv-segmentation/fcn_resnet101.isc)
- [deeplabv3_mobilenet_v3_large.isc](./tv-segmentation/deeplabv3_mobilenet_v3_large.isc)

### tv-detection

(from https://github.com/pytorch/vision/tree/main/references/detection)

- [maskrcnn_resnet50_fpn.isc](./tv-detection/fasterrcnn_resnet50_fpn.isc)
- [retinanet_resnet50_fpn.isc](./tv-detection/retinanet_resnet50_fpn.isc)
