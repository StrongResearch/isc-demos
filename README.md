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

# fashion mnist
cd ~/isc-demos/fashion_mnist
isc train fashion_mnist.isc

# cifar100 + ResNet50
cd ~/isc-demos/cifar100-resnet50
isc train cifar100-resnet50.isc

# torchvision segmentation
cd ~/isc-demos/tv-segmentation
isc train config.isc
```

You can also use the following commands to view the status
of your experiments and clusters.

```bash
isc experiments # view a list of your experiments
isc clusters # view the status of the clusters
```