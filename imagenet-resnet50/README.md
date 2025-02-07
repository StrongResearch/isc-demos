# ImageNet x ResNet50 Speed Run
## Quickstart
If you've just generated a new Freedom Container in [Control Plane](https://cp.strongcompute.ai/), start by installing python, git, and nano.
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```
Clone this repo if you haven't already.
```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
```
Create and source a virtual environment called `~/.imagenet`.
```bash
python3 -m virtualenv ~/.imagenet
source ~/.imagenet/bin/activate
```
Install dependencies.
```bash
cd ~/isc-demos/imagenet-resnet50
pip install -r requirements.txt
```
Update the experiment launch file with your `isc_project_id`.
```bash
nano resnet50_bench.isc
```
Launch your experiment.
```bash
isc train resnet50_bench.isc
```
