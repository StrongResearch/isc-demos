# ImageNet x ResNet50 Speed Run
## Step 1a: `isc-demos` image container
If you have created your container based on the `isc-demos` Image in [Control Plane](https://cp.strongcompute.ai/), 
then your container already has all necessary dependencies installed including a python virtual environment with 
necessary python dependencies at `/opt/venv`. Please clone this `isc-demos` repository and then skip to Step 2.

```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
```

## Step 1b: other image container
If you've just generated a new container in [Control Plane](https://cp.strongcompute.ai/), start by installing python, git, and nano.
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```
Clone this repo if you haven't already.
```bash
cd ~
git clone https://github.com/StrongResearch/isc-demos.git
```
Create and source a virtual environment.
```bash
python3 -m virtualenv /opt/venv
source /opt/venv/bin/activate
```
Install dependencies.
```bash
cd ~/isc-demos/imagenet-resnet50
pip install -r requirements.txt
```

## Step 2: prepare and launch experiment
Update the experiment launch file with your `isc_project_id`.
```bash
nano resnet50_bench.isc
```
Launch your experiment.
```bash
isc train resnet50_bench.isc
```
