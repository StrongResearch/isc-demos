# project inventory
```
isc-demos/deepspeed/
├── requirements.txt        # Python project dependencies
├── experiment.isc          # Strong Compute experiment launch file
├── launch.sh               # Per-node launch wrapper
├── train.py                # Main training script
└── ds_config.json          # ZeRO-3 DeepSpeed config
```

# quickstart on Strong Compute
1. Create and start a new container based on the following image.
```
nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
```

2. Inside the container install python, git, and nano.
```
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
```

3. Clone the isc-demos repo and install the `deepspeed` project dependencies to a virtual environment
```
cd ~
python3 -m virtualenv /opt/venv
source /opt/venv/bin/activate
git clone https://github.com/StrongResearch/isc-demos.git
cd ~/isc-demos/deepspeed
pip install -r requirements.txt
```

4. Update the project experiment launch file with your project ID.
```
cd ~/isc-demos/deepspeed
nano experiment.isc
```

5. Launch your experiment
```
cd ~/isc-demos/deepspeed
isc train experiment.isc
```

# Notes on the project setup
## What does this project demonstrate?
Using DeepSpeed to perform LoRA fine-tuning of an LLM. DeepSpeed is configured to
apply Zero3 for reduced GPU memory overhead in distributed training.

## DeepSpeed vs torchrun
DeepSpeed impliments much of the functionality of torchrun, while also taking care
of most of the hassle of sharded training state management.

DeepSpeed is used in place of torchrun to launch the distributed process group, for
details refer to `launch.sh`. DeepSpeed ordinarily wants the user to start the init
process on a single machine, from which DeepSpeed automatically connects via 
passwordless ssh to the other cluster nodes and starts processes there.

Strong Compute clusters do not support passwordless ssh, so we start DeepSpeed
torchrun-style by dynamically generating a `hostfile` and calling `deepspeed` with
`--no_ssh`.

## Model and data
This demo uses the `Llama-3.2-1B-Instruct` model and `Microsoft Research WikiQA Corpus` 
dataset, both from HuggingFace via the Strong Compute Datasets facility.

This model is selected for rapid demonstration purposes, much larger models can be
accomodated by larger training clusters.
