# Introduction
This demonstration trains a CompRx model on the open-mg dataset. 

Tensorboard logs are saved by default to checkpoints/tb.


# Getting started
## 1. Configure a suitable virtual environment

```bash
python3 -m virtualenv ~/.comprx_venv
source ~/.comprx_venv/bin/activate
```

Install the dependencies with these commands:

```bash
cd ~/isc-demos/stanford/CompRx
pip install -e .
pip install -r requirements.txt
```

## 3. Update the ISC file and config files
Open the ISC file at `~/isc-demos/stanford/CompRx/comprx.isc` to configure the parameters for you job.

This job uses various .yaml files as the config; for this specific job, we will be using the `~/isc-demos/stanford/CompRx/configs/experiment/vae.yaml` file for hyper parameters, as well as `~/isc-demos/stanford/CompRx/configs/train_vae.yaml` file for other options such as checkpoint frequency and location. 

`~/isc-demos/stanford/CompRx/configs/paths/default.yaml` should also be updated to include the correct absolute path in `project_dir`

## 3. Pre-loading and setup

This demo requires some data to be downloaded from the internet that is best done before running on the ISC to save time. To prepare the job before it goes on the ISC, use the following command:

```bash
python3 prep.py
```

## 4. Launch the experiment on the ISC
Launch the experiment on the ISC with the following command.

```bash
isc train comprx.isc
```

## 5. Running tensorboard
Launch tensorboard with the following command.
```bash
tensorboard --host 192.168.127.70 --logdir ./checkpoints/tb
```

On your local machine, navigate to the link it outputs in the terminal.

On your local machine, navigate to the link it outputs in the terminal.

Changes:
(mg_dataset.py)
Class that extends GenericDataset that is used to accommodate for the mg dataset file structure.

(vae_losses.py)
Added extra values to the log so they can be easily added to the tensorboard

(reconstruction_metrics.py)
Specify the cuda device so pre-trained clip model can be loaded onto the correct GPU
Corrected the CLIP text config kwargs keys to the proper so create_model_and_transforms will not raise an exception

(vae.yaml)
Changed to only use a single mg dataset in training
Altered batch size to fit our GPU VRAM (2 samples per GPU)
Altered gradient accumulatino to reach a total batch size of 72 in training
Train dataset shuffle set to false

(paths/default.yaml)
Project dir hardcoded instead of reading an environment variable
Output dir set to ./checkpoints/
Checkpoint path set to ./checkpoints/latest.pt

(requirements.txt)
Inlcuded requirements to run that were not included initially
Added cycling_utils

(train_vae.py)
Manually prepare the dataloaders so that they use the interruptible sampler
Global step calculation based off epoch and local step from checkpoint
Uses the sampler context in the train loop to pass over training step when loading checkpoint during evaluation

(train_components.py)
Tensorboard support
Change dataloader iteration from enumerating the dataloader to loading local step from checkpoint
Atomic saving of checkpoints
