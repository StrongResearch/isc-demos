# Introduction
This demonstration trains a BLIP model on the COCO dataset. The demonstration aims to mimic the same configuration used for pre-training in the 

Tensorboard logs are saved by default to `./LAVIS/logs`.


# Getting started
## 1. Configure a suitable virtual environment

```bash
python3 -m virtualenv ~/.blip_venv
source ~/.blip_venv/bin/activate
```

Install the dependencies with these commands:

```bash
cd ~/isc-demos/stanford/LAVIS
pip install -e .
```

## 2. Pre-loading and setup

This demo requires some data to be downloaded from the internet that is best done before running on the ISC to save time. To prepare the job before it goes on the ISC, use the following command:

```bash
./prep.sh
```

Due to the short time a job may have on the cluster, it is recommended that the SPICE evaluation is removed from the evaluation step. To do this, navigate the the following file:
```
~/isc-demos/stanford/LAVIS/.venv/lib/python3.10/site-packages/pycocoevalcap/eval.py
```

and comment out line 45. This change should be reverted if you expect to have more time for your job.


## 3. Update the ISC file
Open the ISC file at `~/isc-demos/stanford/LAVIS/blip.isc` to configure the parameters for you job.
This job uses various .yaml files as the config; for this specific job, we will be using the `~/isc-demos/stanford/LAVIS/lavis/config/tasks/blip/train/pretrain_14m.yaml`

## 4. Launch the experiment on the ISC
Launch the experiment on the ISC with the following command.

```bash
isc train blip.isc
```

## 5. Running tensorboard
Launch tensorboard with the following command.
```bash
tensorboard --host 192.168.127.70 --logdir ./logs
```

On your local machine, navigate to the link it outputs in the terminal.

(runner_base.py)
Added eval_freq argument
added tensorboard and a tensorboard path argument in caption_coco_ft.yaml
Loads checkpoint from the path specified in the yaml
changed distributedsampler to InterruptableDistributedSampler
Changed the train/eval loop to use the interruptabledistributedsampler context
Iteration is saved in the checkpoint and resumed from in the train loop
Pass args + iters + tb writer into train function
pass writer to eval
torch.save changed to atomic_torch_save

(base_task.py)
changed the loop to use the checkpointed iteration count for when the loop has been interrupted mid-epoch
Checkpoint frequency is set in the config (checkpoint_freq)
tensorboard metrics are recorded (loss and learning rate)

(captioning.py)
added a name variable to captioning.py
added tensorboard support for writing evaluation metrics
removed the spice evaluation since it takes too long

(logger.py)
Added ability to start from an index in log_every()

(dist_utils.py)
Allow print from other ranks 

updated requirements to include specific torch version + compatible transformers version, tensorboard and jdk

added a prep.sh script that should be run before training to pre-load files that would waste time on the isc

added/updated various paths (cache path, checkpoint path)