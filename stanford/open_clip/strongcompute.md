# Introduction
This demonstration trains a CLIP model with (by default) ViT-L-14 tranformer architecture on the COCO dataset. 

Tensorboard logs are saved by default to open_clip/src/logs/tensorboard.


# Getting started
## 1. Configure a suitable virtual environment

```bash
python3 -m virtualenv ~/.clip_venv
source ~/.clip_venv/bin/activate
```

Install the dependencies with these commands:

```bash
cd ~/isc-demos/stanford/open_clip
pip install -e .
make install-training
make install-test
```

## 2. Update the ISC file
Navigate to the 'src' directory

```bash
cd ~/isc-demos/stanford/open_clip/src
```
Open the ISC file at `~/isc-demos/stanford/open_clip/src/clip.isc` and configure the parameters for your job. Parameters specific to this job (python args) go on the 'command' line, a full list of these can be found in the `./training/params.py` file.

Note that path to the --resume directory must exist before training.

## 3. Launch the experiment on the ISC
Launch the experiment on the ISC with the following command.

```bash
isc train clip.isc
```

### Changes:
Updated requirements to include more specific torch version and additional requirement (braceexpand)

#### (data.py)

Updated the csvdataset class to include funcitonality to choose how many samples you want to use from the dataset.

Added captiondataset class specifically for the COCO dataset with values hardcoded to work with our install of COCO (set --train-data '/mnt/.node1/Open-Datasets/coco' as a param and it will use the 2014 coco dataset)

Altered the DataInfo class to include the cycling_utils InterruptableDistributedSampler 

Added get_coco_dataset function for use with inbuilt method so it is called when dataset type is set to auto and the path contains "coco" string

#### (main.py)

Changed tensorboard path so on cycling, the same path is used.

Updated existing checkpoint loaded functionality to include additional info such as iteration and sampler state

Updated the train loop to use the sampler context for handling inner-epoch progress

#### (params.py)

Changed some default values (logging, checkpointing)

Checkpointing now refers to iterations/steps instead of epochs

//Added distributed-evaluation

#### (train.py)

Implemented a checkpoint saving function inside train.py for inner-epoch saving

Updated the loop and items inside loop to use the loaded iters variable so loading inner-epoch checkpoints is seamless

Writes loss and learning rate data to tensorboard log directory

//Evaluation is now distributed if distributed-evaluation is true in args