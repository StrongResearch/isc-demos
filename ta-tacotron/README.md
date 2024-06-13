This is an example pipeline for text-to-speech using Tacotron2 configured for the ISC.


## Install required packages

Assuming you are in a virtualenv:

```bash
pip install -r requirements.txt
```
This installs all extra requirements for Tacotron2.

## Training Tacotron2 on the isc with character as input

First you will need to run a job to preprocess the raw training data.

```bash
isc train pre-train-ta-tacotron.isc
```
This should generate a cached version of the training set which has been pre-processed.


Then you can use the generated pre processed data
```bash 
isc train ta-tacotron.isc
```

## Changes to tacotron

Several differences to the original training script

- Dataset is pre-processed and saved, batched, in safetensors format
- Main training loop loads batched safetensors and trains on the batches

