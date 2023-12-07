# Introduction

This demonstration features the fairseq library, with a focus on the HuBERT model. Fairseq has checkpointing inbuilt, so integrating with the ISC will be very simple.

The Fairseq GitHub repository by Meta Research was cloned on 1 December 2023 for the purpose of developing this demonstration, and is not updated with any subsequent changes to the llama-recipes GitHub repository.

# Getting started

## 1. Configure a suitable virtual environment

For the purposes of this demo, we will be using virtualenv as our pip virtual environment module.

```bash
python3 -m virtualenv ~/.fairseq
source ~/.fairseq/bin/activate
```

Install the required version of Pytorch with the following command.

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Navigate into the fairseq repository and install fairseq and soundfile.

```bash
cd ~/isc-demos/fairseq
pip install -e .
pip install soundfile==0.12.1
```

## 2. Pre-process the dataset

The LibriSpeech (960h) dataset has been downloaded and setup using the following steps:

NOTE: The directory paths used here exist publicly, so change the paths if you're creating a new one.

```bash
cd /mnt/.node1/Open-Datasets/librispeech

wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/train-other-500.tar.gz

tar -xvf train-clean-100.tar.gz
tar -xvf train-clean-360.tar.gz
tar -xvf train-other-500.tar.gz

mkdir /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/train-960

cd LibriSpeech

mv train-clean-100/* train-960
mv train-clean-360/* train-960
mv train-other-500/* train-960
```

The following command is used to generate the train and validation splits.

```bash
cd ~/isc-demos/fairseq

python3 examples/wav2vec/wav2vec_manifest.py /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/train-960 --dest /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/train-960 --ext flac --valid-percent 0.01
```

### Stage 1 Pre-Training

To generate the train and validation features for the first HuBERT pre-training iteration, use the following command:

```bash
cd ~/isc-demos/fairseq/examples/hubert/simple_kmeans/
python dump_mfcc_feature.py train 1 0 /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/features
python dump_mfcc_feature.py valid 1 0 /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/features
```

The next step is to fit the k-means model with 100 clusters on the training features:

```bash
python3 learn_kmeans.py /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/features train 1 /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/kmeans/kmeans_model.pt 100 --percent -1
```

To generate the train and validation labels with the kmeans model execute this command:

```bash
python dump_km_label.py /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/features/ train /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/kmeans/kmeans_model.pt 1 0 /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/labels
python dump_km_label.py /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/features/ valid /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/kmeans/kmeans_model.pt 1 0 /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/labels
```

Change the name of the generated labels:

```bash
mv train_0_1.km train.km
mv valid_0_1.km valid.km
```

The final step is to create a dummy dictionary:

```bash
for x in $(seq 0 $((100 - 1))); do
  echo "$x 1"
done >> /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/labels/dict.km.txt
```

### Stage 2 Pre-Training

After the first pre-training stage has complete, run the following command to extract the new features using the 6th transformer layer of the pre-trained model.

```bash
python3 dump_hubert_feature.py /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/train-960 train /mnt/.node1/Open-Datasets/librispeech/checkpoints/checkpoint_last.pt 6 1 0 /mnt/.node1/Open-Datasets/librispeech/LibriSpeech/stage_2_features
```

After this, repeat the previous steps to train the kmeans model on these features and extract the labels.

Ensure that you use 500 clusters instead of 100 clusters.


## 4. Creating an ISC file
There is an example ISC file at `~/isc-demos/fairseq/hubert_cycling.isc`

To modify the entry point, change the command to your specificatinos

## 5. Launch the experiment on the ISC
Launch the experiment on the ISC with the following command.

```bash
cd ~/isc-demos/fairseq
isc train hubert_cycling.isc
```
