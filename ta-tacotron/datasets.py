# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
"""
Modified from
https://github.com/pytorch/audio/blob/main/examples/pipeline_tacotron2/datasets.py
for Strong Compute ISC

Changes: 
- Dataset no longer is cached to memory
- Users are expected to use this code to cache the processed dataset
  so this is not done at startup
"""

from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.utils.data.dataset import random_split
from torchaudio.datasets import LJSPEECH


class SpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.log(torch.clamp(input, min=1e-5))


class InverseSpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.exp(input)


class Processed(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, text_preprocessor):
        self.dataset = dataset
        self.transforms = transforms
        self.text_preprocessor = text_preprocessor

    def __getitem__(self, key):
        item = self.dataset[key]
        return self.process_datapoint(item)

    def __len__(self):
        return len(self.dataset)

    def process_datapoint(self, item):
        melspec = self.transforms(item[0])
        text_norm = torch.IntTensor(self.text_preprocessor(item[2]))
        return text_norm, torch.squeeze(melspec, 0)


def process_dataset(
    dataset: str,
    file_path: str,
    transforms: Callable,
    text_preprocessor: Callable[[str], List[int]],
) -> torch.utils.data.Dataset:
    """Returns the processed dataset"""
    if dataset == "ljspeech":
        data = LJSPEECH(root=file_path, download=False)
    else:
        raise ValueError(f"Expected datasets: `ljspeech`, but found {dataset}")

    dataset = Processed(data, transforms, text_preprocessor)

    return dataset


def text_mel_collate_fn(
    batch: Tuple[Tensor, Tensor], n_frames_per_step: int = 1
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The collate function padding and adjusting the data based on `n_frames_per_step`.
    Modified from https://github.com/NVIDIA/DeepLearningExamples

    Args:
        batch (tuple of two tensors): the first tensor is the mel spectrogram with shape
            (n_batch, n_mels, n_frames), the second tensor is the text with shape (n_batch, ).
        n_frames_per_step (int, optional): The number of frames to advance every step.

    Returns:
        text_padded (Tensor): The input text to Tacotron2 with shape (n_batch, max of ``text_lengths``).
        text_lengths (Tensor): The length of each text with shape (n_batch).
        mel_specgram_padded (Tensor): The target mel spectrogram
            with shape (n_batch, n_mels, max of ``mel_specgram_lengths``)
        mel_specgram_lengths (Tensor): The length of each mel spectrogram with shape (n_batch).
        gate_padded (Tensor): The ground truth gate output
            with shape (n_batch, max of ``mel_specgram_lengths``)
    """
    text_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
    )
    max_input_len = text_lengths[0]

    text_padded = torch.zeros((len(batch), max_input_len), dtype=torch.int64)
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]][0]
        text_padded[i, : text.size(0)] = text

    # Right zero-pad mel-spec
    num_mels = batch[0][1].size(0)
    max_target_len = max([x[1].size(1) for x in batch])
    if max_target_len % n_frames_per_step != 0:
        max_target_len += n_frames_per_step - max_target_len % n_frames_per_step
        assert max_target_len % n_frames_per_step == 0

    # include mel padded and gate padded
    mel_specgram_padded = torch.zeros(
        (len(batch), num_mels, max_target_len), dtype=torch.float32
    )
    gate_padded = torch.zeros((len(batch), max_target_len), dtype=torch.float32)
    mel_specgram_lengths = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        mel = batch[ids_sorted_decreasing[i]][1]
        mel_specgram_padded[i, :, : mel.size(1)] = mel
        mel_specgram_lengths[i] = mel.size(1)
        gate_padded[i, mel.size(1) - 1 :] = 1

    return (
        text_padded,
        text_lengths,
        mel_specgram_padded,
        mel_specgram_lengths,
        gate_padded,
    )
