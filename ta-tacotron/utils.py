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

# Modified to work with isc job cycling
# 2023, Strong Compute

import logging
import os
import shutil
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from datasets import SpectralNormalization, process_dataset, text_mel_collate_fn
from text.text_preprocessing import available_phonemizers, available_symbol_set, get_symbol_list, text_to_sequence
from functools import partial 

import torchaudio

def get_datasets(args):
    text_preprocessor = partial(
        text_to_sequence,
        symbol_list=args.text_preprocessor,
        phonemizer=args.phonemizer,
        checkpoint=args.phonemizer_checkpoint,
        cmudict_root=args.cmudict_root,
    )

    transforms = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            f_min=args.mel_fmin,
            f_max=args.mel_fmax,
            n_mels=args.n_mels,
            mel_scale="slaney",
            normalized=False,
            power=1,
            norm="slaney",
        ),
        SpectralNormalization(),
    )
    trainset = process_dataset(
        args.dataset, args.dataset_path, args.val_ratio, transforms, text_preprocessor
    )
    return trainset, []


def pad_sequences(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    r"""Right zero-pad all one-hot text sequences to max input length.

    Modified from https://github.com/NVIDIA/DeepLearningExamples.
    """
    input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x) for x in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, : text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts: List[str], text_processor: Callable[[str], List[int]]) -> Tuple[Tensor, Tensor]:
    d = []
    for text in texts:
        d.append(torch.IntTensor(text_processor(text)[:]))

    text_padded, input_lengths = pad_sequences(d)
    return text_padded, input_lengths
