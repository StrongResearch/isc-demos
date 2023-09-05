import torch.distributed as dist

from datasets import SpectralNormalization, process_dataset
from text.text_preprocessing import available_phonemizers, available_symbol_set, text_to_sequence

from functools import partial

import torchaudio 
import torch 
import argparse

from safetensors.torch import save_file
from itertools import islice

from pathlib import Path 
import os 

def parse_args(parser):
    """Parse commandline arguments."""

    parser.add_argument(
        "--dataset", default="ljspeech", choices=["ljspeech"], type=str, help="select dataset to train with"
    )
    parser.add_argument("--dataset-path", type=str, default="./", help="path to dataset")

    preprocessor = parser.add_argument_group("text preprocessor setup")
    preprocessor.add_argument(
        "--text-preprocessor",
        default="english_characters",
        type=str,
        choices=available_symbol_set,
        help="select text preprocessor to use.",
    )
    preprocessor.add_argument(
        "--phonemizer",
        type=str,
        choices=available_phonemizers,
        help='select phonemizer to use, only used when text-preprocessor is "english_phonemes"',
    )
    preprocessor.add_argument(
        "--phonemizer-checkpoint",
        type=str,
        help="the path or name of the checkpoint for the phonemizer, "
        'only used when text-preprocessor is "english_phonemes"',
    )
    
    preprocessor.add_argument(
        "--cmudict-root", default="./", type=str, help="the root directory for storing cmudictionary files"
    )

    audio = parser.add_argument_group("audio parameters")
    audio.add_argument("--sample-rate", default=22050, type=int, help="Sampling rate")
    audio.add_argument("--n-fft", default=1024, type=int, help="Filter length for STFT")
    audio.add_argument("--hop-length", default=256, type=int, help="Hop (stride) length")
    audio.add_argument("--win-length", default=1024, type=int, help="Window length")
    audio.add_argument("--n-mels", default=80, type=int, help="")
    audio.add_argument("--mel-fmin", default=0.0, type=float, help="Minimum mel frequency")
    audio.add_argument("--mel-fmax", default=8000.0, type=float, help="Maximum mel frequency")
    return parser


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
    dataset = process_dataset(
        args.dataset, args.dataset_path, transforms, text_preprocessor
    )
    return dataset


def main(args):
    torch.manual_seed(0)

    dist.init_process_group(backend="nccl") 

    torch.manual_seed(0)
    
    world_size = dist.get_world_size()     
    global_rank = dist.get_rank()

    output_dir = Path("~/ljspeech_tensors")

    trainset = get_datasets(args)
    length = len(trainset)
    num_processed = 0 

    for k, (text_norm, melspec) in islice(enumerate(trainset), global_rank, None, world_size):
        num_processed += 1 

        if num_processed % 10 == 0:
            
            with open("~pre_train_checkpoint.tmp", "w") as f:
                f.write(num_processed)
            os.rename("~pre_train_checkpoint.tmp", "~pre_train_checkpoint.txt")


        if global_rank == 0:
            print(k)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch Audio Tacotron 2 pre training script")
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    main(args)
