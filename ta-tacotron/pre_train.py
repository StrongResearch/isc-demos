import argparse
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
from cycling_utils import InterruptableDistributedSampler, atomic_torch_save
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from datasets import (SpectralNormalization, process_dataset,
                      text_mel_collate_fn)
from parser_utils import parse_args
from text.text_preprocessing import text_to_sequence


def get_dataset(args):
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

    global_rank = dist.get_rank()

    dataset = get_dataset(args)
    dataset_sampler = InterruptableDistributedSampler(dataset)
    output_dir = Path("ljspeech_batches")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path("pre_train_checkpoint.pt")
    
    # record how many times we have cycled
    # used to create unique name for each batch
    num_cycles = 0 

    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path)
        dataset_sampler.load_state_dict(checkpoint["sampler"])
        num_cycles = checkpoint["num_cycles"]
        num_cycles += 1

    loader_params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "prefetch_factor": 1024,
        "persistent_workers": True,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": partial(
            text_mel_collate_fn, n_frames_per_step=args.n_frames_per_step
        ),
    }

    data_loader = DataLoader(dataset, sampler=dataset_sampler, **loader_params)

    for batch_idx, batch in enumerate(data_loader):
        (
            text_padded,
            text_lengths,
            mel_specgram_padded,
            mel_specgram_lengths,
            gate_padded
        ) = batch

        save_path = output_dir / Path(f"b{num_cycles}{global_rank}{batch_idx}.safetensors")
        temp_path = str(save_path) + ".tmp"
        save_file(
            {
                "text_padded":text_padded,                    
                "text_lengths":text_lengths,
                "mel_specgram_padded":mel_specgram_padded,
                "mel_specgram_lengths":mel_specgram_lengths,
                "gate_padded":gate_padded
            },
            temp_path 
        )   
        os.rename(temp_path,save_path)
        
        # must advance the sampler 
        dataset_sampler.advance(len(batch[0]))
        
        if global_rank == 0:
            if batch_idx % 10 == 0:
                atomic_torch_save({"sampler":dataset_sampler.state_dict(), "num_cycles":num_cycles}, checkpoint_path)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Torch Audio Tacotron 2 pre training script"
    )
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    main(args)
