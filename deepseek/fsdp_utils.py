import os
import argparse
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
from pathlib import Path
from looseversion import LooseVersion

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--chk-path", type=str, default=os.environ.get("CHECKPOINT_ARTIFACT_PATH"))
    parser.add_argument("--dataset-id", help="Dataset ID for the dataset", type=Path, required=True)
    return parser

def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:,.3f}%)"

class AppState(Stateful):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.state_dict_options = StateDictOptions(ignore_frozen_params=True, strict=False)

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, 
            self.optimizer, 
            options=self.state_dict_options
        )

        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        outcome = set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=self.state_dict_options
        )

def bfSixteen_ready():
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )
    return bf16_ready

bfSixteen_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)
