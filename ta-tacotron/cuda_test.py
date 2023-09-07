from datetime import datetime

print("Start: {}".format(str(datetime.now())))

import os


import matplotlib.pyplot as plt
from torch import no_grad, tile, manual_seed, load

import torch.distributed as dist
import torch.cuda as cuda


def train(global_rank, world_size):
    count = cuda.device_count()
    local_rank = global_rank % count
    print(f"cuda vars: {local_rank}, {global_rank}, {world_size}")
    cuda.set_device(local_rank)


def main():
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    train(global_rank, world_size)
    

if __name__ == "__main__":
    main() 
