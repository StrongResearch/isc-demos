import torch
import torch.distributed as dist
import time

def main():

    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = dist.get_rank()  # Rank of this GPU in cluster
    device_id = rank % torch.cuda.device_count()  # Rank on local node
    is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(device_id)  # Enables calling 'cuda'


    time.sleep(70)
    if rank == 0:
        raise torch.cuda.OutOfMemoryError
    
    while True:
        pass

if __name__ == "__main__":
    main()

