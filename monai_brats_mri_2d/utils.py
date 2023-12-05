import torch, os, errno
import torch.distributed as dist
from torch.utils import data
from h5py import File as h5pyFile

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, section, transform=None):
        super().__init__()
        self.file_path = file_path
        self.section = section
        self.transform = transform

        with h5pyFile(self.file_path, 'r') as hf:
            self.len = hf[self.section].shape[0]
            
    def __getitem__(self, index):

        with h5pyFile(self.file_path, 'r') as hf:
            x = hf[self.section][index]

        x = torch.from_numpy(x)

        if self.transform:
            x = self.transform(x)
            
        return torch.unsqueeze(x, 0)

    def __len__(self):
        return self.len
    

class Tensor3dDataset(data.Dataset):
    def __init__(self, file_path, section, transform=None):
        super().__init__()
        self.file_path = file_path
        self.section = section
        self.transform = transform

        with h5pyFile(self.file_path, 'r') as hf:
            self.scans, self.channels, self.dx, self.dy, self.slices = hf[self.section].shape
            self.len = int(self.scans * self.slices)
            
    def __getitem__(self, index):

        scan = index // self.slices
        slc = index - (scan * self.slices)
        
        with h5pyFile(self.file_path, 'r') as hf:
            x = hf[self.section][scan, :, :, :, slc]

        x = torch.from_numpy(x)

        if self.transform:
            x = self.transform(x)
            
        return x

    def __len__(self):
        return self.len