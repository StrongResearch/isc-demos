from pathlib import Path
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import json

def is_master(global_rank):
    return global_rank == 0

def load_checkpoint(*, path, model, optimizer, device):
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    print(f"Loaded checkpoint from {path}. step = {step}")
    return step

def save_checkpoint(*, path, model, optimizer, step):
    print(f"Saving checkpoint to {path}. step = {step}")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(checkpoint, path)

def main(args):
    checkpoint_path = args.checkpoint_path
    max_steps = args.max_steps

    dist.init_process_group("nccl")
    global_rank = dist.get_rank()
    device_id = global_rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    num_nodes = world_size // torch.cuda.device_count()
    num_procs_per_node = torch.cuda.device_count()

    # raise Exception("testing what happens if crashes")

    torch.cuda.set_device(device_id)
    model = torchvision.models.resnet18().to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    step = 0
    print(f"step = {step}")
    if checkpoint_path.exists():
        step = load_checkpoint(path=checkpoint_path, model=model, optimizer=optimizer, device=f'cuda:{device_id}')
        print(f"Rank {global_rank} loaded checkpoint. step = {step}")
    if step >= max_steps:
        print(f"Rank {global_rank} already completed training. step = {step}")
        return

    model = DDP(model, device_ids=[device_id])

    batch_size = 64
    batch = torch.randn(batch_size, 3, 224, 224).cuda(device_id)

    # warmup
    model(batch)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for i in range(step, max_steps):
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        if is_master(global_rank) and i % 10 == 0:
            save_checkpoint(path=checkpoint_path, model=model.module, optimizer=optimizer, step=i)
            print(f"Rank {global_rank} completed step {i}. loss = {loss.item()}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Rank {global_rank} completed training. loss = {loss.item()}")

    if is_master(global_rank):
        save_checkpoint(path=checkpoint_path, model=model.module, optimizer=optimizer, step=max_steps)
        images_seen = max_steps * batch_size * world_size
        images_per_second = images_seen / elapsed_time
        print(f"Training took {elapsed_time:.2f}s. {images_per_second:.2f} images per second")
        print(json.dumps({"GREPME": "grepme", "num_nodes": num_nodes, "num_procs_per_node": num_procs_per_node, "images_per_second": images_per_second, "elapsed_time": elapsed_time, "images_seen": images_seen, "num_steps": max_steps, "world_size": world_size, "device_id": device_id, "global_rank": global_rank}))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    main(args)
