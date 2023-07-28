from pathlib import Path
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import json

def is_master(global_rank):
    return global_rank == 0


def main():
    dist.init_process_group("nccl")
    global_rank = dist.get_rank()
    device_id = global_rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    num_nodes = world_size // torch.cuda.device_count()
    num_procs_per_node = torch.cuda.device_count()

    # raise Exception("testing what happens if crashes")

    torch.cuda.set_device(device_id)
    model = torchvision.models.resnet18(pretrained=True).to(device_id)
    model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch_size = 64
    batch = torch.randn(batch_size, 3, 224, 224).cuda(device_id)

    unix_time_string = str(int(time.time()))
    checkpoint_folder = Path("checkpoints") / unix_time_string
    checkpoint_folder.mkdir(exist_ok=True, parents=True)

    # warmup
    model(batch)
    torch.cuda.synchronize()

    num_steps = 400
    start_time = time.perf_counter()
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        if is_master(global_rank) and i % 10 == 0:
            torch.save(model.state_dict(), checkpoint_folder / f"model_{i}.pth")
            print(f"Rank {global_rank} completed step {i}. loss = {loss.item()}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Rank {global_rank} completed training. loss = {loss.item()}")
    if is_master(global_rank):
        torch.save(model.state_dict(), checkpoint_folder / f"model_final.pth")
        images_seen = num_steps * batch_size * world_size
        images_per_second = images_seen / elapsed_time
        print(f"Training took {elapsed_time:.2f}s. {images_per_second:.2f} images per second")
        print(json.dumps({"GREPME": "grepme", "num_nodes": num_nodes, "num_procs_per_node": num_procs_per_node, "images_per_second": images_per_second, "elapsed_time": elapsed_time, "images_seen": images_seen, "num_steps": num_steps, "world_size": world_size, "device_id": device_id, "global_rank": global_rank}))

if __name__ == '__main__':
    main()
