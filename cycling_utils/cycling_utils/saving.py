from pathlib import Path
import os
import torch

def atomic_torch_save(obj, f: str | Path, **kwargs):
    f = str(f)
    temp_f = f + ".temp"
    torch.save(obj, temp_f, **kwargs)
    os.replace(temp_f, f)