from pathlib import Path
import os
import torch

def atomic_torch_save(obj, f: str | Path, timer=None, **kwargs):
    f = str(f)
    temp_f = f + ".temp"
    torch.save(obj, temp_f, **kwargs)
    if timer is not None:
        timer.report(f'saving temp checkpoint')
    os.replace(temp_f, f)
    if timer is not None:
        timer.report(f'replacing temp checkpoint with checkpoint')
        return timer
    else:
        return