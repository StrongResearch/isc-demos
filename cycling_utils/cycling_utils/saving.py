from pathlib import Path
import os
import torch
import time

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
    
class Timer:
    def __init__(self, start_time=None, running=0):
        self.start_time = start_time if start_time is not None else time.time()
        self.running = running
    def report(self, annot):
        now = time.time()
        duration = now - self.start_time
        self.running += duration
        print("Completed {:<70}{:>12} milliseconds, {:>12} seconds total".format(annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
        self.start_time = now