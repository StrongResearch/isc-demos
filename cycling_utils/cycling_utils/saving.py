from pathlib import Path
import os
import torch
import torch.distributed as dist
from collections import defaultdict

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

class MetricsTracker:
    '''
    This is a general purpose MetricsTracker to assist with recording metrics from
    a disributed cluster.

    The MetricsTracker is initialised without any prior knowledge of the metrics
    to be tracked.

    >>> metrics = MetricsTracker()

    Metrics can be accumulated as required, for example after each batch is procesed
    by the model, by passing a dictionary with metrics to be updated, then reduced 
    accross all nodes. Metric values are stored in a defaultdict.

    >>> preds = model(input)
    >>> loss = loss_fn(preds, targs)
    >>> metrics.update({"images_seen": len(images), "loss": loss.item()})
    >>> metrics.reduce()

    Metrics are assumed to be summable scalar values. After calling reduce(), the 
    metrics.local object contains the sum of corresponding metrics from all nodes
    which can be used for intermediate reporting or logging.

    >>> writer = SummaryWriter()
    >>> for metric,val in metrics.local.items():
    >>>     writer.add_scalar(metric, val, step)
    >>> writer.flush()
    >>> writer.close()

    Once all processing of the current batch has been completed, the MetricsTracker
    can be prepared for the next batch using reset_local().

    >>> metrics.reset_loca()

    Metrics are also accumulated for consecutive batches in the metrics.agg object.
    At the end of an epoch the MetricsTracker can be reset using end_epoch().

    >>> metrics.end_epoch()

    The MetricsTracker saves a copy of the accumulated metrics (metrics.agg) for
    each epoch which can be stored within a checkpoint.
    '''
    def __init__(self):
        self.local = defaultdict(float)
        self.agg = defaultdict(float)
        self.epoch_reports = []

    def update(self, metrics: dict):
        for m,v in metrics.items():
            self.local[m] += v
        
    def reduce(self):
        names, local = zip(*self.local.items())
        local = torch.tensor(local, dtype=torch.float16, requires_grad=False, device='cuda')
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        self.local = defaultdict(float, zip(names, local.cpu().numpy()))
        for k in self.local:
            self.agg[k] += self.local[k]

    def reset_local(self):
        self.local = defaultdict(float)
    
    def end_epoch(self):
        self.epoch_reports.append(dict(self.agg))
        self.local = defaultdict(float)
        self.agg = defaultdict(float)
    

