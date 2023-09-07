import lightning as L
import torch.distributed as dist
from cycling_utils import atomic_torch_save

class EpochHandler(L.Callback):

    def __init__(self, sampler, checkpoint_dir, save_freq):
        super().__init__()
        self.sampler = sampler
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq

    def on_train_epoch_end(self, trainer, pl_module):
        self.sampler._reset_progress()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.sampler.advance(len(batch))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        self.sampler.advance(len(batch))
        
        if batch_idx % self.save_freq == 0 and dist.get_rank() == 0:

            atomic_torch_save({
                "sampler_state_dict": self.sampler.state_dict(),
            }, f"{self.checkpoint_dir}/sampler_last.pt")