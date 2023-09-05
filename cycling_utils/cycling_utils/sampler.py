import math
import torch
from torch.utils.data import Dataset, DistributedSampler
from contextlib import contextmanager

class HasNotResetProgressError(Exception):
    pass

class AdvancedTooFarError(Exception):
    pass

class InterruptableDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        This is a DistributedSampler that can be suspended and resumed.

        This works by keeping track of the epoch and progress within the epoch.
        The progress is the number of samples that have been returned by the
        sampler. The epoch is the number of times the sampler has been iterated
        over.

        The epoch is incremented at the start of each epoch. The epoch is set
        to 0 at initialization.

        The progress is incremented by the number of samples returned by the
        sampler. The progress is reset to 0 at the end of each epoch.

        Suspending and resuming the sampler is done by saving and loading the
        state dict. The state dict contains the epoch and progress. This works
        because the permutation of the dataset is deterministic given the seed
        and epoch.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.progress = 0
        self._has_reset_progress = True

    def _reset_progress(self):
        self.progress = 0
        self._has_reset_progress = True

    def set_epoch(self, epoch):
        if not self._has_reset_progress:
            raise HasNotResetProgressError("You must reset progress before setting epoch e.g. `sampler.reset_progress()`\nor use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`")
        self.epoch = epoch

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if not self.progress <= self.num_samples:
            raise AdvancedTooFarError(f"progress should be less than or equal to the number of samples. progress: {self.progress}, num_samples: {self.num_samples}")
        self.epoch = state_dict["epoch"]

    def advance(self, n: int):
        """
        Record that n samples have been consumed.
        """
        self.progress += n
        if self.progress > self.num_samples:
            raise AdvancedTooFarError("You have advanced too far. You can only advance up to the total size of the dataset.")

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # slice from progress to pick up where we left off
    
        for idx in indices[self.progress :]:
            yield idx

    @contextmanager
    def in_epoch(self, epoch):
        """
        This context manager is used to set the epoch. It is used like this:

        ```
        for epoch in range(0, 10):
            with sampler.in_epoch(epoch):
                for step, (x, ) in enumerate(dataloader):
                    # work would be done here...
        ```
        """
        self.set_epoch(epoch)
        yield
        self._reset_progress()
