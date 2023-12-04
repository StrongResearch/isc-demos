import bisect
import copy
import math
from collections import defaultdict
from itertools import chain, repeat
from contextlib import contextmanager

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.model_zoo import tqdm


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

# class GroupedBatchSampler(BatchSampler):
#     """
#     Wraps another sampler to yield a mini-batch of indices.
#     It enforces that the batch only contain elements from the same group.
#     It also tries to provide mini-batches which follows an ordering which is
#     as close as possible to the ordering from the original sampler.
#     Args:
#         sampler (Sampler): Base sampler.
#         group_ids (list[int]): If the sampler produces indices in range [0, N),
#             `group_ids` must be a list of `N` ints which contains the group id of each sample.
#             The group ids must be a continuous set of integers starting from
#             0, i.e. they must be in the range [0, num_groups).
#         batch_size (int): Size of mini-batch.
#     """

#     def __init__(self, sampler, group_ids, batch_size):
#         if not isinstance(sampler, Sampler):
#             raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
#         self.sampler = sampler
#         self.group_ids = group_ids
#         self.batch_size = batch_size

#     def __iter__(self):
#         buffer_per_group = defaultdict(list)
#         samples_per_group = defaultdict(list)

#         num_batches = 0
#         for idx in self.sampler:
#             group_id = self.group_ids[idx]
#             buffer_per_group[group_id].append(idx)
#             samples_per_group[group_id].append(idx)
#             if len(buffer_per_group[group_id]) == self.batch_size:
#                 yield buffer_per_group[group_id]
#                 num_batches += 1
#                 del buffer_per_group[group_id]
#             assert len(buffer_per_group[group_id]) < self.batch_size

#         # now we have run out of elements that satisfy
#         # the group criteria, let's return the remaining
#         # elements so that the size of the sampler is
#         # deterministic
#         expected_num_batches = len(self)
#         num_remaining = expected_num_batches - num_batches
#         if num_remaining > 0:
#             # for the remaining batches, take first the buffers with the largest number
#             # of elements
#             for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
#                 remaining = self.batch_size - len(buffer_per_group[group_id])
#                 samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
#                 buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
#                 assert len(buffer_per_group[group_id]) == self.batch_size
#                 yield buffer_per_group[group_id]
#                 num_remaining -= 1
#                 if num_remaining == 0:
#                     break
#         assert num_remaining == 0

#     def __len__(self):
#         return len(self.sampler) // self.batch_size

class HasNotResetProgressError(Exception):
    pass

class AdvancedTooFarError(Exception):
    pass

class InterruptableDistributedGroupedBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        group_ids: list[int], 
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        This is a DistributedSampler that can be suspended and resumed. 
        This works by keeping track of the sample batches that have already been 
        dispatched. 
        
        This InterruptableDistributedGroupedBatchSampler also enables the sampling 
        strategy exhibited in the torch vision detection reference wherein batches 
        are created from images from within the same 'group', defined in the 
        torchvision example by similarity of image aspect ratio. 

        https://github.com/pytorch/vision/tree/main/references/detection

        Any grouping can be similarly applied by passing suitable group_ids.

        For this reason, InterruptableDistributedGroupedBatchSampler progress is
        tracked in units of batches, not samples. This is an important
        distinction from the InterruptableDistributedSampler which tracks progress
        in units of samples. The progress is reset to 0 at the end of each epoch.

        The epoch is set to 0 at initialization and incremented at the start 
        of each epoch.

        Suspending and resuming the sampler is done by saving and loading the
        state dict. The state dict contains the epoch and progress.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        # OVERALL STATUS INDICATOR
        self.progress = 0
        self._has_reset_progress = True
        self.batch_size = batch_size
        self.group_ids = group_ids
        self.batches = self._create_batches()

    def _create_batches(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make dataset evenly divisible accross ranks
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make dataset evenly divisible accross ranks
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample indices to use on this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # PRE-COMPUTE GROUPED BATCHES
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)
        self.num_batches = math.ceil(len(indices)/ self.batch_size)

        batches = [] # pre-computed so progress refers to batches, not samples.
        for idx in indices:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                batches.append(buffer_per_group[group_id])
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        num_remaining = self.num_batches - len(batches)
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with the largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = self._repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                batches.append(buffer_per_group[group_id])
                num_remaining -= 1
                if num_remaining == 0:
                    break

        # Check that the batches are all good to go
        assert len(batches) == self.num_batches
        return batches
    
    def _repeat_to_at_least(self, iterable, n):
        repeat_times = math.ceil(n / len(iterable))
        repeated = chain.from_iterable(repeat(iterable, repeat_times))
        return list(repeated)

    def _reset_progress(self):
        self.progress = 0
        self._has_reset_progress = True

    def set_epoch(self, epoch: int) -> None:
        raise NotImplementedError("Use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`")

    def _set_epoch(self, epoch):
        if not self._has_reset_progress:
            raise HasNotResetProgressError("You must reset progress before setting epoch e.g. `sampler.reset_progress()`\nor use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`")
        self.epoch = epoch

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if not self.progress <= self.num_batches:
            raise AdvancedTooFarError(f"progress should be less than or equal to the number of batches. progress: {self.progress}, num_batches: {self.num_batches}")
        self.epoch = state_dict["epoch"]

    def advance(self):
        """
        Record that one batch has been consumed.
        """
        self.progress += 1
        if self.progress > self.num_batches:
            raise AdvancedTooFarError(f"You have advanced too far. You can only advance up to the total number of batches: {self.num_batches}.")

    def __iter__(self):

        # slice from progress to pick up where we left off
        for batch in self.batches[self.progress:]:
            yield batch

    def __len__(self):
        return self.num_batches

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
        self._set_epoch(epoch)
        yield
        self._reset_progress()


class InterruptableDistributedGroupedBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        group_ids: list, 
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        This is a DistributedSampler that can be suspended and resumed.

        This works by keeping track of the sample batches that have already been 
        dispatched. This InterruptableDistributedGroupedBatchSampler also 
        reproduces the sampling strategy exhibited in the torch vision detection
        reference wherein batches are created from images from within the same
        'group', defined in the torchvision example by similarity of image 
        aspect ratio. 

        https://github.com/pytorch/vision/tree/main/references/detection

        For this reason, InterruptableDistributedGroupedBatchSampler progress is
        tracked in units of batches, not samples. This is an important
        distinction from the InterruptableDistributedSampler which tracks progress
        in units of samples. The progress is reset to 0 at the end of each epoch.

        The epoch is set to 0 at initialization and incremented at the start 
        of each epoch.

        Suspending and resuming the sampler is done by saving and loading the
        state dict. The state dict contains the epoch and progress.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        # OVERALL STATUS INDICATOR
        self.progress = 0
        self._has_reset_progress = True
        self.batch_size = batch_size
        self.group_ids = group_ids
        self.batches = self._create_batches()

    def _create_batches(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make dataset evenly divisible accross ranks
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make dataset evenly divisible accross ranks
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample indices to use on this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # PRE-COMPUTE GROUPED BATCHES
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)
        self.num_batches = math.ceil(len(indices)/ self.batch_size)

        batches = [] # pre-computed so progress refers to batches, not samples.
        for idx in indices:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                batches.append(buffer_per_group[group_id])
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        num_remaining = self.num_batches - len(batches)
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with the largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                batches.append(buffer_per_group[group_id])
                num_remaining -= 1
                if num_remaining == 0:
                    break

        # Check that the batches are all good to go
        assert len(batches) == self.num_batches
        return batches

    def _reset_progress(self):
        self.progress = 0
        self._has_reset_progress = True

    def set_epoch(self, epoch: int) -> None:
        raise NotImplementedError("Use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`")

    def _set_epoch(self, epoch):
        if not self._has_reset_progress:
            raise HasNotResetProgressError("You must reset progress before setting epoch e.g. `sampler.reset_progress()`\nor use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`")
        self.epoch = epoch

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if not self.progress <= self.num_batches:
            raise AdvancedTooFarError(f"progress should be less than or equal to the number of batches. progress: {self.progress}, num_batches: {self.num_batches}")
        self.epoch = state_dict["epoch"]

    def advance(self):
        """
        Record that one batch has been consumed.
        """
        self.progress += 1
        if self.progress > self.num_batches:
            raise AdvancedTooFarError(f"You have advanced too far. You can only advance up to the total number of batches: {self.num_batches}.")

    def __iter__(self):

        # slice from progress to pick up where we left off
        for batch in self.batches[self.progress:]:
            yield batch

    def __len__(self):
        return self.num_batches

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
        self._set_epoch(epoch)
        yield
        self._reset_progress()


def _compute_aspect_ratios_slow(dataset, indices=None):
    print(
        "Your dataset doesn't support the fast path for "
        "computing the aspect ratios, so will iterate over "
        "the full dataset and load every image instead. "
        "This might take some time..."
    )
    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0],
    )
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset) # list of aspect ratios for each image in the dataset
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins) # list of bin indexes to which each image belongs
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print(f"Using {fbins} as bins for aspect ratio quantization")
    print(f"Count of instances per bin: {counts}")
    return groups
