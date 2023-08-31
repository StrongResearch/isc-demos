import os
import pytest
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader
from cycling_utils.sampler import InterruptableDistributedSampler, AdvancedTooFarError, ResetProgressTooEarlyError

SEED = 13006555

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test.
    We need a process group to be initialized before each test
    because the (Interruptable)DistributedSampler uses the process group.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()

def test_constructor():
    t = torch.arange(10)
    dataset = TensorDataset(t)
    sampler = InterruptableDistributedSampler(dataset)
    assert sampler.progress == 0

def test_cannot_advance_too_much():
    t = torch.arange(10)
    dataset = TensorDataset(t)
    sampler = InterruptableDistributedSampler(dataset)
    with pytest.raises(AdvancedTooFarError):
        sampler.advance(11)

def test_cannot_reset_progress_early():
    t = torch.arange(10)
    dataset = TensorDataset(t)
    sampler = InterruptableDistributedSampler(dataset)
    with pytest.raises(ResetProgressTooEarlyError):
        with sampler.in_epoch(0):
            pass

@pytest.mark.parametrize("num_workers,", [0, 3])
@pytest.mark.parametrize("batch_size,", [1, 2, 3, 4, 5, 6])
def test_dataloader_equal_to_torch(batch_size, num_workers):
    n = 10
    t = torch.arange(n)
    dataset = TensorDataset(t)
    interruptable_sampler = InterruptableDistributedSampler(dataset, seed=SEED)
    torch_sampler = DistributedSampler(dataset, seed=SEED)

    interruptable_dataloader = DataLoader(
        dataset, sampler=interruptable_sampler, batch_size=batch_size, num_workers=num_workers
    )
    torch_dataloader = DataLoader(dataset, sampler=torch_sampler, batch_size=batch_size, num_workers=num_workers)

    for epoch in range(0, 10):
        torch_sampler.set_epoch(epoch)
        with interruptable_sampler.in_epoch(epoch):
            for step, (x_i, x_t) in enumerate(zip(interruptable_dataloader, torch_dataloader)):
                # work would be done here...
                assert torch.all(x_i[0] == x_t[0])
                interruptable_sampler.advance(len(x_i[0]))

@pytest.mark.parametrize("batch_size,", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_workers,", [0, 3])
def test_advance(batch_size, num_workers):
    n = 10
    t = torch.arange(n)
    dataset = TensorDataset(t)
    # shuffle false so that we can predict the order of the samples
    sampler = InterruptableDistributedSampler(dataset, seed=SEED, shuffle=False)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)

    with sampler.in_epoch(0):
        for step, (x, ) in enumerate(data_loader):
            # work would be done here...
            # plus one because of 0 indexing
            sampler.advance(len(x))
            assert sampler.progress == x[-1].item() + 1, "progress should be equal to the number of samples seen so far"

        assert sampler.progress == n, "progress should be equal to the number of samples"
    assert sampler.progress == 0, "progress should be reset to 0"

@pytest.mark.parametrize("batch_size,", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_workers,", [0, 3])
def test_advance_epochs(batch_size, num_workers):
    n = 10
    t = torch.arange(n)
    dataset = TensorDataset(t)
    # shuffle false so that we can predict the order of the samples
    sampler = InterruptableDistributedSampler(dataset, seed=SEED, shuffle=False)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)

    for epoch in range(0, 10):
        with sampler.in_epoch(epoch):
            for step, (x, ) in enumerate(data_loader):
                # work would be done here...
                # plus one because of 0 indexing
                sampler.advance(len(x))
                assert sampler.progress == x[-1].item() + 1, "progress should be equal to the number of samples seen so far"

            assert sampler.progress == n, "progress should be equal to the number of samples"
        assert sampler.progress == 0, "progress should be reset to 0"



class TrainingInterrupt(Exception):
    pass

@pytest.mark.parametrize("batch_size,", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_workers,", [0, 3])
def test_dataloader_suspend_resume(batch_size, tmp_path, num_workers):
    interrupt_epoch = 4
    interrupt_step = 7

    # implemented with functions as to not share namspaces
    def suspend_section():
        n = 50
        t = torch.arange(n)
        dataset = TensorDataset(t)
        interruptable_sampler = InterruptableDistributedSampler(dataset, seed=SEED, shuffle=False)

        interruptable_dataloader = DataLoader(
            dataset, sampler=interruptable_sampler, batch_size=batch_size, num_workers=num_workers
        )

        # run for a bit
        try:
            for epoch in range(0, 10):
                with interruptable_sampler.in_epoch(epoch):
                    for step, (x, ) in enumerate(interruptable_dataloader):
                        # work would be done here...
                        interruptable_sampler.advance(len(x))
                        if epoch == interrupt_epoch and step == interrupt_step:
                            raise TrainingInterrupt # simulate interrupt
        except TrainingInterrupt:
            pass

        print("suspend: ", x)
        # suspend
        sd = interruptable_sampler.state_dict()
        assert sd["epoch"] == interrupt_epoch
        assert sd["progress"] == (interrupt_step+1)*batch_size
        torch.save(interruptable_sampler.state_dict(), tmp_path / "interruptable_sampler.pt")
        return x[-1].item()
    last_item = suspend_section()

    # resume
    def resume_section():
        n = 50
        t = torch.arange(n)
        dataset = TensorDataset(t)
        interruptable_sampler = InterruptableDistributedSampler(dataset, seed=SEED, shuffle=False)

        interruptable_sampler.load_state_dict(torch.load(tmp_path / "interruptable_sampler.pt"))
        assert interruptable_sampler.epoch == interrupt_epoch
        assert interruptable_sampler.progress == (interrupt_step+1)*batch_size

        interruptable_dataloader = DataLoader(
            dataset, sampler=interruptable_sampler, batch_size=batch_size
        )

        first_step = True
        for epoch in range(interruptable_sampler.epoch, 10):
            with interruptable_sampler.in_epoch(epoch):
                for step, (x, ) in enumerate(interruptable_dataloader, start=interruptable_sampler.progress//batch_size):
                    # work would be done here...
                    interruptable_sampler.advance(len(x))
                    if first_step:
                        print("resume:", x)
                        assert last_item+1 == x[0].item(), "should be the same as the last item from the previous run"
                        first_step = False

    resume_section()
