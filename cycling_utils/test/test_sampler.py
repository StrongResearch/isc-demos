import os
import pytest
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader
from hypothesis import given, assume
from hypothesis import strategies as st
from cycling_utils.sampler import InterruptableDistributedSampler, AdvancedTooFarError

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


@given(st.integers(min_value=0, max_value=1000))
def test_constructor(n):
    t = torch.arange(n)
    dataset = TensorDataset(t)
    sampler = InterruptableDistributedSampler(dataset)
    assert sampler.progress == 0


def test_cannot_advance_too_much():
    t = torch.arange(10)
    dataset = TensorDataset(t)
    sampler = InterruptableDistributedSampler(dataset)
    with pytest.raises(AdvancedTooFarError):
        sampler.advance(11)


@pytest.mark.parametrize("num_workers,", [0, 3])
@pytest.mark.parametrize("batch_size,", [1, 2, 3, 4, 5, 6])
def test_dataloader_equal_to_torch(batch_size, num_workers):
    n = 10
    t = torch.arange(n)
    dataset = TensorDataset(t)
    interruptable_sampler = InterruptableDistributedSampler(dataset, seed=SEED)
    torch_sampler = DistributedSampler(dataset, seed=SEED)

    interruptable_dataloader = DataLoader(
        dataset,
        sampler=interruptable_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    torch_dataloader = DataLoader(
        dataset, sampler=torch_sampler, batch_size=batch_size, num_workers=num_workers
    )

    for epoch in range(0, 10):
        torch_sampler.set_epoch(epoch)
        with interruptable_sampler.in_epoch(epoch):
            for step, (x_i, x_t) in enumerate(
                zip(interruptable_dataloader, torch_dataloader)
            ):
                # work would be done here...
                assert torch.all(x_i[0] == x_t[0])
                interruptable_sampler.advance(len(x_i[0]))


@given(
    st.integers(1, 200),
    st.integers(min_value=1, max_value=100),
    st.booleans(),
    st.booleans(),
    st.integers(1, 7),
)
def test_dataloader_equal_to_torch_hypo(n, batch_size, drop_last, shuffle, epoch_end):
    assume(n > batch_size)
    num_workers = 0
    t = torch.arange(n)
    dataset = TensorDataset(t)
    interruptable_sampler = InterruptableDistributedSampler(
        dataset, seed=SEED, shuffle=shuffle
    )
    torch_sampler = DistributedSampler(dataset, seed=SEED, shuffle=shuffle)

    interruptable_dataloader = DataLoader(
        dataset,
        sampler=interruptable_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    torch_dataloader = DataLoader(
        dataset,
        sampler=torch_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    for epoch in range(0, epoch_end):
        torch_sampler.set_epoch(epoch)
        with interruptable_sampler.in_epoch(epoch):
            for step, (x_i, x_t) in enumerate(
                zip(interruptable_dataloader, torch_dataloader)
            ):
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
    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers
    )

    with sampler.in_epoch(0):
        for step, (x,) in enumerate(data_loader):
            # work would be done here...
            # plus one because of 0 indexing
            sampler.advance(len(x))
            assert (
                sampler.progress == x[-1].item() + 1
            ), "progress should be equal to the number of samples seen so far"

        assert (
            sampler.progress == n
        ), "progress should be equal to the number of samples"
    assert sampler.progress == 0, "progress should be reset to 0"


@pytest.mark.parametrize("batch_size,", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_workers,", [0, 3])
def test_advance_epochs(batch_size, num_workers):
    n = 10
    t = torch.arange(n)
    dataset = TensorDataset(t)
    # shuffle false so that we can predict the order of the samples
    sampler = InterruptableDistributedSampler(dataset, seed=SEED, shuffle=False)
    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers
    )

    for epoch in range(0, 10):
        with sampler.in_epoch(epoch):
            for step, (x,) in enumerate(data_loader):
                # work would be done here...
                # plus one because of 0 indexing
                sampler.advance(len(x))
                assert (
                    sampler.progress == x[-1].item() + 1
                ), "progress should be equal to the number of samples seen so far"

            assert (
                sampler.progress == n
            ), "progress should be equal to the number of samples"
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
        interruptable_sampler = InterruptableDistributedSampler(
            dataset, seed=SEED, shuffle=False
        )

        interruptable_dataloader = DataLoader(
            dataset,
            sampler=interruptable_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # run for a bit
        try:
            for epoch in range(0, 10):
                with interruptable_sampler.in_epoch(epoch):
                    for step, (x,) in enumerate(interruptable_dataloader):
                        # work would be done here...
                        interruptable_sampler.advance(len(x))
                        if epoch == interrupt_epoch and step == interrupt_step:
                            raise TrainingInterrupt  # simulate interrupt
        except TrainingInterrupt:
            pass

        print("suspend: ", x)
        # suspend
        sd = interruptable_sampler.state_dict()
        assert sd["epoch"] == interrupt_epoch
        assert sd["progress"] == (interrupt_step + 1) * batch_size
        torch.save(
            interruptable_sampler.state_dict(), tmp_path / "interruptable_sampler.pt"
        )
        return x[-1].item()

    last_item = suspend_section()

    # resume
    def resume_section():
        n = 50
        t = torch.arange(n)
        dataset = TensorDataset(t)
        interruptable_sampler = InterruptableDistributedSampler(
            dataset, seed=SEED, shuffle=False
        )

        interruptable_sampler.load_state_dict(
            torch.load(tmp_path / "interruptable_sampler.pt")
        )
        assert interruptable_sampler.epoch == interrupt_epoch
        assert interruptable_sampler.progress == (interrupt_step + 1) * batch_size

        interruptable_dataloader = DataLoader(
            dataset, sampler=interruptable_sampler, batch_size=batch_size
        )

        first_step = True
        for epoch in range(interruptable_sampler.epoch, 10):
            with interruptable_sampler.in_epoch(epoch):
                for step, (x,) in enumerate(
                    interruptable_dataloader,
                    start=interruptable_sampler.progress // batch_size,
                ):
                    # work would be done here...
                    interruptable_sampler.advance(len(x))
                    if first_step:
                        print("resume:", x)
                        assert (
                            last_item + 1 == x[0].item()
                        ), "should be the same as the last item from the previous run"
                        first_step = False

    resume_section()


@given(
    st.integers(1, 200),
    st.integers(1, 200),
    st.integers(min_value=1, max_value=7),
    st.integers(0, 100),
    st.booleans(),
    st.booleans(),
    st.integers(1, 7),
    st.booleans(),
)
def test_deterministic(
    progress, epoch, max_epochs, n, seed, shuffle, batch_size, drop_last
):
    state_dict = {
        "progress": progress,
        "epoch": epoch,
    }

    dataset = TensorDataset(torch.arange(n))

    sampler1 = InterruptableDistributedSampler(dataset, seed=seed, shuffle=shuffle)
    sampler2 = InterruptableDistributedSampler(dataset, seed=seed, shuffle=shuffle)

    if progress <= n:
        sampler1.load_state_dict(state_dict)
        sampler2.load_state_dict(state_dict)
    else:
        with pytest.raises(AdvancedTooFarError):
            sampler1.load_state_dict(state_dict)
        with pytest.raises(AdvancedTooFarError):
            sampler2.load_state_dict(state_dict)

    assert sampler1.progress == sampler2.progress
    assert sampler1.epoch == sampler2.epoch

    loader1 = DataLoader(
        dataset, sampler=sampler1, batch_size=batch_size, drop_last=drop_last
    )
    loader2 = DataLoader(
        dataset, sampler=sampler2, batch_size=batch_size, drop_last=drop_last
    )

    for epoch in range(0, max_epochs):
        with sampler1.in_epoch(epoch), sampler2.in_epoch(epoch):
            for (x1,), (x2,) in zip(loader1, loader2):
                assert torch.all(x1 == x2)
                sampler1.advance(len(x1))
                sampler2.advance(len(x2))
