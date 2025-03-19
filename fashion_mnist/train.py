#####################################
# Progress Timing and Logging
# ----------------------------
# Important milestones in progress through this script are logged using a timing utility which also includes details of
# when milestones were reached and the elapsedtime between milestones. This can assist with debugging and optimisation.

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import argparse
import os
from operator import itemgetter
from pathlib import Path

import torch
import torch.distributed as dist
from model import ConvNet
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, PILToTensor, RandAugment

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)

timer.report("Completed imports")

#######################################
# Hyperparameters
# -----------------
# Hyperparameters are adjustable parameters that let you control the model optimization process. Different
# hyperparameter values can impact model training and convergence rates (`read more <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`
# about hyperparameter tuning). We define the following hyperparameters for training:
#  - **Number of Epochs** - the number times to iterate over the dataset
#  - **Batch Size** - the number of data samples propagated through the network before the parameters are updated
#  - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning
#       speed, while large values may result in unpredictable behavior during training.

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, required=True)
    # ---------------------------------------
    parser.add_argument("--epochs", type=int, default=100)
    # The number of times to loop over the whole dataset
    # ---------------------------------------
    parser.add_argument("--test-epochs", type=int, default=5)
    # Testing model performance on a test every "test-epochs" epochs
    # ---------------------------------------
    parser.add_argument("--dropout", type=float, default=0.2)
    # A model training regularisation technique to reduce over-fitting
    # ---------------------------------------
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-step-epochs", type=float, default=100)
    parser.add_argument("--lr-decay-rate", type=float, default=0.8)
    # This example demonstrates a StepLR learning rate scheduler. Different schedulers will require different
    # hyper-parameters.
    # ---------------------------------------
    parser.add_argument("--batch-size", type=int, default=16)
    # For distributed training it is important to distinguish between the per-GPU or "local" batch size (which this
    # hyper-parameter sets) and the "effective" batch size which is the product of the local batch size and the number
    # of GPUs in the cluster. With a local batch size of 16, and 10 nodes with 6 GPUs per node, the effective batch size
    # is 960. Effective batch size can also be increased using gradient accumulation which is not demonstrated here.
    # ---------------------------------------
    parser.add_argument("--save-freq", type=int, default=100)
    return parser

#####################################
# We define ``train_loop`` that loops over our optimization code, and ``test_loop`` that evaluates the model's
# performance against our test data. Inside the training loop, optimization happens in three steps:
#  * Call ``optimizer.zero_grad()`` to reset the gradients of model parameters. Gradients by default add up; to prevent
#       double-counting, we explicitly zero them at each iteration.
#  * Backpropagate the prediction loss with a call to ``loss.backward()``. PyTorch deposits the gradients of the loss
#       w.r.t. each parameter.
#  * Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the
#       backward pass.

def train_loop(model, optimizer, lr_scheduler, loss_fn, train_dataloader, test_dataloader, metrics, writer, saver, args):
    epoch = train_dataloader.sampler.epoch
    train_batches_per_epoch = len(train_dataloader)
    # Set the model to training mode - important for layers with different training / inference behaviour
    model.train()

    for inputs, targets in train_dataloader:

        # Prepare for batch processing
        optimizer.zero_grad()
        batch = train_dataloader.sampler.progress // train_dataloader.batch_size
        is_last_batch = (batch + 1) == train_batches_per_epoch
        is_save_batch = ((batch + 1) % args.save_freq == 0) or is_last_batch

        # Move input and targets to device
        inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)

        # Forward pass
        predictions = model(inputs)

        # Compute loss and log to metrics
        loss = loss_fn(predictions, targets)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Advance sampler - essential for interruptibility
        train_dataloader.sampler.advance(len(inputs))

        # Report training metrics
        metrics["train"].update({"examples_seen": len(inputs), "loss": loss.item()})
        metrics["train"].reduce().reset_local()  # Sum results from all ranks into "agg"
        total_batch_loss, examples_seen = itemgetter("loss", "examples_seen")(metrics["train"].agg)
        batch_avg_loss = total_batch_loss / examples_seen

        if is_last_batch:
            lr_scheduler.step()  # Step learning rate scheduler at the end of the epoch
            metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch

        # Save checkpoint
        if is_save_batch:
            checkpoint_directory = saver.prepare_checkpoint_directory()

            # Saving and reporting
            if args.is_master:
                total_progress = train_dataloader.sampler.progress + epoch * train_batches_per_epoch
                writer.add_scalar("Train/avg_loss", batch_avg_loss, total_progress)
                writer.add_scalar("Train/learn_rate", lr_scheduler.get_last_lr()[0], total_progress)

                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_sampler": train_dataloader.sampler.state_dict(),
                        "test_sampler": test_dataloader.sampler.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_metrics": metrics["train"].state_dict(),
                        "test_metrics": metrics["test"].state_dict(),
                        "best_accuracy": metrics["best_accuracy"]
                    },
                    os.path.join(checkpoint_directory, "checkpoint.pt")
                )

            saver.symlink_latest(checkpoint_directory)

            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - batch avg loss: {batch_avg_loss:,.3f}")

def test_loop(model, optimizer, lr_scheduler, loss_fn, train_dataloader, test_dataloader, metrics, writer, saver, args):
    epoch = test_dataloader.sampler.epoch
    test_batches_per_epoch = len(test_dataloader)
    # Set the model to evaluation mode - important for layers with different training / inference behaviour
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode also serves to
    # reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for inputs, targets in test_dataloader:

            # Prepare for batch processing
            batch = test_dataloader.sampler.progress // test_dataloader.batch_size
            is_last_batch = (batch + 1) == test_batches_per_epoch
            is_save_batch = ((batch + 1) % args.save_freq == 0) or is_last_batch

            # Move input and targets to device
            inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)

            # Inference
            predictions = model(inputs)

            # Test loss
            test_loss = loss_fn(predictions, targets)

            # Advance sampler
            test_dataloader.sampler.advance(len(inputs))

            # Performance metrics logging
            num_correct = (predictions.argmax(1) == targets).type(torch.float).sum()
            metrics["test"].update({"examples_seen": len(inputs), "loss": test_loss.item(), "num_correct": num_correct.item()})
            metrics["test"].reduce().reset_local()  # Sum results from all nodes into "agg"

            # Performance summary at the end of the epoch
            pct_test_correct = float("-inf")
            if args.is_master and is_last_batch:
                total_loss, examples_seen, num_correct = itemgetter("loss", "examples_seen", "num_correct")(metrics["test"].agg)
                avg_test_loss = total_loss / examples_seen
                pct_test_correct = num_correct / examples_seen
                writer.add_scalar("Test/avg_test_loss", avg_test_loss, epoch)
                writer.add_scalar("Test/pct_test_correct", pct_test_correct, epoch)
                metrics["test"].end_epoch()

                timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: {avg_test_loss:.3f}, TEST ACC: {pct_test_correct:.3f}")

            # Save checkpoint
            if is_save_batch:

                # sync pct_test_correct to determine force_save
                sync_pct_test_correct = torch.tensor(pct_test_correct).to('cuda')
                dist.broadcast(sync_pct_test_correct, src=0)

                # force save checkpoint if test performance improves, only after 20 epochs
                if (epoch > 20) and is_last_batch and (sync_pct_test_correct > metrics["best_accuracy"]):
                    force_save = True
                    metrics["best_accuracy"] = sync_pct_test_correct
                else:
                    force_save = False
                
                checkpoint_directory = saver.prepare_checkpoint_directory(force_save=force_save)

                if args.is_master:
                    atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "train_metrics": metrics["train"].state_dict(),
                            "test_metrics": metrics["test"].state_dict(),
                            "best_accuracy": metrics["best_accuracy"]
                        },
                        os.path.join(checkpoint_directory, "checkpoint.pt")
                    )
                
                saver.symlink_latest(checkpoint_directory)

timer.report("Defined helper function/s, loops, and model")

def main(args, timer):
    ##############################################
    # Distributed Training Configuration
    # -----------------
    # The following steps demonstrate configuration of the local GPU for distributed training. This includes
    # initialising the process group, obtaining and setting the rank of the GPU within the cluster and on the local
    # node.

    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    args.rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    _world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = args.rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'

    timer.report("Setup for distributed training")

    ##############################################
    # Data Transformation and Augmentation
    # ----------------------
    # Training data often requires pre-processing to ensure it is suitable for training, for example converting from a
    # PIL image to a Pytorch Tensor. This is also an opporuntiy to apply random perturbations to the training data each
    # time it is obtained from the dataset which has the effect of augmenting the size of the dataset and reducing
    # over-fitting.

    train_transform = Compose([PILToTensor(), RandAugment(), Lambda(lambda v: v.to(torch.float32) / 255.0)])
    test_transform = Compose([PILToTensor(), Lambda(lambda v: v.to(torch.float32) / 255.0)])

    data_path = os.path.join("/data", args.dataset_id)
    training_data = datasets.FashionMNIST(root=data_path, train=True, download=False, transform=train_transform)
    test_data = datasets.FashionMNIST(root=data_path, train=False, download=False, transform=test_transform)
    timer.report("Initialized datasets")

    ##############################################
    # Data Samplers and Loaders
    # ----------------------
    # Samplers for distributed training will typically assign a unique subset of the dataset to each GPU, shuffling
    # after each epoch, so that each GPU trains on a distinct subset of the dataset each epoch. The
    # InterruptibleDistributedSampler from cycling_utils by Strong Compute does this while also tracking progress of the
    # sampler through the dataset.

    train_sampler = InterruptableDistributedSampler(training_data)
    test_sampler = InterruptableDistributedSampler(test_data)
    timer.report("Initialized samplers")

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=3)
    test_dataloader = DataLoader(test_data, batch_size=64, sampler=test_sampler)
    timer.report("Initialized dataloaders")

    ##############################################
    # Model Preparation
    # ----------------------
    # This example demonstrates a Convolutional Neural Network (ConvNet, refer to the accompanying model.py for
    # details). After instantiating the model, we move it to the local device and prepare it for distributed training
    # with DistributedDataParallel (DDP).

    model = ConvNet(args.dropout)
    model = model.to(args.device_id)
    model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    ########################################
    # We initialize the loss function, optimizer, and learning rate scheduler and pass them to ``train_loop`` and
    # ``test_loop``. By setting the loss function reduction strategy to "sum" we are able to confidently summarise the
    # loss accross the whole cluster by summing the loss computed by each node. In general, it is important to consider
    # the validity of the metric summarisation strategy when using distributed training.

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=args.lr_decay_rate)
    timer.report(
        f"Ready for training with hyper-parameters: \ninitial learning_rate: {args.lr}, \nbatch_size: \
                 {args.batch_size}, \nepochs: {args.epochs}"
    )

    #####################################
    # Metrics and logging
    # -----------------
    # Metrics are commonly tracked and plotted during training to report on progress and model performance.

    metrics = {
        "train": MetricsTracker(), 
        "test": MetricsTracker(), 
        "best_accuracy": float("-inf")
    }
    writer = SummaryWriter(log_dir=os.environ["LOSSY_ARTIFACT_PATH"])

    #####################################
    # Checkpoint Saving - AtomicDirectory
    # -----------------
    # The AtomicDirectory saver is designed for use with Checkpoint Artifacts on Strong Compute. The User is responsible for 
    # implementing AtomicDirectory saver and saving checkpoints at their desired frequency.

    # Checkpoint Artifacts are synchronized every 10 minutes and/or at the end of each cycle on Strong Compute. Upon synchronization, 
    # the latest symlinked checkpoint/s saved by AtomicDirectory saver/s in the $CHECKPOINT_ARTIFACT_PATH directory will be shipped 
    # to Checkpoint Artifacts for the experiment. Any non-latest checkpoints saved since the previous Checkpoint Artifact sychronization 
    # will be deleted and not shipped.

    # The user can force non-latest checkpoints to also ship to Checkpoint Artifacts by calling `prepare_checkpoint_directory`
    # with `force_save = True`. This can be used, for example:
    # - to ensure every Nth saved checkpoint is archived for later analysis, or 
    # - to ensure that checkpoints are saved each time model performance improves. 

    output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    saver = AtomicDirectory(output_directory=output_directory, is_master=args.is_master)

    #####################################
    # Retrieve the checkpoint if the experiment is resuming from pause

    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        checkpoint_path = os.path.join(latest_checkpoint_path, "checkpoint.pt")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        metrics["train"].load_state_dict(checkpoint["train_metrics"])
        metrics["test"].load_state_dict(checkpoint["test_metrics"])
        metrics["best_accuracy"] = checkpoint["best_accuracy"]
        timer.report("Retrieved savedcheckpoint")

    #####################################
    # Main training loop
    # --------------------
    # Each epoch the training loop is called within a context set from the training InterruptibleDistributedSampler

    for epoch in range(train_dataloader.sampler.epoch, args.epochs):

        # important for use with InterruptableDistributedSampler
        train_dataloader.sampler.set_epoch(epoch)
        test_dataloader.sampler.set_epoch(epoch)

        train_loop(
            model, 
            optimizer, 
            lr_scheduler, 
            loss_fn, 
            train_dataloader, 
            test_dataloader, 
            metrics, 
            writer, 
            saver,
            args
        )

        if epoch % args.test_epochs == 0:

            test_loop(
                model,
                optimizer,
                lr_scheduler,
                loss_fn,
                train_dataloader,
                test_dataloader,
                metrics,
                writer,
                saver,
                args
            )

        # important for use with InterruptableDistributedSampler
        train_dataloader.sampler.reset_progress()
        test_dataloader.sampler.reset_progress()

    print("Done!")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)

#################################################################
# Further Reading
# -----------------------
# - `Loss Functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `Warmstart Training a Model <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_
