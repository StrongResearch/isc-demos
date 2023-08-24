"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
**Optimization** ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Optimizing Model Parameters
===========================

Now that we have a model and data it's time to train, validate and test our model by optimizing its parameters on
our data. Training a model is an iterative process; in each iteration the model makes a guess about the output, calculates
the error in its guess (*loss*), collects the derivatives of the error with respect to its parameters (as we saw in
the `previous section  <autograd_tutorial.html>`_), and **optimizes** these parameters using gradient descent. For a more
detailed walkthrough of this process, check out this video on `backpropagation from 3Blue1Brown <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__.

Prerequisite Code
-----------------
We load the code from the previous sections on `Datasets & DataLoaders <data_tutorial.html>`_
and `Build Model  <buildmodel_tutorial.html>`_.
"""

from pathlib import Path
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse

parser = argparse.ArgumentParser()
# get lr and batch size from command line
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--save-dir", type=Path, required=True)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()
print(args)

checkpoint_latest = args.save_dir / "latest.pt"
checkpoint_latest_temp = args.save_dir / "latest.temp.pt"

dist.init_process_group("nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
device_id = rank % torch.cuda.device_count()
is_master = rank == 0

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_sampler = DistributedSampler(training_data)
train_dataloader = DataLoader(training_data, batch_size=64, sampler=train_sampler)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model = model.to(device_id)
model = DDP(model, device_ids=[device_id])


##############################################
# Hyperparameters
# -----------------
#
# Hyperparameters are adjustable parameters that let you control the model optimization process.
# Different hyperparameter values can impact model training and convergence rates
# (`read more <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`__ about hyperparameter tuning)
#
# We define the following hyperparameters for training:
#  - **Number of Epochs** - the number times to iterate over the dataset
#  - **Batch Size** - the number of data samples propagated through the network before the parameters are updated
#  - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
#

learning_rate = args.lr
# batch_size = 64
batch_size = args.batch_size
epochs = args.epochs
print(f"learning_rate: {learning_rate}, batch_size: {batch_size}, epochs: {epochs}")


#####################################
# Inside the training loop, optimization happens in three steps:
#  * Call ``optimizer.zero_grad()`` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
#  * Backpropagate the prediction loss with a call to ``loss.backward()``. PyTorch deposits the gradients of the loss w.r.t. each parameter.
#  * Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the backward pass.


########################################
# .. _full-impl-label:
#
# Full Implementation
# -----------------------
# We define ``train_loop`` that loops over our optimization code, and ``test_loop`` that
# evaluates the model's performance against our test data.

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device_id)
        y = y.to(device_id)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if is_master and batch % 1 == 0:
            loss, current = loss.item(), (batch) #* len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device_id)
            y = y.to(device_id)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Learning Rate: {learning_rate}, Batch Size: {batch_size} \n")


########################################
# We initialize the loss function and optimizer, and pass it to ``train_loop`` and ``test_loop``.
# Feel free to increase the number of epochs to track the model's improving performance.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


completed_epochs = 0
if os.path.exists(checkpoint_latest):
    checkpoint = torch.load(checkpoint_latest)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epochs = checkpoint["epoch"]
    print(f"Loaded checkpoint from {checkpoint_latest}")

for t in range(completed_epochs, epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_sampler.set_epoch(t)
    train_loop(train_dataloader, model, loss_fn, optimizer)
    if is_master: # TODO distributed validation
        test_loop(test_dataloader, model, loss_fn)
        torch.save({
            "epoch": t + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_latest_temp)
        os.replace(checkpoint_latest_temp, checkpoint_latest)
        print(f"Saved checkpoint to {checkpoint_latest}")
print("Done!")



#################################################################
# Further Reading
# -----------------------
# - `Loss Functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `Warmstart Training a Model <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_
#
