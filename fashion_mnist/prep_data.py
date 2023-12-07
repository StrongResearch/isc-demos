# Download dataset ahead of time
from torchvision import datasets

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
)