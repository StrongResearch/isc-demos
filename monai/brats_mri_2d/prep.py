# Obtain the dataset
from monai.apps import DecathlonDataset

_ = DecathlonDataset(root_dir="/mnt/Datasets/Open-Datasets/MONAI", task="Task01_BrainTumour", section="training", download=True)