import os
from typing import Callable, List, Union

import polars as pl
from PIL import Image
import pydicom
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

class IBDataset(Dataset):
    def __init__(
        self,
        split_path: Union[os.PathLike, str],
        data_dir: Union[os.PathLike, str],  # DEPRECATED
        dataset_id: int,
        *,
        split_column: str = None,
        split_name: str = None,
        img_column: str = None,
        img_dir: str = None,
        img_suffix: str = None,
        img_transform: Callable = None,
        lbl_columns: List[str] = None,
        lbl_transform: Callable = None,
        msk_column: str = None,
        msk_dir: str = None,
        msk_suffix: str = None,
        msk_transform: Callable = None,
        txt_column: str = None,
        txt_dir: str = None,
        txt_suffix: str = None,
        txt_transform: Callable = None,
        com_transform: Callable = None,
        **kwargs,
    ):
        self.split_path = split_path
        self.split_column = split_column
        self.split_name = split_name
        self.dataset_id = dataset_id
        self.img_dir = img_dir if img_dir is not None else data_dir
        self.img_column = img_column
        self.img_suffix = img_suffix
        self.img_transform = img_transform
        self.lbl_columns = lbl_columns
        self.lbl_transform = lbl_transform
        self.msk_column = msk_column
        self.msk_dir = msk_dir
        self.msk_suffix = msk_suffix
        self.msk_transform = msk_transform
        self.txt_column = txt_column
        self.txt_dir = txt_dir
        self.txt_suffix = txt_suffix
        self.txt_transform = txt_transform
        self.com_transform = com_transform
        self.kwargs = kwargs

        self.samples = {}

        # Generate image paths
        if img_column is not None:
            self.samples["img"] = []
            i = 1
            while i != 411:
                self.samples["img"].append("/mnt/.node1/Open-Datasets/INB/AllDICOMs/" + str(i) + self.img_suffix)
                i += 1
        self.df = None
        # Extract the columns with labels
        if lbl_columns is not None:
            self.samples["lbl"] = self.df.select(lbl_columns)

        # Extract the column with text or a path to a text file
        if txt_column is not None:
            self.samples["txt"] = self.df.get_column(txt_column).to_list()

        # Extract the column with masks
        if msk_column is not None:
            self.samples["msk"] = (
                self.df.get_column(msk_column)
                .apply(lambda x: os.path.join(self.msk_dir, f"{x}{msk_suffix or ''}"))
                .to_list()
            )

        self.print_stats()
    
    def __getitem__(self, idx: int):
        """Return a dictionary with the requested sample."""
        sample = {"group_id": self.dataset_id}

        # Image
        if "img" in self.samples:
            sample["img"] = torch.from_numpy(np.array(pydicom.dcmread(self.samples["img"][idx]).pixel_array, dtype=np.float32)).apply_(lambda x: x / 255)
            if sample["img"].dim() < 4:
                sample["img"] = sample["img"].unsqueeze(0)

            if callable(self.img_transform):
                sample["img"] = self.img_transform(sample["img"])

        # Labels
        if "lbl" in self.samples:
            sample["lbl"] = self.samples["lbl"][idx]
            if callable(self.lbl_transform):
                sample["lbl"] = self.lbl_transform(sample["lbl"])

        # Mask
        if "msk" in self.samples:
            sample["msk"] = self.samples["msk"][idx]
            if callable(self.msk_transform):
                sample["msk"] = self.msk_transform(sample["msk"])

        # Text
        if "txt" in self.samples:
            sample["txt"] = self.samples["txt"][idx]
            if callable(self.txt_transform):
                sample["txt"] = self.txt_transform(sample["txt"])

        # Common transform applied on sample level
        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample

    def __len__(self):
        return 410

    def print_stats(self):
        print(
            f"""
            === Dataset stats for split={self.split_name or "full"} ===
            CSV file: {self.split_path}
            Data directory: {self.img_dir}
            Number of samples: {410}
        """
        )