import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import Dataset
from src.utils.data_utils import pil2tensor
from torch.utils.data import DataLoader, random_split


class InterestDataset(Dataset):

    def __init__(self, csv_path: Path, shape: Tuple, transform: Optional = None):
        super(InterestDataset, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.shape = shape
        self.transform = transform
        self.df = self.omit_bad_records(self.df)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def omit_bad_records(df: pd.DataFrame) -> pd.DataFrame:
        inds = []
        for index in range(len(df)):
            path = df.loc[index, "abs_path"]
            try:
                assert Image.open(path)
            except:
                inds.append(index)

        print(f"Found {len(inds)} bad records.")
        df.drop(inds, inplace=True)
        df.reset_index(inplace=True)
        return df

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        path = self.df.loc[index, "abs_path"]
        img = pil2tensor(path, self.shape, self.transform)
        label = self.df.loc[index, "Interest"]

        return img, label


def create_loader_from_csv(csv_path, batch_size: int, inp_shape: Tuple[int, int] = (224, 224),
                           ratio: float = 0.8, augment: Optional = None, shuffle: bool = True,
                           drop_last: bool = True) -> Tuple[DataLoader, DataLoader]:

    ds = InterestDataset(csv_path, inp_shape, augment)
    tr_size = int(len(ds) * ratio)
    vl_size = len(ds) - tr_size
    train_ds, val_ds = random_split(ds, [tr_size, vl_size])
    train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle, drop_last=drop_last)
    val_dl = DataLoader(val_ds, batch_size, shuffle=shuffle, drop_last=drop_last)
    return train_dl, val_dl
