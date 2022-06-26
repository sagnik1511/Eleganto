import os
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import Dataset
from src.utils.data import default_augment, pil2tensor


class InterestDataset(Dataset):

    def __init__(self, csv_path: Path, shape: Tuple, transform: Optional = None):
        super(InterestDataset, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.shape = shape
        self.transform = transform if transform else default_augment()
        self.df = self.omit_bad_records(self.df)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def omit_bad_records(df: pd.DataFrame) -> pd.DataFrame:
        inds = []
        for index in range(len(df)):
            if not os.path.isfile(df.loc[index, "abs_path"]):
                inds.append(index)
        print(f"Found {len(inds)} bad records.")
        df.drop(inds, inplace=True)
        return df

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        path = self.df.loc[index, "abs_path"]
        img = pil2tensor(path, self.shape, self.transform)
        label = self.df.loc[index, "Interest"]

        return img, label
