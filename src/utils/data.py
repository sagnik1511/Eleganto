from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, ToTensor, Resize
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional
from torch.utils.data import DataLoader, random_split
from src.data.custom_dataset import InterestDataset


def load_img(path: Path):
    img = Image.open(path)

    return img


def default_augment():
    return transforms.Compose(
        [
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )


def pil2tensor(path: Path, shape: Tuple[int, int], aug=None):
    if not aug:
        aug = default_augment()
    img = load_img(path)
    img = aug(img)
    img = Resize(shape)(img)

    return img


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
