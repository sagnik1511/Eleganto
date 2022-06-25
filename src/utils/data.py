from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, ToTensor, Resize
from pathlib import Path
from typing import Tuple, Optional


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


def pil2tensor(path: Path, shape: Tuple[int, int], aug: Optional = None):
    if not aug:
        aug = default_augment()
    img = load_img(path)
    img = aug(img)
    img = Resize(shape)(img)

    return img
