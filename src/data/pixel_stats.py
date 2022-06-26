import os
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.data_utils import load_img


def pixel_distribution(path: Path, show_curve: bool = True) -> List[int]:
    img = load_img(path)
    img_arr = np.array(img).reshape(-1)
    if show_curve:
        plt.figure(figsize=(20, 6))
        plt.hist(img_arr, range=(0, 256))
        plt.show()

    return img_arr


def get_img_stats(path: Path, show_curve: bool = True) -> List[int]:
    try:
        img_arr = pixel_distribution(path, show_curve=show_curve)
        return [round(np.mean(img_arr)[0]), int(np.median(img_arr)), pd.Series(img_arr).value_counts().index[0]]
    except:
        return [0, 0, 0]


def add_pixel_dist_stats(dataframe: pd.DataFrame,
                         root_dir: Path,
                         train_im_dir: Path) -> pd.DataFrame:
    dataframe["abs_path"] = dataframe["Id"].apply(lambda x: '/'.join([root_dir, train_im_dir, x+".jpg"]))
    dataframe[["pix_mean", "pix_median", "pix_mode"]] = \
        dataframe["abs_path"].apply(lambda x: get_img_stats(x, show_curve=False)).tolist()
    return dataframe


if __name__ == '__main__':
    root_directory = Path(os.getcwd())
    op_dir = os.path.join(root_directory, "processed_train.csv")
    train_image_directory = Path("dataset/images/train")
    df = pd.read_csv(os.path.join(root_directory, "dataset/train.csv"))

    df = add_pixel_dist_stats(df, root_dir=root_directory,
                              train_im_dir=train_image_directory)

    df.to_csv(op_dir, index=False)
