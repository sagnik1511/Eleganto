import matplotlib.pyplot as plt
from typing import Tuple, Dict
from pathlib import Path
import os


def save_plots(metric_dict: Dict[str, float], directory: Path,
               figure_shape: Tuple[int, int] = (15, 4),
               dataname: str = "training"):
    for title, array in metric_dict.items():
        plt.figure(figsize=figure_shape)
        plt.plot(array, label=title)
    plt.legend()
    filename = os.path.join(directory, "scores", f"{dataname}_results.png")
    print(f"{dataname} saved at {filename}")
    plt.savefig(filename)
    plt.close()
