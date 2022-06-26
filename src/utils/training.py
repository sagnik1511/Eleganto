import os
import torch
from typing import Dict
from pathlib import Path
from termcolor import cprint


def show_metric_scores(metrics: Dict[str, float]):
    metrics = {k: round(v, 6) for k, v in metrics.items()}
    print(metrics)


def save_best_loss(model: torch.nn.Module, curr: float, best: float, directory: Path) -> float:
    if best > curr:
        cprint("Model Updated...", "green")
        chkp = {
            "model": model.state_dict()
        }
        filename = os.path.join(directory, "models", "best_model.pt")
        torch.save(chkp, filename)
        print(f"Model saved at {filename}")
    else:
        cprint("Model didn't updated", "red")
    return min(best, curr)
