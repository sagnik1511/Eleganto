from sklearn.metrics import (r2_score,
                             mean_squared_error)
import torch
import numpy as np
from typing import Tuple


def fetch_in_device(device: str, true: torch.Tensor, pred: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    if device != "cpu":
        pred = pred.cpu()
        true = true.cpu()
    return pred.detach().numpy(), true.detach().numpy()


def R2(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred, true = fetch_in_device(pred.device, pred, true)
    return r2_score(true, pred)


def MSE(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred, true = fetch_in_device(pred.device, pred, true)
    return mean_squared_error(true, pred)


__METRIC_DICT__ = {
    "r2": R2
}