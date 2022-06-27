from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)
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


def ACC(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred = torch.argmax(torch.nn.Softmax(dim=1)(pred), dim=1)
    true = true.int()
    pred, true = fetch_in_device(pred.device, pred, true)
    return accuracy_score(true, pred)


def PRC(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred = torch.argmax(torch.nn.Softmax(dim=1)(pred), dim=1)
    true = true.int()
    pred, true = fetch_in_device(pred.device, pred, true)
    return precision_score(true, pred, average="weighted")


def RCL(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred = torch.argmax(torch.nn.Softmax(dim=1)(pred), dim=1)
    true = true.int()
    pred, true = fetch_in_device(pred.device, pred, true)
    return recall_score(true, pred, average="weighted")


def F1(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred = torch.argmax(torch.nn.Softmax(dim=1)(pred), dim=1)
    true = true.int()
    pred, true = fetch_in_device(pred.device, pred, true)
    return f1_score(true, pred, average="weighted")


__REG_METRIC_DICT__ = {
    "r2_score": R2
}

__CLF_METRIC_DICT__ = {
    "accuracy_score": ACC,
    "precision": PRC,
    "recall": RCL,
    "f1_score": F1
}