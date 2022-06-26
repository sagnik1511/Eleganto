from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             log_loss)
import torch


def fetch_in_device(device, true, pred):
    if device == "cpu":
        pred = pred.cpu()
        true = true.cpu()
    return pred.detach().numpy(), true.detach().numpy()


def R2(pred, true):
    pred, true = fetch_in_device(pred.device, pred, true)
    return r2_score(true, pred)


def LOG_LOSS(pred, true):
    pred, true = fetch_in_device(pred.device, pred, true)
    return log_loss(pred, true)


def MSE(pred, true):
    pred, true = fetch_in_device(pred.device, pred, true)
    return mean_squared_error(true, pred)


__METRIC_DICT__ = {
    "r2": R2,
    "log_loss": LOG_LOSS
}