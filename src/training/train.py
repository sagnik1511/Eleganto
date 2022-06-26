import time
import torch
from termcolor import cprint
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict
from src.utils.training import show_metric_scores, save_best_loss
from src.viz.training import save_plots
from torch.nn import MSELoss

LOSS_FN = MSELoss()


def run_single_batch(inputs: Tuple[torch.Tensor, torch.Tensor],
                     model: torch.nn.Module, metrics: Dict[str, None],
                     device: torch.device = torch.device("cpu")) -> Tuple[None, Dict[str, float]]:
    res_dict = dict()
    X, y = inputs
    X.to(device)
    y.to(device)
    op = model(X)
    loss = LOSS_FN(op, y)
    res_dict["loss"] = loss.item()
    for name, metric in metrics.items():
        res_dict[name] = metric(op, y)
    return loss, res_dict


def train_single_epoch(loader: DataLoader, model: torch.nn.Module,
                       optim: torch.optim, metrics: Dict[str, None],
                       device: torch.device = torch.device("cpu"),
                       log_index: int = 10) -> Tuple[Tuple[torch.nn.Module, None], Dict[str, float]]:
    epoch_res_dict = {k: 0.0 for k, _ in metrics.items()}
    for step, inputs in enumerate(loader):
        optim.zero_grad()
        batch_loss, res_dict = run_single_batch(inputs, model, metrics, device)
        for name, value in res_dict.items():
            epoch_res_dict[name] += value
        batch_loss.backward()
        optim.step()
        if step % log_index == 0:
            print(f"[Step {step}] Results : {show_metric_scores(res_dict)}")

    return (model, optim), epoch_res_dict


def validate_single_epoch(loader: DataLoader, model: torch.nn.Module, metrics: Dict[str, None],
                          device: torch.device = torch.device("cpu")) -> Dict[str, float]:
    epoch_res_dict = {k: 0.0 for k, _ in metrics.items()}
    for step, inputs in enumerate(loader):
        batch_loss, res_dict = run_single_batch(inputs, model, metrics, device)
        for name, value in res_dict.items():
            epoch_res_dict[name] += value

    return epoch_res_dict


def train_model(train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module,
                optim: torch.optim, metrics: Dict[str, None],
                num_epochs: int, result_dir: Path, device: torch.device = torch.device("cpu"),
                log_index: int = 10):
    cprint("Initializing Training Job", "blue")
    process_init = time.time()
    model.to(device)
    print(f"Model loaded on {device.type}")
    best_loss = torch.inf
    train_results = {k: [] for k, _ in metrics.items()}
    val_results = {k: [] for k, _ in metrics.items()}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} :")
        epoch_init = time.time()
        model.train()
        (model, optim), train_res_dict = train_single_epoch(train_loader, model, optim, metrics, device, log_index)
        for name, value in train_res_dict.items():
            train_results[name].append(value)
        model.eval()
        val_res_dict = validate_single_epoch(val_loader, model, metrics, device)
        for name, value in val_res_dict.items():
            val_results[name].append(value)
        print(f"Training Scores : {show_metric_scores(train_res_dict)}")
        print(f"Validation Scores : {show_metric_scores(val_res_dict)}")
        best_loss = save_best_loss(model, val_res_dict["loss"], best_loss, result_dir)
        print(f"Current Best Loss: {round(best_loss, 6)}")
        print(f"Epoch Execution time : {round(time.time() - epoch_init, 6)}\n")
    save_plots(train_results, result_dir, dataname="training")
    save_plots(val_results, result_dir, dataname="validation")
    cprint("Training completed...", "blue")
    print(f"Total Execution Time : {round(time.time() - process_init, 6)}\n")
