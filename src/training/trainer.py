import time
import torch
import warnings
from pathlib import Path
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from src.viz.training import save_plots
from src.utils.training_utils import show_metric_scores, save_best_loss
warnings.filterwarnings("ignore")


def run_single_batch(inputs: Tuple[torch.Tensor, torch.Tensor],
                     model: torch.nn.Module, loss_fn: torch.nn.Module, metrics: Dict[str, None],
                     device: str = "cpu", problem_type: str = "clf") -> Tuple[torch.nn.Module, Dict[str, float]]:
    res_dict = dict()
    X, y = inputs
    if problem_type == "clf":
        y = y.long()
    else:
        y = y.float()
    if device != "cpu":
        X = X.cuda()
        y = y.cuda()
    op = model(X)
    loss = loss_fn(op, y)
    res_dict["loss"] = loss.item()
    for name, metric in metrics.items():
        res_dict[name] = metric(op, y)
    return loss, res_dict


def train_single_epoch(loader: DataLoader, model: torch.nn.Module,
                       optim: torch.optim, loss_fn: torch.nn.Module, metrics: Dict[str, None],
                       device: str = "cpu", log_index: int = 10,
                       problem_type: str = "clf",) -> Tuple[Tuple[torch.nn.Module, torch.nn.Module], Dict[str, float]]:
    epoch_res_dict = {k: 0.0 for k, _ in metrics.items()}
    epoch_res_dict["loss"] = 0.0
    for step, inputs in enumerate(loader):
        optim.zero_grad()
        batch_loss, res_dict = run_single_batch(inputs, model, loss_fn, metrics, device, problem_type)
        for name, value in res_dict.items():
            epoch_res_dict[name] += value
        batch_loss.backward()
        optim.step()
        if step % log_index == 0:
            print(f"[Step {step}] Results : {show_metric_scores(res_dict)}")

    for name, value in epoch_res_dict.items():
        epoch_res_dict[name] /= step+1

    return (model, optim), epoch_res_dict


def validate_single_epoch(loader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module,
                          metrics: Dict[str, None], device: str = "cpu",
                          problem_type: str = "clf") -> Dict[str, float]:
    epoch_res_dict = {k: 0.0 for k, _ in metrics.items()}
    epoch_res_dict["loss"] = 0.0
    for step, inputs in enumerate(loader):
        _, res_dict = run_single_batch(inputs, model, loss_fn, metrics, device, problem_type)
        for name, value in res_dict.items():
            epoch_res_dict[name] += value

    for name, value in epoch_res_dict.items():
        epoch_res_dict[name] /= step+1

    return epoch_res_dict


def train_model(train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module,
                optim: torch.optim, loss_fn: torch.nn.Module, metrics: Dict[str, None],
                num_epochs: int, result_dir: Path, device: str = "cpu",
                log_index: int = 10, problem_type: str = "clf"):
    print("Initializing Training Job")
    process_init = time.time()
    if device != "cpu":
        model = model.cuda()
    print(f"Model loaded on {device}")
    best_loss = torch.inf
    train_results = {k: [] for k, _ in metrics.items()}
    train_results["loss"] = []
    val_results = {k: [] for k, _ in metrics.items()}
    val_results["loss"] = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} :")
        epoch_init = time.time()
        print("Training Cycle:")
        model.train()
        (model, optim), train_res_dict = train_single_epoch(train_loader, model, optim,
                                                            loss_fn, metrics, device, log_index, problem_type)
        for name, value in train_res_dict.items():
            train_results[name].append(value)
        print("Validation Cycle:")
        model.eval()
        val_res_dict = validate_single_epoch(val_loader, model, loss_fn, metrics, device, problem_type)
        for name, value in val_res_dict.items():
            val_results[name].append(value)
        print(f"Training Scores : {show_metric_scores(train_res_dict)}")
        print(f"Validation Scores : {show_metric_scores(val_res_dict)}")
        best_loss = save_best_loss(model, val_res_dict["loss"], best_loss, result_dir)
        print(f"Current Best Loss: {round(best_loss, 2)} seconds")
        print(f"Epoch Execution time : {round(time.time() - epoch_init, 6)}\n")
    save_plots(train_results, result_dir, dataname="training")
    save_plots(val_results, result_dir, dataname="validation")
    print("Training completed...")
    print(f"Total Execution Time : {round(time.time() - process_init, 2)} seconds\n")
