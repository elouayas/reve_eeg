"""Shared training utilities for downstream experiments (seed, warmup, patience)."""

import random

import numpy as np
import torch
from typing import Callable

from models.classifier import ReveClassifier


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _get_exponential_warmup_lambda(total_steps: int) -> Callable[[int], float]:
    """
    Returns a lambda function for exponential warmup.
    The function takes a single argument (step) and returns the warmup value.
    """

    def exponential_warmup_lambda(step: int) -> float:
        return min(1.0, (10 ** (step / total_steps) - 1) / 9) if step < total_steps else 1.0

    return exponential_warmup_lambda


def get_warmup_scheduler(total_steps: int, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Returns a LambdaLR scheduler for exponential warmup.
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_get_exponential_warmup_lambda(total_steps))


def freeze_model(model: torch.nn.Module):
    """
    Disable gradient on all parameters of the model except for the linear head.
    """
    assert hasattr(model, "linear_head"), "Model must have a linear head to freeze parameters."
    assert hasattr(model.linear_head, "parameters"), "Linear head must have parameters to unfreeze."

    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear_head.parameters():
        param.requires_grad = True
    if hasattr(model, "cls_query_token"):
        model.cls_query_token.requires_grad = True # type: ignore

    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters are trainable")


def unfreeze_model(model: ReveClassifier | torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True

    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters are trainable")


class PatienceMonitor:
    """
    Monitor the validation acc and stop training if it doesn't improve for a certain number of epochs.
    """

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_acc = 0.0
        self.counter = 0

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
