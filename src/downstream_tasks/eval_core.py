"""
Core functions for evaluating fine-tuned models.
"""

import os

import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from models.classifier import ReveClassifier


def eval_model(model: ReveClassifier | torch.nn.Module, loader, config):
    """
    Evaluate the model on the validation or test set.
    """

    device = torch.device(config.trainer.device)
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    binary = config.task.classifier.n_classes == 2  # noqa: PLR2004

    model.to(device)
    model.eval()

    y_decisions = []
    y_targets = []
    y_probs = []
    score, count = 0, 0
    with torch.no_grad():
        for batch_data in loader:
            with torch.autocast(device_type=device_type, enabled=True, dtype=torch.float16):
                data, target, pos = batch_data["sample"], batch_data["label"], batch_data["pos"]
                data, target, pos = (
                    data.to(device, non_blocking=True),
                    target.to(device, non_blocking=True),
                    pos.to(device, non_blocking=True),
                )
                with torch.inference_mode():
                    output = model(data, pos) if isinstance(model, ReveClassifier) else model(data)

                decisions = torch.argmax(output, dim=1)
                score += (decisions == target).int().sum().item()
                count += target.shape[0]
                y_decisions.append(decisions)
                y_targets.append(target)
                y_probs.append(output)

    gt = torch.cat(y_targets).cpu().numpy()
    pr = torch.cat(y_decisions).cpu().numpy()
    pr_probs = torch.cat(y_probs).cpu().numpy()
    acc = score / count
    balanced_acc = balanced_accuracy_score(gt, pr)
    cohen_kappa = cohen_kappa_score(gt, pr)
    f1 = f1_score(gt, pr, average="weighted")

    if binary:
        auroc = roc_auc_score(gt, pr_probs[:, 1])
        auc_pr = average_precision_score(gt, pr_probs[:, 1])
        return acc, balanced_acc, cohen_kappa, f1, auroc, auc_pr
    else:
        return acc, balanced_acc, cohen_kappa, f1, 0, 0


def soup_model(models: list[ReveClassifier]):
    """Merge multiple models by averaging their weights."""
    model = models[0]
    state_dict = model.state_dict()
    for k in state_dict:
        state_dict[k] = torch.stack([m.state_dict()[k] for m in models]).mean(0)
    model.load_state_dict(state_dict)
    return model


def log_metrics(metrics: list[float], epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metrics_with_epoch = [epoch] + metrics
    df = pd.DataFrame([metrics_with_epoch])

    if not os.path.exists(path):
        df.to_csv(path, header=["epoch", "acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr"], index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)
