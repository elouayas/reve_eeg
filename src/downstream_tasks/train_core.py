"""
Core functions for fine-tuning models.
"""

import random

import torch
from torch.nn import functional as F
from tqdm import tqdm

from downstream_tasks.eval_core import eval_model
from downstream_tasks.utils import PatienceMonitor, freeze_model, get_warmup_scheduler, set_seed, unfreeze_model
from models.classifier import ReveClassifier
from models.lora import get_lora_model
from utils.optim import get_lr_scheduler, get_optimizer


def _forward_and_loss_reve(
    model: ReveClassifier,
    data: torch.Tensor,
    pos: torch.Tensor,
    target: torch.Tensor,
    mixup: bool = False,
):
    if mixup:
        mm = random.random()
        perm = torch.randperm(data.size(0))
        output = model(mm * data + (1 - mm) * data[perm], pos)
        loss = mm * F.cross_entropy(output, target) + (1 - mm) * F.cross_entropy(output, target[perm])
    else:
        output = model(data, pos)
        loss = F.cross_entropy(output, target)

    return loss


def _forward_and_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    pos: torch.Tensor,
    target: torch.Tensor,
    mixup: bool = False,
):
    if isinstance(model, ReveClassifier):
        return _forward_and_loss_reve(model, data, pos, target, mixup)

    output = model(data)  # other models don't process position
    loss = F.cross_entropy(output, target)
    return loss


def train_epoch(  # noqa
    model: torch.nn.Module,
    optimizer,
    scaler,
    warmup_scheduler,
    train_dataloader,
    config,
    stage,
    warmup: bool = False,
):
    """
    Perform a single training stage (linear probing or fine-tuning).
    """

    device = torch.device(config.trainer.device)
    grad_clip = config.trainer.clip_grad

    if stage == "lp":
        mixup = config.task.linear_probing.mixup
    elif stage == "ft":
        mixup = config.task.fine_tuning.mixup
    else:
        raise ValueError("Unknown stage")

    model.to(device)
    model.train()
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    pbar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))

    for batch in pbar:
        data, target, pos = batch["sample"], batch["label"], batch["pos"]
        data, target, pos = (
            data.to(device, non_blocking=True),
            target.long().to(device, non_blocking=True),
            pos.to(device, non_blocking=True),
        )
        optimizer.zero_grad()
        with torch.autocast(device_type=device_type, enabled=True, dtype=torch.float16):
            loss = _forward_and_loss(model, data, pos, target, mixup)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = scale != scaler.get_scale()
            if not skip_lr_sched and warmup:
                warmup_scheduler.step()

            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})


def train_dt(model: ReveClassifier | torch.nn.Module, config, train_dataloader, val_dataloader, test_dataloader):
    """
    Train the model in two stages: linear probing and fine-tuning."
    """

    epochs_lp = config.task.linear_probing.n_epochs
    es_patience = config.task.linear_probing.early_stop_patience
    warmup_epochs = config.task.linear_probing.warmup_epochs
    warump_steps = len(train_dataloader) * warmup_epochs

    set_seed(config.seed)
    device = torch.device(config.trainer.device)
    model.to(device)
    model.train()

    freeze_model(model)
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), config.task.linear_probing.optimizer)
    warmup_scheduler = get_warmup_scheduler(warump_steps, optimizer)
    patience_monitor = PatienceMonitor(patience=es_patience)

    reduce_scheduler = get_lr_scheduler(optimizer, config.task.linear_probing, n_iter=None)

    scaler = torch.amp.GradScaler()

    print("Linear probing...")
    for epoch in range(epochs_lp):
        print(f"Epoch {epoch + 1}/{epochs_lp}")
        train_epoch(model, optimizer, scaler, warmup_scheduler, train_dataloader, config, "lp", warmup=epoch <= warmup_epochs)
        metrics = eval_model(model, val_dataloader, config)
        accuracy = metrics[0]
        reduce_scheduler.step(accuracy)
        print(f"Validation accuracy: {accuracy:.4f}")
        if patience_monitor(accuracy):
            print("Early stopping...")
            break
    metrics = eval_model(model, test_dataloader, config)
    accuracy = metrics[0]
    print(f"Test accuracy: {accuracy:.4f}")

    if config.task.fine_tuning is None:  # early exit if only linear probing is needed
        print("Training complete. No fine-tuning stage.")
        return model

    epochs_ft = config.task.fine_tuning.n_epochs
    es_patience = config.task.fine_tuning.early_stop_patience
    warmup_epochs = config.task.fine_tuning.warmup_epochs
    warump_steps = len(train_dataloader) * warmup_epochs

    unfreeze_model(model)
    if isinstance(model, ReveClassifier):
        model = get_lora_model(model, config.lora)
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), config.task.fine_tuning.optimizer)
    warmup_scheduler = get_warmup_scheduler(warump_steps, optimizer)
    patience_monitor = PatienceMonitor(patience=es_patience)

    reduce_scheduler = get_lr_scheduler(optimizer, config.task.fine_tuning, n_iter=None)

    print("Fine-tuning...")
    for epoch in range(epochs_ft):
        print(f"Epoch {epoch + 1}/{epochs_ft}")
        train_epoch(model, optimizer, scaler, warmup_scheduler, train_dataloader, config, "ft", warmup=epoch <= warmup_epochs)
        metrics = eval_model(model, val_dataloader, config)
        accuracy = metrics[0]
        reduce_scheduler.step(accuracy)
        print(f"Validation accuracy: {accuracy:.4f}")
        if patience_monitor(accuracy):
            print("Early stopping...")
            break
    metrics = eval_model(model, test_dataloader, config)
    accuracy = metrics[0]
    print(f"Test accuracy: {accuracy:.4f}")

    print("Training complete.")
    return model
