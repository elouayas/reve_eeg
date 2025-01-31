"""Main downstream training script for linear probing and fine-tuning."""

import os
import random
from builtins import print as bprint
from contextlib import nullcontext
from os.path import join as pjoin
from types import SimpleNamespace

import hydra
import idr_torch
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.resolver import register_resolvers
from downstream_tasks.dataloaders import get_data_loaders
from models.classifier import ReveClassifier
from models.encoder import REVE
from utils.model_utils import (
    freeze_model,
    get_flattened_output_dim,
    load_cls_query_token,
    load_encoder_checkpoint,
    unfreeze_model,
)
from utils.optim import get_lr_scheduler, get_optimizer


dtype_map = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}
FREQ = 200


# Registry must be called to handle custom resolvers like ${env:SCRATCH}
register_resolvers()

# UTILS


def print(*args, **kwargs):
    if idr_torch.is_master or kwargs.pop("force", False):
        bprint(*args, **kwargs)


### MAIN


def train_stage(  # noqa: C901, PLR0912, PLR0913, PLR0915
    config: DictConfig,
    current_cfg: DictConfig,
    model: ReveClassifier | DDP,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    stage_name: str,
):
    scaler = torch.amp.GradScaler(
        device=config.trainer.device,
        enabled="cuda" in config.trainer.device and "16" in config.get("torch_dtype", "fp32"),
    )
    criterion = nn.CrossEntropyLoss()
    device = config.trainer.device

    dtype_str = config.trainer.get("torch_dtype", "fp32")
    torch_dtype = dtype_map.get(dtype_str)

    optimizer = get_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        current_cfg.optimizer,
    )

    # Scheduler
    n_iter_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, current_cfg, n_iter_per_epoch)

    # Warmup
    warmup_epochs = current_cfg.warmup_epochs
    if warmup_epochs > 0:
        total_steps = len(train_loader) * warmup_epochs

        def exponential_warmup_lambda(step):
            if step < total_steps:
                return (10 ** (step / total_steps) - 1) / 9
            else:
                return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)
    else:
        warmup_scheduler = None

    model.train()

    best_val, best_test = 0, 0
    patience = 0

    # Epochs from specific config
    n_epochs = current_cfg.n_epochs
    patience_limit = current_cfg.patience

    og_model = model.module if isinstance(model, DDP) else model

    print(f"Starting training stage: {stage_name} for {n_epochs} epochs. Device: {device}")

    for epoch in range(n_epochs):
        warmup = epoch < warmup_epochs
        train_loss = train_one_epoch(
            model,
            criterion,
            optimizer,
            scaler,
            train_loader,
            warmup_scheduler if warmup else None,
            config,
            current_cfg,
        )

        if idr_torch.is_master:
            val_metrics = test(og_model, val_loader, device=device, binary=False, amp_dtype=torch_dtype)
        else:
            val_metrics = (0, 0, 0, 0, 0, 0)

        if dist.is_initialized():
            val_metrics_tensor = torch.tensor(val_metrics, device=device, dtype=torch.float32)
            dist.broadcast(val_metrics_tensor, src=0)
            val_metrics = val_metrics_tensor.tolist()

        val_acc, val_balanced_acc, val_cohen_kappa, val_f1, val_auroc, val_auc_pr = val_metrics

        if best_val < val_balanced_acc:
            best_val = val_balanced_acc
            if idr_torch.is_master:
                test_metrics = test(og_model, test_loader, device=device, binary=False, amp_dtype=torch_dtype)
            else:
                test_metrics = (0, 0, 0, 0, 0, 0)

            if dist.is_initialized():
                test_metrics_tensor = torch.tensor(test_metrics, device=device, dtype=torch.float32)
                dist.broadcast(test_metrics_tensor, src=0)
                test_metrics = test_metrics_tensor.tolist()

            test_acc, test_balanced_acc, test_cohen_kappa, test_f1, test_auroc, test_auc_pr = test_metrics
            best_test = test_balanced_acc
            patience = 0

            # Save best model
            if idr_torch.is_master:
                target_dir = os.getcwd()
                torch.save(og_model.state_dict(), pjoin(target_dir, "model_best.pth"))
                print(f"New best model saved with val_balanced_acc: {best_val:.4f}")
        else:
            patience += 1

        lr_curr = optimizer.param_groups[0]["lr"]
        log = "Stage {} | Val ({:3d} best test: {:.3f},best val: {:.3f},val: {:.3f},  ".format(
            stage_name,
            epoch,
            best_test,
            best_val,
            val_balanced_acc,
        )
        log += "LR {:.8f}, patience{:3d}/{:3d}, train_loss: {:.4f} ".format(
            lr_curr,
            patience,
            patience_limit + 1,
            train_loss,
        )
        print(log)

        if epoch > warmup_epochs:
            scheduler.step(val_acc)
        if patience > patience_limit:
            print(f"Stage {stage_name} finished due to patience limit.")
            break

    return model


def train_one_epoch(  # noqa: PLR0913
    model: ReveClassifier | DDP,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_loader: torch.utils.data.DataLoader,
    warmup_scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    config: DictConfig,
    current_cfg: DictConfig,
) -> float:
    losses: list[float] = []

    model.train()

    # Check mixup from current mode config
    use_mixup = current_cfg.get("mixup", False)
    device = config.trainer.device

    dtype = config.trainer.get("torch_dtype", "fp32")
    torch_dtype = dtype_map.get(dtype, torch.float32)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype) if "16" in dtype else nullcontext()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=not idr_torch.is_master)
    ema_loss = None
    for batch_idx, batch_data in pbar:
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            if isinstance(batch_data, dict):
                data = batch_data["sample"]
                target = batch_data["label"]
                pos = batch_data["pos"]
            else:
                data, target, pos = batch_data

            data, target, pos = (
                data.to(device, non_blocking=True),
                target.long().to(device, non_blocking=True),
                pos.to(device, non_blocking=True),
            )

            if use_mixup:
                mm = random.random()
                perm = torch.randperm(data.shape[0])
                output = model(mm * data + (1 - mm) * data[perm], pos)
                loss = mm * criterion(output, target) + (1 - mm) * criterion(output, target[perm])

            else:
                output = model(data, pos)
                loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.trainer.clip_grad)
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()

        losses.append(loss.item())

        ema_loss = loss.item() if ema_loss is None else 0.95 * ema_loss + 0.05 * loss.item()
        pbar.set_postfix(ema_loss=f"{ema_loss:.4f}")

        skip_lr_sched = scale != scaler.get_scale()
        if not skip_lr_sched and warmup_scheduler is not None:
            warmup_scheduler.step()

    return sum(losses) / len(losses)


def test(
    model: ReveClassifier,
    test_loader: torch.utils.data.DataLoader,
    device="cuda",
    binary=False,
    amp_dtype=torch.float16,
):
    score, count = 0, 0
    model.eval()
    y_decisions = []
    y_targets = []
    y_probs = []
    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if device == "cuda" else nullcontext()

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), disable=not idr_torch.is_master)
    for batch_idx, batch_data in pbar:
        with torch.no_grad(), ctx:
            if isinstance(batch_data, dict):
                data = batch_data["sample"]
                target = batch_data["label"]
                pos = batch_data["pos"]
            else:
                data, target, pos = batch_data

            data, target, pos = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
                pos.to(device, non_blocking=True),
            )
            output = model(data, pos)
            decisions = torch.argmax(output, dim=1)
            score += (decisions == target).int().sum().item()
            count += target.shape[0]
            y_decisions.append(decisions.detach().cpu())
            y_targets.append(target.detach().cpu())
            y_probs.append(output.detach().cpu())

    gt = torch.cat(y_targets).detach().cpu().numpy()
    pr = torch.cat(y_decisions).detach().cpu().numpy()
    pr_probs = torch.cat(y_probs).detach().cpu().numpy()
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


@hydra.main(version_base=None, config_name="config_dt", config_path="configs")
def main(args):  # noqa: C901, PLR0912, PLR0915
    if idr_torch.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=idr_torch.world_size,
            rank=idr_torch.rank,
        )
    device = f"cuda:{idr_torch.local_rank}"
    torch.cuda.set_device(device)

    init_seed = args.seed + idr_torch.rank
    torch.manual_seed(init_seed)
    random.seed(init_seed)

    print(f"Running with task: {args.task.name}")
    print(f"Output directory: {os.getcwd()}")

    cls_query_token = None
    if args.pretrained_path and "hf:" in args.pretrained_path:
        model_id = args.pretrained_path.split("hf:")[-1]
        print(f"Loading encoder from Hugging Face: {model_id}")
        encoder, cls_query_token = REVE.from_pretrained(
            model_id,
            cache_dir=args.get("cache_dir", ".cache"),
        )

    else:
        backbone_args = SimpleNamespace(
            embed_dim=args.encoder.transformer.embed_dim,
            depth=args.encoder.transformer.depth,
            heads=args.encoder.transformer.heads,
            head_dim=args.encoder.transformer.head_dim,
            mlp_dim_ratio=args.encoder.transformer.mlp_dim_ratio,
            use_geglu=args.encoder.transformer.use_geglu,
        )

        encoder = REVE(
            args_backbone=backbone_args,
            freqs=args.encoder.freqs,
            patch_size=args.encoder.patch_size,
            overlap_size=args.encoder.patch_overlap,
            noise_ratio=args.encoder.noise_ratio,
        )

        if args.pretrained_path and os.path.exists(args.pretrained_path):
            load_encoder_checkpoint(encoder, args.pretrained_path)
        else:
            print(f"Warning: Pretrained path {args.pretrained_path} not found or not specified.")

    if "n_chans" not in args.task:
        raise ValueError("n_chans must be specified in the task config")
    if "duration" not in args.task:
        raise ValueError("duration must be specified in the task config")

    n_chans = args.task.n_chans
    n_timepoints = int(args.task.duration * FREQ)

    print(f"Detected input shape: Chans={n_chans}, Timepoints={n_timepoints}")

    out_shape = None
    if args.task.classifier.pooling == "no":
        out_shape = get_flattened_output_dim(args, n_timepoints, n_chans)

    dropout = args.get("dropout", 0.0)

    model = ReveClassifier(
        encoder=encoder,
        n_classes=args.task.classifier.n_classes,
        dropout=dropout,
        pooling=args.task.classifier.pooling,
        out_shape=out_shape,
    )

    if args.pretrained_path and "hf:" in args.pretrained_path:
        if "cls_query_token" in locals() and cls_query_token is not None:
            print("Loading cls_query_token from Hugging Face checkpoint")
            model.cls_query_token.data.copy_(cls_query_token)
        else:
            print("WARNING: cls_query_token not found in Hugging Face checkpoint")
    elif args.pretrained_path and os.path.exists(args.pretrained_path):
        load_cls_query_token(model, args.pretrained_path)
    else:
        print(
            f"Warning: Pretrained path {args.pretrained_path} not found or not specified. cls_query_token not loaded.",
        )

    model.to(args.trainer.device)

    og_model = model

    training_mode = args.get("training_mode", "lp")
    data_loaders = None
    if args.task.data_loader.batch_size is not None:
        data_loaders = get_data_loaders(args.task.data_loader, args.loader)
    modes = []

    if training_mode == "lp":
        modes.append("lp")
    elif training_mode == "ft":
        modes.append("ft")
    elif training_mode == "lp+ft":
        modes = ["lp", "ft"]
    else:
        raise ValueError(f"Unknown training_mode {training_mode}")

    for mode in modes:
        if mode == "lp":
            print(">>> Setup Linear Probing (LP)")
            freeze_model(og_model)
            current_cfg = args.task.linear_probing
        elif mode == "ft":
            print(">>> Setup Fine-Tuning (FT)")
            unfreeze_model(og_model)
            current_cfg = args.task.fine_tuning
        else:
            raise ValueError(f"Unknown sub-mode {mode}")

        if idr_torch.size > 1:
            find_unused = mode == "lp"
            model = DDP(og_model, device_ids=[idr_torch.local_rank], find_unused_parameters=find_unused)
        else:
            model = og_model

        # Update batch size if specified for the stage
        if "batch_size" in current_cfg or data_loaders is None:
            new_bs = current_cfg.batch_size
            print(f"Batch size for stage {mode}: {new_bs}")
            args.task.data_loader.batch_size = new_bs
            # Refresh loaders
            data_loaders = get_data_loaders(args.task.data_loader, args.loader)

        train_stage(
            args,
            current_cfg,
            model,
            data_loaders["train"],
            data_loaders["val"],
            data_loaders["test"],
            stage_name=mode,
        )

    # Save model
    target_dir = os.getcwd()  # Hydra changes CWD
    if idr_torch.is_master:
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), pjoin(target_dir, "model_final.pth"))
        else:
            torch.save(model.state_dict(), pjoin(target_dir, "model_final.pth"))
        print(f"Model saved to {target_dir}")

    if idr_torch.size > 1 or dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
