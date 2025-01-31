"""Standalone downstream checkpoint evaluation script across data splits."""

import os
import random
from types import SimpleNamespace

import hydra
import torch

from configs.resolver import register_resolvers
from downstream_tasks.dataloaders import get_data_loaders
from dt import dtype_map, test
from models.classifier import ReveClassifier
from models.encoder import REVE
from utils.model_utils import get_flattened_output_dim


# Registry must be called to handle custom resolvers
register_resolvers()

# python src/eval_repro.py --config-name local_config.yaml --config-dir .


@hydra.main(version_base=None, config_name="config_dt", config_path="configs")
def main(args):
    device = args.trainer.device

    # Fix seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading data loaders for task: {args.task.name}")
    data_loaders = get_data_loaders(args.task.data_loader, args.loader)

    # Model Setup (Replicated from dt_hydra.py)
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

    n_chans = args.get("n_chans")
    n_timepoints = args.get("n_timepoints")
    if n_chans is None or n_timepoints is None:
        raise ValueError("n_chans and n_timepoints must be specified in the config")

    out_shape = None
    if args.task.classifier.pooling == "no":
        out_shape = get_flattened_output_dim(args, n_timepoints, n_chans)

    model = ReveClassifier(
        encoder=encoder,
        n_classes=args.task.classifier.n_classes,
        dropout=args.get("dropout", 0.0),
        pooling=args.task.classifier.pooling,
        out_shape=out_shape,
    )

    # Load weights
    checkpoint_path = "model_best.pth"
    if not os.path.exists(checkpoint_path):
        # Check if provided via command line override
        checkpoint_path = args.get("checkpoint_path", "model_best.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Evaluate
    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        if split not in data_loaders:
            continue

        dtype_str = args.trainer.get("torch_dtype", "fp32")
        torch_dtype = dtype_map.get(dtype_str)

        print(f"\n>>> Evaluating {split} split...")
        metrics = test(model, data_loaders[split], device=device, binary=False, amp_dtype=torch_dtype)
        acc, balanced_acc, cohen_kappa, f1, auroc, auc_pr = metrics

        results[split] = {
            "Accuracy": acc,
            "Balanced Acc": balanced_acc,
            "Cohen Kappa": cohen_kappa,
            "F1 Score": f1,
        }

        print(f"{split.upper()} Metrics:")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Cohen Kappa:       {cohen_kappa:.4f}")
        print(f"  F1 Score:          {f1:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
