"""Model utility helpers for freezing, checkpoint loading, and Optuna parsing."""

import math
from builtins import print as bprint
from typing import Any

import idr_torch
import torch
from omegaconf import DictConfig, ListConfig

from models.classifier import ReveClassifier
from models.encoder import REVE


def print(*args, **kwargs):
    if idr_torch.is_master or kwargs.pop("force", False):
        bprint(*args, **kwargs)


def get_flattened_output_dim(config, n_timepoints: int, n_chans: int) -> int:
    """Helper function to compute the flattened output dimension after the transformer."""
    pooling = config.task.classifier.pooling
    embed_dim = config.encoder.transformer.embed_dim

    if pooling in ["last", "all"]:
        return embed_dim

    patch_size = config.encoder.patch_size
    overlap_size = config.encoder.patch_overlap

    n_patches = math.ceil(
        (n_timepoints - patch_size) / (patch_size - overlap_size),
    )

    if (n_timepoints - patch_size) % (patch_size - overlap_size) == 0:
        n_patches += 1

    flat_dim = (n_chans * n_patches + 1) * embed_dim  # +1 for cls token
    return flat_dim


def freeze_model(model: ReveClassifier):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear_head.parameters():
        param.requires_grad = True
    model.cls_query_token.requires_grad = True


def unfreeze_model(model: ReveClassifier):
    for param in model.parameters():
        param.requires_grad = True


def load_encoder_checkpoint(encoder: REVE, checkpoint_path: str):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)

    # Remove module. prefix if present (distributed training)
    new_state_dict = {}
    for k, v in state_dict.items():
        k_ = k.replace("module.", "")
        new_state_dict[k_] = v

    # Filter for encoder keys and strip prefix
    encoder_state_dict = {}
    for k, v in new_state_dict.items():
        if k.startswith("encoder."):
            new_key = k.replace("encoder.", "")
            encoder_state_dict[new_key] = v

    if len(encoder_state_dict) == 0:
        print("WARNING: No 'encoder.' keys found in checkpoint. Trying to load as raw encoder weights.")
        encoder_state_dict = new_state_dict

    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    print(f"Loaded encoder weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def load_cls_query_token(reve_classifier: ReveClassifier, checkpoint_path: str):
    print(f"Loading cls_query_token from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)

    # Remove module. prefix if present (distributed training)
    new_state_dict = {}
    for k, v in state_dict.items():
        k_ = k.replace("module.", "")
        new_state_dict[k_] = v

    cls_key = "cls_query_token"
    if cls_key in new_state_dict:
        reve_classifier.cls_query_token.data.copy_(new_state_dict[cls_key])
        print("Loaded cls_query_token successfully.")
    else:
        print(f"WARNING: {cls_key} not found in checkpoint.")


def parse_optuna_config(cfg, current_path=""):  # noqa: C901, PLR0912, PLR0915
    optuna_params: dict[str, dict[str, Any]] = {}

    if isinstance(cfg, (DictConfig, dict)):
        for key in list(cfg.keys()):
            val = cfg[key]
            path = f"{current_path}.{key}" if current_path else str(key)

            if isinstance(val, str) and val.startswith("optuna:"):
                parts = val.split(":")
                param_type = parts[1]
                initial_val_str = parts[2]

                if param_type in ["float", "int"]:
                    low_str = parts[3]
                    high_str = parts[4]
                    is_log = len(parts) > 5 and parts[5] == "log"

                    if param_type == "float":
                        init_val = float(initial_val_str)
                        low = float(low_str)
                        high = float(high_str)
                    else:
                        init_val = int(initial_val_str)
                        low = int(low_str)
                        high = int(high_str)

                    optuna_params[path] = {
                        "type": param_type,
                        "init": init_val,
                        "low": low,
                        "high": high,
                        "log": is_log,
                    }
                    cfg[key] = init_val

                elif param_type in {"categorical", "cat"}:
                    choices = parts[3:]
                    init_val = initial_val_str
                    try:
                        if init_val.isdigit():
                            init_val = int(init_val)
                            choices = [int(c) for c in choices]
                        else:
                            try:
                                init_val = float(init_val)
                                choices = [float(c) for c in choices]
                            except ValueError:
                                pass  # keep as string
                    except Exception:
                        pass

                    optuna_params[path] = {
                        "type": "categorical",
                        "init": init_val,
                        "choices": choices,
                    }
                    cfg[key] = init_val

            elif isinstance(val, (DictConfig, dict, ListConfig, list)):
                child_params = parse_optuna_config(val, current_path=path)
                optuna_params.update(child_params)

    elif isinstance(cfg, (ListConfig, list)):
        for i in range(len(cfg)):
            val = cfg[i]
            path = f"{current_path}[{i}]"

            if isinstance(val, str) and val.startswith("optuna:"):
                parts = val.split(":")
                param_type = parts[1]
                initial_val_str = parts[2]

                if param_type in ["float", "int"]:
                    low_str = parts[3]
                    high_str = parts[4]
                    is_log = len(parts) > 5 and parts[5] == "log"

                    if param_type == "float":
                        init_val = float(initial_val_str)
                        low = float(low_str)
                        high = float(high_str)
                    else:
                        init_val = int(initial_val_str)
                        low = int(low_str)
                        high = int(high_str)

                    optuna_params[path] = {
                        "type": param_type,
                        "init": init_val,
                        "low": low,
                        "high": high,
                        "log": is_log,
                    }
                    cfg[i] = init_val

                elif param_type in {"categorical", "cat"}:
                    choices = parts[3:]
                    init_val = initial_val_str
                    try:
                        if init_val.isdigit():
                            init_val = int(init_val)
                            choices = [int(c) for c in choices]
                        else:
                            try:
                                init_val = float(init_val)
                                choices = [float(c) for c in choices]
                            except ValueError:
                                pass  # keep as string
                    except Exception:
                        pass

                    optuna_params[path] = {
                        "type": "categorical",
                        "init": init_val,
                        "choices": choices,
                    }
                    cfg[i] = init_val

            elif isinstance(val, (DictConfig, dict, ListConfig, list)):
                child_params = parse_optuna_config(val, current_path=path)
                optuna_params.update(child_params)

    return optuna_params
