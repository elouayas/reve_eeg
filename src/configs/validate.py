"""Validation and pretty-print utilities for resolved Hydra configurations."""

import os
import warnings

import rich
import rich.syntax
import rich.tree
from omegaconf import OmegaConf


def pprint_cfg(cfg):
    OmegaConf.resolve(cfg)
    with open(".hydra/resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(f"Resolved configuration at {os.path.abspath('.hydra/resolved.yaml')}")

    _print_nested_dict(OmegaConf.to_container(cfg))


def _print_nested_dict(d):
    style = "default"
    tree = rich.tree.Tree("Resolved configuration", guide_style=style, style=style)

    def add_branch(parent: rich.tree.Tree, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                branch = parent.add(key)
                add_branch(branch, value)
            else:
                parent.add(rich.syntax.Syntax(f"{key}: {value}", "yaml", line_numbers=False, word_wrap=True))

    add_branch(tree, d)
    print()
    rich.print(tree)
    print()


def validate_train(args):
    if args.encoder.transformer.embed_dim != args.decoder.transformer.embed_dim:
        warnings.warn("Encoder and decoder embed dimensions do not match")

    if args.mode == "debug":
        pprint_cfg(args)
        os.environ["HYDRA_FULL_ERROR"] = "1"

    else:
        OmegaConf.resolve(args)

    return 0


def validate_dt(args):
    if args.mode == "debug":
        pprint_cfg(args)
        os.environ["HYDRA_FULL_ERROR"] = "1"
    else:
        OmegaConf.resolve(args)

    assert os.path.exists(args.pretrained_path), f"Model {args.pretrained_path} does not exist"

    return 0
