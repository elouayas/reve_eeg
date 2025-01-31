"""
Initialization functions for the model.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    default = "default"
    """
    All weights are initialized with the default HuggingFace Bert method. Set init_std=0.02 to match.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


############################################################################################################
# Initialization functions
############################################################################################################


@dataclass(frozen=True)
class ConfigInit:
    init_method: InitFnType = InitFnType.full_megatron
    init_std: float = 0.02
    init_cutoff_factor: float = 3.0
    hidden_size: int = 512
    num_hidden_layers: int = 8
    init_cls: bool = True


def init_weights(config: ConfigInit, module: nn.Linear, type_of_module: ModuleType | None = None) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param layer_dim: The effective input dim of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    assert config.init_method == InitFnType.full_megatron, "Only full_megatron init is currently supported"
    assert type_of_module is not None, "type_of_module must be provided when using full_megatron init"

    if type_of_module == ModuleType.in_module:
        # att_proj (same as QKV), ff_proj
        std = config.init_std
    elif type_of_module == ModuleType.out_module:
        # attn_out, ff_out
        std = config.init_std / math.sqrt(2.0 * config.num_hidden_layers)
    elif type_of_module == ModuleType.emb:
        # positional embeddings, token embeddings
        std = config.init_std
    elif type_of_module == ModuleType.final_out:
        # final output (ff_out)
        std = config.hidden_size**-0.5
    else:
        raise RuntimeError(f"Unknown module type '{type_of_module}'")

    nn.init.trunc_normal_(
        module if isinstance(module, nn.Parameter) else module.weight,
        mean=0.0,
        std=std,
        a=-config.init_cutoff_factor * std,
        b=config.init_cutoff_factor * std,
    )

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_method == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.num_hidden_layers))


###############################################


def init_mae(model, config_megatron: ConfigInit) -> None:
    """
    Initialize the weights of the MAE model.
    model: models.mae.MAE
    """
    init_weights(config_megatron, model.mask_token, type_of_module=ModuleType.in_module)
    if model.token_avg:
        init_weights(config_megatron, model.cls_query_token, type_of_module=ModuleType.in_module)
    for name, param in model.named_modules():
        if isinstance(param, nn.Linear) and ("encoder.transformer.layers" in name or "decoder.layers" in name):
            init_weights(config_megatron, param, type_of_module=ModuleType.in_module)


def init_cls(model, config_megatron: ConfigInit) -> None:
    """
    Initialize the weights of the classifier wrapper model.
    model: models.classifier.ReveClassifier
    """
    if config_megatron.init_cls:
        init_weights(config_megatron, model.cls_query_token, type_of_module=ModuleType.out_module)
    for name, param in model.named_modules():
        if "linear_head" and isinstance(param, nn.Linear):
            init_weights(config_megatron, param, type_of_module=ModuleType.final_out)
