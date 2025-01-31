"""LoRA configuration and adapter wrapping utilities for REVE classifiers."""

from peft import LoraConfig, get_peft_model

from models.classifier import ReveClassifier


def get_lora_config(model: ReveClassifier, rank: int, apply_to=("patch", "mlp4d", "attention", "ffn")):
    """
    Get the Lora configuration for the model.
    """

    print(f"Applying Lora on {', '.join(apply_to)}")

    patch = "patch" in apply_to
    mlp4d = "mlp4d" in apply_to
    attention = "attention" in apply_to
    ffn = "ffn" in apply_to

    target_modules = []
    encoder = model.encoder

    if any([patch, mlp4d, attention, ffn]) is False:
        raise ValueError("At least one of the flags must be True.")

    # Patch Embedding
    if patch:
        target_modules.append("to_patch_embedding.0")

    # MLP4D
    if mlp4d:
        target_modules.append("mlp4d.0")

    # Transformer
    for i in range(len(encoder.transformer.layers)):
        if attention:
            target_modules.extend([f"transformer.layers.{i}.0.to_qkv", f"layers.{i}.0.to_out"])
        if ffn:
            target_modules.extend([f"transformer.layers.{i}.1.net.1", f"layers.{i}.1.net.3"])

    return LoraConfig(
        r=rank,
        lora_alpha=rank,  # set lora_alpha to rank
        target_modules=target_modules,
    )


class CustomGetLora:
    def __init__(self, config, train_all=False):
        """
        Initializes the Lora class with the given configuration.
        Args:
            config (LoraConfig): The configuration for the Lora model.
            train_all (bool, optional): A flag indicating whether to train all parameters. Defaults to False.

        If train_all is False, only the Lora parameters are trained.
        Else, all (except the base layer) parameters are trained.
        """

        self.config = config
        self.train_all = train_all

    def get_model(self, model):
        ret = get_peft_model(model, self.config)

        if self.train_all:
            for name, param in ret.named_parameters():
                if "base_layer" not in name:
                    param.requires_grad = True

        return ret

    def get_opt_params(self, model, verbose=False):
        ret = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                ret.append(param)
            elif verbose:
                print(f"Skipping {name}")

        return ret


def get_lora_model(model: ReveClassifier, args_lora):
    """ "
    Get the Lora model with the given configuration.
    Possibly skip the Lora model if args_lora.enable is False.
    """
    if args_lora.enabled is False:
        return model

    rank = args_lora.rank
    apply_to = args_lora.apply_to

    config = get_lora_config(model=model, rank=rank, apply_to=apply_to)
    return CustomGetLora(config=config, train_all=False).get_model(model=model)
