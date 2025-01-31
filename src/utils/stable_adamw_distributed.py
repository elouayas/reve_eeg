# Modified from https://github.com/warner-benjamin/optimi

# Copyright (c) 2023 Benjamin Warner
# SPDX-License-Identifier: MIT

# Based on PyTorch Optimizers
# PyTorch - PyTorch BSD-style license - Copyright (c) 2013-present PyTorch contributors

# Kahan summation inspired by Torch Distributed Experimental's `AnyPrecisionAdamW`
# torchdistX - BSD 3-Clause License - Copyright (c) Meta Platforms, Inc. and affiliates

# Learning rate decoupled weight decay inspired by Composer's `DecoupledSGDW` & `DecoupledAdamW`
# Composer - Apache License 2.0 - Copyright (c) 2022 MosaicML Composer authors

from __future__ import annotations

from typing import Any, Callable, Iterable
from warnings import warn

import torch
import torch.distributed as dist
from packaging.version import parse
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer, _default_to_fused_or_foreach
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


MIN_TORCH_2_1 = parse(torch.__version__) >= parse("2.1")


def debias(beta: float, step: int) -> float:
    """Adam-style debias correction. Returns `1 - beta ** step`."""
    return 1 - beta**step


def debias_beta(beta: float, step: int) -> float:
    """Applies the Adam-style debias correction into beta.

    Simplified version of `betahat = beta*(1-beta**(step-1))/(1-beta**step)`
    """
    return (beta**step - beta) / (beta**step - 1)


# modified from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
# Copyright 2019 Ross Wightman, Apache-2.0 License
def param_groups_weight_decay(
    model: nn.Module, weight_decay: float = 1e-2, additional_layers: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Creates parameter groups, excluding bias and normalization layers from weight decay.

    Parameters:
        model: Model to optimize
        weight_decay: Weight decay coefficient (default: 1e-2)
        additional_layers: Additional layer names to exclude from weight decay (default: None)

    Returns:
        List of parameter groups with and without weight decay.
    """
    additional_layers = set(additional_layers) if additional_layers is not None else set()
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in additional_layers:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class OptimiOptimizer(Optimizer):
    """Provides common functionality for optimi optimizers."""

    def __init__(self, params: Iterable[Tensor] | Iterable[dict], defaults: dict[str, Any]):
        if not 0.0 <= defaults["lr"]:
            raise ValueError(f"Invalid learning rate: lr={defaults['lr']}")
        if not 0.0 <= defaults["weight_decay"]:
            raise ValueError(f"Invalid weight decay: weight_decay={defaults['weight_decay']}")
        if defaults["decouple_lr"] and defaults["max_lr"] is None:
            defaults["max_lr"] = defaults["lr"]
        if defaults["max_lr"] is not None and not 0.0 <= defaults["max_lr"]:
            raise ValueError(f"Invalid maximum learning rate: max_lr={defaults['max_lr']}")

        if not MIN_TORCH_2_1:
            if defaults["foreach"]:
                raise ValueError(
                    f"foreach={defaults['foreach']} requires PyTorch 2.1 or later. Set foreach=False or upgrade PyTorch."
                )
            else:
                defaults["foreach"] = False
            if defaults["gradient_release"]:
                raise ValueError(
                    f"gradient_release={defaults['gradient_release']} requires PyTorch 2.1 or later. Upgrade PyTorch to use."
                )

        if defaults["decouple_lr"] and defaults["weight_decay"] >= 1e-3:
            warn(
                f"You are using weight_decay={defaults['weight_decay']} which is potentially high for decouple_lr={defaults['decouple_lr']}"
                f". Unlike decoupled weight decay, fully decoupled weight decay does not reduce weight decay by the learning rate.",
                category=UserWarning,
            )

        super().__init__(params, defaults)

        # by default perform the normal parameter update step
        self._optimizer_accumulation = False

        # if gradient_release is enabled, disable foreach step so normal optimizer step won't error
        if self.defaults["gradient_release"]:
            self.defaults["foreach"] = False
            for group in self.param_groups:
                group["foreach"] = False
                for p in group["params"]:
                    self.state[p]["group"] = group

    @property
    def optimizer_accumulation(self) -> bool:
        "Accumulate gradients in optimizer states during gradient release instead of a full step."
        return self._optimizer_accumulation

    @optimizer_accumulation.setter
    def optimizer_accumulation(self, optimizer_accumulation: bool):
        "Accumulate gradients in optimizer states during gradient release instead of a full step."
        self._optimizer_accumulation = optimizer_accumulation

    def step(self, closure: Callable | None = None, param: Tensor | None = None):
        """Performs a single optimization step on the whole model or individual parameter.

        Args:
            closure: A closure which reevaluates the model and returns the loss. Incompatible with
                performing an optimization step on a single `param`.
            param: An individual parameter to perform a fused optimization step during the backward
                pass. Requires optimizer to be initialized with `gradient_release=True` and model
                hooks created with `register_gradient_release`. Incompatible with `closure`.
        """
        raise NotImplementedError

    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True, param: Tensor | None = None):
        """Resets the gradients of all optimized parameters or individual parameter.

        Args:
            set_to_none: If True, the gradients will be deallocated after the call (default: True)
            param: Resets the gradients of the passed `param`. For use with `gradient_release=True`.
        """
        if param is None:
            super().zero_grad(set_to_none=set_to_none)
        elif param.grad is not None:
            if set_to_none:
                param.grad = None
            elif param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)


class StableAdamW(OptimiOptimizer):
    """StableAdamW optimizer. An AdamW-Adafactor hybrid with learning rate update clipping.

    This variant keeps the foreach-only implementation and adds optional cross-rank reduction for
    sharded training (e.g. FSDP flat parameters) so stabilization and reported gradient norms
    can match the full-parameter update.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate
        betas: Coefficients for gradient and squared gradient moving averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient. If `decouple_lr` is False, applies decoupled weight
            decay (default: 1e-2)
        eps: Added to denominator to improve numerical stability (default: 1e-6)
        decouple_lr: Apply fully decoupled weight decay instead of decoupled weight decay
            (default: False)
        max_lr: Maximum scheduled learning rate. Set if `lr` is not the maximum scheduled learning
            rate and `decouple_lr` is True (default: None)
        kahan_sum: Enables Kahan summation for more accurate parameter updates when training in low
            precision (float16 or bfloat16). If unspecified, automatically applies for low precision
            parameters (default: None)
        distributed_sharding: Whether parameters are sharded across ranks. If None, this is inferred
            from parameter metadata (`_is_sharded`) and reduction is enabled only when distributed
            training is initialized.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-2,
        eps: float = 1e-6,
        decouple_lr: bool = False,
        max_lr: float | None = None,
        kahan_sum: bool | None = None,
        return_norms: bool = True,
        distributed_sharding: bool | None = None,
    ):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]=}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]=}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps=}")

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
            "decouple_lr": decouple_lr,
            "max_lr": max_lr,
            "kahan_sum": kahan_sum,
            "foreach": True,
            "gradient_release": False,
            "setup": False,
            "distributed_sharding": distributed_sharding,
        }
        super().__init__(params, defaults)
        self.return_norms = return_norms
        self.grad_norms = {}

    def _init_state(self, group: dict[str, Any], state: dict[Tensor, Any], param: Tensor):
        if "kahan_comp" not in state:
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["eps_sq"] = torch.tensor(group["eps"] ** 2, dtype=param.dtype, device=param.device)

            if (group["kahan_sum"] or group["kahan_sum"] is None) and param.dtype in [torch.float16, torch.bfloat16]:
                state["kahan_comp"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                group["kahan_sum"] = True
            else:
                state["kahan_comp"] = None

            if group["gradient_release"]:
                state["step"] = torch.tensor(0, dtype=torch.int32)

    def _init_group(
        self,
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        eps_sqs: list[Tensor],
        kahan_comps: list[Tensor],
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]

            self._init_state(group, state, p)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            eps_sqs.append(state["eps_sq"])
            kahan_comps.append(state["kahan_comp"])

        if not group["setup"]:
            group["setup"] = True
            group["step"] = torch.tensor(0, dtype=torch.int32)

            if group["foreach"] is None:
                _, group["foreach"] = _default_to_fused_or_foreach(params, False, False)
                if not group["foreach"]:
                    raise ValueError("Foreach is required for this version supporting returning the gradnorm.")

    @torch.no_grad()
    def step(self, closure: Callable | None = None, param: Tensor | None = None):
        """Performs a single optimization step on the whole model or individual parameter.

        Args:
            closure: A closure which reevaluates the model and returns the loss. Incompatible with
                performing an optimization step on a single `param`.
            param: An individual parameter to perform a fused optimization step during the backward
                pass. Requires optimizer to be initialized with `gradient_release=True` and model
                hooks created with `register_gradient_release`. Incompatible with `closure`.
        """
        loss = None
        if closure is not None and param is None:
            with torch.enable_grad():
                loss = closure()

        l1_norm, l2_norm = None, None
        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, eps_sqs, kahan_comps = [], [], [], [], [], []
            self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, eps_sqs, kahan_comps)
            if not params:
                continue

            l1_norm, l2_norm = stableadamw(
                params=params,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                eps_sqs=eps_sqs,
                kahan_comps=kahan_comps,
                lr=group["lr"],
                beta1=group["beta1"],
                beta2=group["beta2"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                step=group["step"],
                decouple_lr=group["decouple_lr"],
                max_lr=group["max_lr"],
                kahan_sum=group["kahan_sum"],
                return_norms=self.return_norms,
                distributed_sharding=group["distributed_sharding"],
            )

        if l1_norm is None:
            l1_norm = torch.tensor(0.0)
            l2_norm = torch.tensor(0.0)
        self.grad_norms["l1_norm"] = l1_norm
        self.grad_norms["l2_norm"] = l2_norm

        return loss


def stableadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None] | None = None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: Tensor,
    decouple_lr: bool = False,
    max_lr: float | None = None,
    kahan_sum: bool = False,
    return_norms: bool = True,
    distributed_sharding: bool | None = None,
):
    """Functional API to apply a StableAdamW optimization step.

    See `optimi.StableAdamW` for more details.

    Args:
        params: Parameters to update
        grads: Parameter gradients
        exp_avgs: Gradient moving averages
        exp_avg_sqs: Squared gradient moving averages
        eps_sqs: Squared epsilon term tensors
        kahan_comps: Kahan summation compensations
        lr: Learning rate
        beta1: Gradient moving average coefficient
        beta2: Squared gradient moving average coefficient
        weight_decay: Weight decay coefficient
        eps: Added to denominator to improve numerical stability
        step: Step counter used for bias correction
        decouple_lr: Apply fully decoupled weight decay
        max_lr: Maximum scheduled learning rate for `decouple_lr`
        kahan_sum: Enables Kahan summation for low precision parameters
        distributed_sharding: Whether parameters are sharded across ranks and should use cross-rank
            reduction for RMS stabilization and returned gradient norms.
    """
    # calculate debiased beta hat & complement terms
    step.add_(1)
    beta1_comp = 1 - debias_beta(beta1, step.item())
    beta2_hat = debias_beta(beta2, step.item())

    if kahan_comps is None:
        kahan_comps = [None] * len(params)

    return _foreach_stableadamw(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        eps_sqs,
        kahan_comps,
        lr=lr,
        beta1_comp=beta1_comp,
        beta2_hat=beta2_hat,
        weight_decay=weight_decay,
        eps=eps,
        decouple_lr=decouple_lr,
        max_lr=max_lr,
        kahan_sum=kahan_sum,
        return_norms=return_norms,
        distributed_sharding=distributed_sharding,
    )


def _foreach_stableadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    weight_decay: float,
    eps: float,
    decouple_lr: bool,
    max_lr: float | None = None,
    kahan_sum: bool = False,
    return_norms: bool = True,
    distributed_sharding: bool | None = None,
    **kwargs,
):
    dist_sync = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if distributed_sharding is None:
        distributed_sharding = any(getattr(p, "_is_sharded", False) for p in params)
    sync_sharded_stats = dist_sync and distributed_sharding

    if return_norms:
        l1_norms = []
        l2_norms = []

    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, eps_sqs, kahan_comps])
    for (_, dtype), (
        (dev_params, dev_grads, dev_exp_avgs, dev_exp_avg_sqs, dev_eps_sqs, dev_kahan_comps),
        _,
    ) in grouped_tensors.items():
        do_kahan_sum = kahan_sum and dtype in [torch.float16, torch.bfloat16]

        if return_norms:
            l1_norms.extend(torch._foreach_norm(dev_grads, 1))
            l2_norms.extend(torch._foreach_norm(dev_grads, 2))

        # update gradient moving averages with debiased betas
        torch._foreach_lerp_(dev_exp_avgs, dev_grads, weight=beta1_comp)
        torch._foreach_mul_(dev_exp_avg_sqs, scalar=beta2_hat)
        torch._foreach_addcmul_(dev_exp_avg_sqs, dev_grads, dev_grads, value=1 - beta2_hat)

        # compute per parameter stabilization terms using dev_grads as temp buffer
        max_exp_avg_sqs = torch._foreach_maximum(dev_exp_avg_sqs, other=dev_eps_sqs)
        torch._foreach_pow_(dev_grads, exponent=2)
        torch._foreach_div_(dev_grads, max_exp_avg_sqs)

        # delete local intermediates to potentially save memory
        del max_exp_avg_sqs

        # calculate RMS stabilized learning rates and optionally weight decay
        if weight_decay != 0:
            neg_lrs, new_wds = [], []

            if sync_sharded_stats:
                sync_stats = []
                for r in dev_grads:
                    sync_stats.append(r.sum(dtype=torch.float64))
                    sync_stats.append(torch.tensor(r.numel(), device=r.device, dtype=torch.float64))
                sync_stats_tensor = torch.stack(sync_stats)
                dist.all_reduce(sync_stats_tensor, op=dist.ReduceOp.SUM)

                stat_idx = 0
                for _ in dev_grads:
                    mean_r = (sync_stats_tensor[stat_idx] / sync_stats_tensor[stat_idx + 1]).item()
                    stat_idx += 2
                    neg_lrs.append(-lr / max(1, mean_r**0.5))
            else:
                neg_lrs = [-lr / max(1, r.mean().sqrt().item()) for r in dev_grads]

            for neg_lr in neg_lrs:
                if decouple_lr:
                    new_wds.append(1 + (neg_lr / max_lr) * weight_decay)
                else:
                    new_wds.append(1 + neg_lr * weight_decay)

            # decoupled weight decay or fully decoupled weight decay
            torch._foreach_mul_(dev_params, scalars=new_wds)
        else:
            if sync_sharded_stats:
                sync_stats = []
                for r in dev_grads:
                    sync_stats.append(r.sum(dtype=torch.float64))
                    sync_stats.append(torch.tensor(r.numel(), device=r.device, dtype=torch.float64))
                sync_stats_tensor = torch.stack(sync_stats)
                dist.all_reduce(sync_stats_tensor, op=dist.ReduceOp.SUM)

                neg_lrs = []
                stat_idx = 0
                for _ in dev_grads:
                    mean_r = (sync_stats_tensor[stat_idx] / sync_stats_tensor[stat_idx + 1]).item()
                    stat_idx += 2
                    neg_lrs.append(-lr / max(1, mean_r**0.5))
            else:
                neg_lrs = [-lr / max(1, r.mean().sqrt().item()) for r in dev_grads]

        # Adam denominator using dev_grads as a temp buffer
        torch._foreach_copy_(dev_grads, dev_exp_avg_sqs)
        torch._foreach_sqrt_(dev_grads)
        torch._foreach_add_(dev_grads, eps)

        if do_kahan_sum:
            # Adam step
            torch._foreach_addcdiv_(dev_kahan_comps, dev_exp_avgs, dev_grads, scalars=neg_lrs)

            # update weights with kahan compensation using dev_grads as temp buffer
            torch._foreach_copy_(dev_grads, dev_params)
            torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

            # save error back to kahan compensation for next iteration
            torch._foreach_sub_(dev_grads, dev_params, alpha=1)
            torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
        else:
            # Adam step
            torch._foreach_addcdiv_(dev_params, dev_exp_avgs, dev_grads, scalars=neg_lrs)

    if return_norms:
        if not l1_norms:
            return torch.tensor(0.0), torch.tensor(0.0)
        l1_norm = torch.linalg.vector_norm(torch.stack(l1_norms), 1)
        l2_norm = torch.linalg.vector_norm(torch.stack(l2_norms), 2)
        if sync_sharded_stats:
            l1_norm_global = l1_norm.to(dtype=torch.float64)
            l2_norm_sq_global = l2_norm.square().to(dtype=torch.float64)
            dist.all_reduce(l1_norm_global, op=dist.ReduceOp.SUM)
            dist.all_reduce(l2_norm_sq_global, op=dist.ReduceOp.SUM)
            l1_norm = l1_norm_global.to(dtype=l1_norm.dtype)
            l2_norm = l2_norm_sq_global.sqrt().to(dtype=l2_norm.dtype)
        return l1_norm, l2_norm
    else:
        return None, None
