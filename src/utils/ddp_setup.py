"""
Utility functions for setting up distributed training with Accelerate.
"""

import os
import shutil
from datetime import timedelta
from logging import getLogger as default_getLogger
from os.path import join as pjoin
from typing import Any, TypeVar

from accelerate import Accelerator
from accelerate.logging import get_logger as accelerate_get_logger
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import OmegaConf


NCCL_TIMEOUT = 30

logger = default_getLogger(__name__)


def get_logger(name: str):
    """Get a logger with the given name. Unifies accelerate and logging modules."""

    # Use accelerate's logger if Accelerator is instantiated
    if "Accelerator" in globals():
        return accelerate_get_logger(name)

    # Otherwise, use the default logger
    return default_getLogger(name)


def get_accelerator(args):
    if args.mode == "debug":
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=NCCL_TIMEOUT))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.trainer.accumulate_grad_batches,
        kwargs_handlers=[timeout],
        log_with="wandb" if args.wandb.log else None,
    )

    accelerator.even_batches = False
    accelerator.step_scheduler_with_optimizer = False

    if accelerator.is_main_process and args.wandb.log:
        os.environ["WANDB_DIR"] = args.wandb.path
        if "lustre" in args.data.path or args.wandb.offline:
            os.environ["WANDB_MODE"] = "offline"
        accelerator.init_trackers(
            project_name=args.wandb.project,
            init_kwargs={
                "entity": args.wandb.entity if args.wandb.offline else "online",
                "tags": args.wandb.tags,
                "notes": args.wandb.comment,
            },
            config=OmegaConf.to_container(args, resolve=True),
        )
    accelerator.print(args)

    return accelerator


def save_state(accelerator: Accelerator, args, epoch):
    """
    Saves the current state of the training process and manages checkpoint files.
    Will only save and create symbolic links if the current process is the main one.
    Args:
        accelerator: The accelerator object used for distributed training, which provides the `save_state` method.
        args: args object parsed with Hydra, containing the checkpointing configuration.
        epoch (int): The current epoch number, used to name the checkpoint file.
    Functionality:
        1. Saves the current state to a directory named after the current epoch.
        2. Creates or updates a symbolic link named "last" pointing to the latest checkpoint.
        3. Removes older checkpoints if the number of saved checkpoints exceeds the specified limit (`keep_last`).
    Raises:
        OSError: If there are issues with file or directory operations, such as creating directories or removing files.
    """

    accelerator.save_state(pjoin(args.checkpointing.state_path, f"epoch_{epoch}"))

    if accelerator.is_main_process:
        if args.checkpointing.keep_last:
            os.makedirs(pjoin(args.checkpointing.state_path, "last"), exist_ok=True)
            accelerator.save_state(pjoin(args.checkpointing.state_path, "last"))

        ls = [int(f.split("_")[-1]) for f in os.listdir(args.checkpointing.state_path) if f.startswith("epoch_")]

        if len(ls) > args.checkpointing.max_states:
            ls.sort()
            logger.info(f"Removing {len(ls) - args.checkpointing.max_states} old checkpoints")
            for f in ls[: -args.checkpointing.keep_last]:
                shutil.rmtree(pjoin(args.checkpointing.state_path, f"epoch_{f}"))


def save_encoder(accelerator: Accelerator, mae, args, epoch):
    encoder = accelerator.unwrap_model(mae).encoder
    save_path = pjoin(args.checkpointing.state_path, f"epoch_{epoch}", "encoder.pth")
    accelerator.save(encoder.state_dict(), save_path)
    accelerator.print(f"Saved to {os.path.abspath(save_path)}")
    logger.info(f"Saved to {os.path.abspath(save_path)}")


T = TypeVar("T")


def ensure_type(item: Any, expected_type: type[T]) -> T:
    """Type checking utils"""
    if not isinstance(item, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(item)}")
    return item
