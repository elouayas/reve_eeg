"""Main Hydra entrypoint for REVE foundation pretraining with Accelerate."""

import os
import time
from os.path import join as pjoin

import hydra
from accelerate.scheduler import AcceleratedScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.resolver import register_resolvers
from configs.validate import validate_train
from models.mae import MAE
from utils.data_loading import get_train_val_loaders
from utils.ddp_setup import ensure_type, get_accelerator, get_logger, save_state
from utils.optim import get_lr_scheduler, get_optimizer


logger = get_logger(__name__)
register_resolvers()


def train_no_val(args):
    validate_train(args)
    init_time = time.time()

    args.checkpointing.state_path = "{:}/{:}".format(args.checkpointing.state_path, args.name)

    accelerator = get_accelerator(args)
    logger.info("Starting training")

    mae = MAE(args)

    train_loader, _, len_train, _, len_train_sampler, _ = get_train_val_loaders(args, return_val=False)
    n_iter_per_train = len_train_sampler // (args.trainer.n_gpus * args.trainer.n_nodes)
    accelerator.print("Train segments:", len_train, "Train batches:", len(train_loader))
    accelerator.print(
        "N GPUS:",
        args.trainer.n_gpus,
        "Train iterations:",
        len_train_sampler,
        "acc steps:",
        args.trainer.accumulate_grad_batches,
    )
    mae = ensure_type(accelerator.prepare(mae), MAE)

    optimizer = get_optimizer(mae.parameters(), args.optimizer)
    scheduler = get_lr_scheduler(optimizer, args, n_iter_per_train)

    train_loader = ensure_type(accelerator.prepare(train_loader), DataLoader)
    optimizer = ensure_type(accelerator.prepare(optimizer), Optimizer)
    scheduler = ensure_type(accelerator.prepare(scheduler), AcceleratedScheduler)
    mae.train()

    if args.checkpointing.load_last_state:
        accelerator.load_state(pjoin(args.checkpointing.state_path, "last"))

    pbar = tqdm(range(args.trainer.epochs))

    for epoch in pbar:
        start = time.time()
        loss_ema = None
        for batch_idx, (x, pos, b_m, b_u) in enumerate(train_loader):
            with accelerator.accumulate(mae), accelerator.autocast():
                optimizer.zero_grad()
                loss = mae(x, pos, b_m, b_u)
                accelerator.backward(loss)
                if args.trainer.grad_clip and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(mae.parameters(), args.trainer.grad_clip_norm)
                optimizer.step()

            loss_g = loss.item()
            loss_ema = loss_g if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_g

            scheduler.step()

            pbar.set_description(
                "Epoch {:3d} (it. {:3d}/{:3d}) Loss (EMA/loss):{:3.3f}/{:3.3f} "
                "LR {:.8f}, shape {:3d}, time {:3.1f}".format(
                    epoch,
                    batch_idx,
                    n_iter_per_train,
                    loss_ema,
                    loss_g,
                    optimizer.param_groups[0]["lr"],
                    x.shape[0],
                    time.time() - start,
                ),
            )

            if args.wandb.log:
                accelerator.log({"epoch": epoch, "it": batch_idx, "loss_ema": loss_ema, "loss": loss_g})

        save_state(accelerator, args, epoch)

    reve_encoder = accelerator.unwrap_model(mae).encoder
    accelerator.save(reve_encoder.state_dict(), "encoder.pth")
    accelerator.print(f"Saved to {os.path.abspath('encoder.pth')}")
    logger.info(f"Saved to {os.path.abspath('encoder.pth')}")

    accelerator.print("Training took", time.time() - init_time, "seconds")
    accelerator.end_training()


@hydra.main(version_base=None, config_name="config_train", config_path="configs")
def main(args):
    if args.mode in ["debug", "train"]:
        train_no_val(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
