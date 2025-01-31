"""Optimizer and scheduler factories, including trapezoid LR schedule wrapping."""

import hydra
import torch


class CyclicTrapezoidLR(torch.optim.lr_scheduler._LRScheduler):
    """
    A cyclic trapezoidal learning rate schedule with distinct start, peak, and end LRs.
    """

    def __init__(  # noqa: PLR0913
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        plateau_steps: int,
        cooldown_steps: int,
        start_lr: float,
        peak_lr: float,
        end_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.cooldown_steps = cooldown_steps
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr

        # Correct total steps in one cycle
        self.steps_per_cycle = warmup_steps + plateau_steps + cooldown_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        step = self.last_epoch
        cycle_step = step % self.steps_per_cycle
        cycle_start_lr = self.start_lr if step < self.steps_per_cycle else self.end_lr

        if cycle_step < self.warmup_steps:
            # Phase 1: warm-up from start_lr -> peak_lr
            progress = cycle_step / self.warmup_steps
            lr = cycle_start_lr + (self.peak_lr - cycle_start_lr) * progress
        elif cycle_step < self.warmup_steps + self.plateau_steps:
            # Phase 2: plateau at peak_lr
            lr = self.peak_lr
        elif cycle_step < self.steps_per_cycle:
            # Phase 3: cool-down from peak_lr -> end_lr
            cooled_step = cycle_step - (self.warmup_steps + self.plateau_steps)
            progress = cooled_step / self.cooldown_steps
            lr = self.peak_lr - (self.peak_lr - self.end_lr) * progress
        else:
            lr = self.end_lr

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.get_lr()


def wrap_trapz_lr(args, n_iter):
    # Convert the relative steps to absolute steps
    if args["_target_"] == "utils.optim.CyclicTrapezoidLR":
        assert args.warmup_steps + args.cooldown_steps + args.plateau_steps == 1, "Should sum to 1"

        w_steps = int(args.warmup_steps * n_iter)
        c_steps = int(args.cooldown_steps * n_iter)
        p_steps = n_iter - w_steps - c_steps

        args.warmup_steps = w_steps
        args.plateau_steps = p_steps
        args.cooldown_steps = c_steps
    return args


####################################################################################################


def get_optimizer(parameters, args):
    """Excpects args to be a dictionary with the following keys:
    - _target_: The target class to instantiate
    - **kwargs: The keyword arguments to pass to the target class
    """
    opt = hydra.utils.instantiate(args, params=parameters)
    return opt


def get_lr_scheduler(optimizer, config, n_iter):
    """Excpects config.scheduler to be a dictionary with the following keys:
    - _target_: The target class to instantiate
    - **kwargs: The keyword arguments to pass to the target class
    """
    args = config.scheduler
    args = wrap_trapz_lr(args, n_iter)
    return hydra.utils.instantiate(args, optimizer=optimizer)
