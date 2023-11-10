from functools import partial
import torch


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


class LinearScheduleWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 1000,
        num_training_steps: int = 100000,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        lr_lambda = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch,
            verbose=verbose,
        )
