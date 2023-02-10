import numpy as np


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length


def assign_learning_rate(optimizer, new_lr):
    if isinstance(optimizer, dict):
        optimizer["lr"] = new_lr
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


class Scheduler:
    def step(self):
        return NotImplementedError

    def current_lr(self):
        # Currently assumes all param_groups have the same learning rate
        return self.optimizer.param_groups[0]["lr"]


class CosineWithWarmup(Scheduler):
    def __init__(self, optimizer, learning_rate, total_steps, warmup_length=0) -> None:
        super().__init__()

        self.optimizer = optimizer
        self.base_lrs = learning_rate
        self.warmup_length = warmup_length
        self.total_steps = total_steps

        if not isinstance(self.base_lrs, list):
            self.base_lrs = [self.base_lrs for _ in optimizer.param_groups]

        assert len(self.base_lrs) == len(optimizer.param_groups)

    def step(self, iteration):
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if iteration < self.warmup_length:
                lr = _warmup_lr(base_lr, self.warmup_length, iteration)
            else:
                e = iteration - self.warmup_length
                es = self.total_steps - self.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr

            assign_learning_rate(param_group, lr)

        return lr


class ConstantWithWarmup(Scheduler):
    def __init__(self, optimizer, learning_rate, total_steps, warmup_length=0) -> None:
        super().__init__()

        self.optimizer = optimizer
        self.base_lrs = learning_rate
        self.warmup_length = warmup_length
        self.total_steps = total_steps

        if not isinstance(self.base_lrs, list):
            self.base_lrs = [self.base_lrs for _ in optimizer.param_groups]

        assert len(self.base_lrs) == len(optimizer.param_groups)

    def current_lr(self):
        # Currently assumes all param_groups have the same learning rate
        return self.optimizer.param_groups[0]["lr"]

    def step(self, iteration):
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if iteration < self.warmup_length:
                lr = _warmup_lr(base_lr, self.warmup_length, iteration)
            else:
                lr = base_lr

            assign_learning_rate(param_group, lr)

        return lr
