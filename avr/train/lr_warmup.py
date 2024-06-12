from abc import ABC, abstractmethod

import pytorch_lightning as pl


class LRWarmup(ABC):
    @abstractmethod
    def step(self, module: pl.LightningModule):
        pass


class NoLRWarmup(LRWarmup):
    def step(self, module: pl.LightningModule):
        pass


class LinearLRWarmup(LRWarmup):
    def __init__(self, num_steps: int = 500, initial_lr: float = 0.001):
        self.num_steps = num_steps
        self.initial_lr = initial_lr

    def step(self, module: pl.LightningModule):
        optimizers = module.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if module.trainer.global_step < self.num_steps:
            lr_scale = min(
                1.0, float(module.trainer.global_step + 1) / float(self.num_steps)
            )
            for optimizer in optimizers:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.initial_lr
        lr = optimizers[0].param_groups[0]["lr"]
        module.log("lr", lr, logger=True)
