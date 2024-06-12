from abc import ABC
from typing import Union, Optional, Callable, Any

import omegaconf
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer


class AVRModule(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        super(AVRModule, self).__init__()
        self.save_hyperparameters(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        self.cfg = cfg
        self.model = instantiate(cfg.avr.model)
        self.lr_warmup = instantiate(cfg.avr.train.lr_warmup)

    def configure_optimizers(self):
        optimizer: Optimizer = instantiate(
            self.cfg.torch.optimizer, params=self.parameters()
        )
        scheduler: object = instantiate(self.cfg.torch.scheduler, optimizer=optimizer)
        interval = (
            "step"
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
            else "epoch"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.cfg.monitor,
                "interval": interval,
            },
        }

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.lr_warmup.step(self)

    def on_train_start(self) -> None:
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, list):
            lr_scheduler = lr_scheduler[0]
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            num_steps_in_epoch = len(self.trainer.train_dataloader)
            total_num_steps = num_steps_in_epoch * self.cfg.max_epochs
            lr_scheduler.T_max = total_num_steps
            self.print(
                f"CosineAnnealingLR: "
                f"num_steps_in_epoch: {num_steps_in_epoch}, "
                f"max_epochs: {self.cfg.max_epochs}, "
                f"total_num_steps: {total_num_steps}, "
                f"lr_scheduler.T_max: {lr_scheduler.T_max}"
            )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def logm(self, value: torch.Tensor, metric: str, split: str):
        self.log(
            f"{split}/{metric}",
            value,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

    def logm_type(self, value: torch.Tensor, metric: str, split: str, type: str):
        self.log(
            f"{split}/{metric}/{type}",
            value,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

    def logm_configuration(
            self, value: torch.Tensor, metric: str, split: str, configuration: str
    ):
        self.log(
            f"{split}/{configuration}/{metric}",
            value,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
        )

    def logm_configuration_type(
            self,
            value: torch.Tensor,
            metric: str,
            split: str,
            configuration: str,
            type: str,
    ):
        self.log(
            f"{split}/{configuration}/{metric}/{type}",
            value,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
        )
