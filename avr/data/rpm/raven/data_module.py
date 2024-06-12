from typing import List, Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from avr.data.rpm.raven.dataset import RavenDataset


class RavenDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(RavenDataModule, self).__init__()
        self.cfg: DictConfig = cfg
        self.train_dataset: RavenDataset = None
        self.val_datasets: List[RavenDataset] = []
        self.test_datasets: List[RavenDataset] = []

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = instantiate(self.cfg.avr.data.rpm.raven.train)
        self.val_datasets = [
            instantiate(self.cfg.avr.data.rpm.raven.val, configurations=[configuration])
            for configuration in self.cfg.avr.data.rpm.raven.val.configurations
        ]
        self.test_datasets = [
            instantiate(self.cfg.avr.data.rpm.raven.test, configurations=[configuration])
            for configuration in self.cfg.avr.data.rpm.raven.test.configurations
        ]

    def train_dataloader(self) -> DataLoader:
        return instantiate(self.cfg.torch.data_loader.train, dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.val, dataset=dataset)
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.test, dataset=dataset)
            for dataset in self.test_datasets
        ]
