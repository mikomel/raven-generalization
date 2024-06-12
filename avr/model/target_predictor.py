from enum import Enum
from typing import Union

from torch import nn


class SingleImagesTargetPredictor(Enum):
    LINEAR = 'linear'
    MLP = 'mlp'

    def create(self, cfg, num_answers: int) -> nn.Module:
        d = cfg.avr.model.embedding_size

        if self == SingleImagesTargetPredictor.LINEAR:
            return nn.Linear(d, num_answers)

        elif self == SingleImagesTargetPredictor.MLP:
            d = cfg.avr.model.embedding_size
            return nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, num_answers))

        else:
            raise ValueError()


class SplitImagesTargetPredictor(Enum):
    LINEAR = 'linear'
    MLP = 'mlp'

    def create(self, cfg, num_answers: int) -> nn.Module:
        d = cfg.avr.model.embedding_size

        if self == SplitImagesTargetPredictor.LINEAR:
            return nn.Linear(d, 1)

        elif self == SplitImagesTargetPredictor.MLP:
            d = cfg.avr.model.embedding_size
            return nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, 1),
                nn.Flatten(-2, -1))

        else:
            raise ValueError()


TargetPredictor = Union[SingleImagesTargetPredictor, SplitImagesTargetPredictor]
