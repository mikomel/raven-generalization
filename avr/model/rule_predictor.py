from enum import Enum
from typing import Union

from torch import nn

from avr.model.neural_blocks import Sum, MLP


class SingleImagesRulePredictor(Enum):
    MLP = "mlp"

    def create(self, cfg, num_rules: int, num_answers: int) -> nn.Module:
        d = cfg.avr.model.embedding_size

        if self == SingleImagesRulePredictor.MLP:
            return MLP(depth=2, in_dim=d, out_dim=num_rules)

        else:
            raise ValueError()


class SplitImagesRulePredictor(Enum):
    FLAT_LINEAR = "flat-linear"
    FLAT_ACT = "flat-act"
    ACT_FLAT_LINEAR = "act-flat-linear"
    ACT_FLAT_ACT = "act-flat-act"
    SUM_LINEAR = "sum-linear"
    SUM_ACT = "sum-act"
    ACT_SUM_LINEAR = "act-sum-linear"
    ACT_SUM_ACT = "act-sum-act"
    CONV_ACT_LINEAR = "conv-act-linear"
    TWICE_CONV_ACT_LINEAR = "twice-conv-act-linear"

    def create(self, cfg, num_rules: int, num_answers: int) -> nn.Module:
        d = cfg.avr.model.embedding_size
        p = num_answers
        r = num_rules

        if self == SplitImagesRulePredictor.FLAT_LINEAR:
            return nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1), nn.Linear(p * d, r)
            )

        if self == SplitImagesRulePredictor.FLAT_ACT:
            return nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(p * d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, r),
            )

        if self == SplitImagesRulePredictor.ACT_FLAT_LINEAR:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(p * d, r),
            )

        if self == SplitImagesRulePredictor.ACT_FLAT_ACT:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(p * d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, r),
            )

        elif self == SplitImagesRulePredictor.SUM_LINEAR:
            return nn.Sequential(Sum(dim=-2), nn.Linear(d, r))

        elif self == SplitImagesRulePredictor.SUM_ACT:
            return nn.Sequential(
                Sum(dim=-2), nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, r)
            )

        elif self == SplitImagesRulePredictor.ACT_SUM_LINEAR:
            return nn.Sequential(
                nn.Linear(d, d), nn.ReLU(inplace=True), Sum(dim=-2), nn.Linear(d, r)
            )

        elif self == SplitImagesRulePredictor.ACT_SUM_ACT:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                Sum(dim=-2),
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, r),
            )

        elif self == SplitImagesRulePredictor.CONV_ACT_LINEAR:
            return nn.Sequential(
                nn.Conv1d(p, 1, kernel_size=5, stride=1, padding=2),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.ReLU(inplace=True),
                nn.Linear(d, r),
            )

        elif self == SplitImagesRulePredictor.TWICE_CONV_ACT_LINEAR:
            return nn.Sequential(
                nn.Conv1d(p, p, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Conv1d(p, 1, kernel_size=5, stride=1, padding=2),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.ReLU(inplace=True),
                nn.Linear(d, r),
            )

        else:
            raise ValueError()


class JointTargetRulePredictor(str, Enum):
    LINEAR = "JointTargetRulePredictor.LINEAR"
    ACT_LINEAR = "JointTargetRulePredictor.ACT_LINEAR"

    def create(self, cfg, num_rules: int) -> nn.Module:
        d = cfg.avr.model.embedding_size
        r = num_rules

        if self == JointTargetRulePredictor.LINEAR:
            return nn.Linear(d, r)

        elif self == JointTargetRulePredictor.ACT_LINEAR:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, r),
            )

        else:
            raise ValueError()


RulePredictor = Union[SingleImagesRulePredictor, SplitImagesRulePredictor]
