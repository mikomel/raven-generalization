from functools import partial
from types import ModuleType
from typing import Dict, List, Iterable, Optional, Union, Callable

from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from torch import nn

from avr.data.dataset import DatasetSplit
from avr.data.rpm.pgm.rule_encoder import create_pgm_rule_encoder
from avr.data.rpm.raven.configuration import RavenConfiguration
from avr.data.rpm.raven.rule_encoder import create_raven_rule_encoder
from avr.model.rule_predictor import (
    RulePredictor,
    SplitImagesRulePredictor,
    JointTargetRulePredictor,
)
from avr.model.target_predictor import SplitImagesTargetPredictor


def compose_config(
        config_path: str = "../config",
        config_name: str = "default",
        overrides: List[str] = (),
) -> DictConfig:
    with initialize(config_path=config_path):
        return compose(config_name=config_name, overrides=overrides)


def get_resolvers() -> Dict[str, Callable]:
    return {
        "List": lambda *args: list(args),
        "Tuple": lambda *args: tuple(args),
        "DatasetSplit": lambda x: DatasetSplit[x],
        "RavenConfiguration": lambda x: RavenConfiguration[x],
        "RavenRuleEncoder": create_raven_rule_encoder,
        "PGMRuleEncoder": create_pgm_rule_encoder,
        "RulePredictor": lambda x: RulePredictor[x],
        "SplitImagesTargetPredictor": lambda x: SplitImagesTargetPredictor[x],
        "SplitImagesRulePredictor": lambda x: SplitImagesRulePredictor[x],
        "JointTargetRulePredictor": lambda x: JointTargetRulePredictor[x],
        "nn": create_torch_nn_module,
    }


def register_omega_conf_resolvers():
    for name, resolver in get_resolvers().items():
        OmegaConf.register_new_resolver(name, resolver)


def resolve(module: ModuleType, class_name: str, *args) -> object:
    class_ = getattr(module, class_name)
    kwargs = to_kwargs(*args)
    return class_(**kwargs)


def create_torch_nn_module(*args) -> Callable[..., nn.Module]:
    kwargs = to_kwargs(*args[1:])
    return partial(getattr(nn, args[0]), **kwargs)


def to_kwargs(*args) -> Dict:
    return {k: v for k, v in pairwise(args)}


def pairwise(iterable: Iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    iterator = iter(iterable)
    return zip(iterator, iterator)


def safe_dict_get(d: Union[Dict, DictConfig], key: str) -> Optional:
    value = d
    for node in key.split("."):
        if value:
            value = value.get(node)
    return value
