from typing import Dict, Any

from omegaconf import DictConfig

from avr.config import safe_dict_get


class Filters:
    @staticmethod
    def from_find_pretraining_checkpoint(
            cfg: DictConfig,
    ) -> Dict[str, Any]:
        filter = dict()
        for key in cfg.find_pretraining_checkpoint:
            if "=" in key:
                key, value = key.split("=")
                if value == "NOT_EXISTS":
                    filter[f"config.{key}"] = {"$exists": False}
                else:
                    filter[f"config.{key}"] = value
            else:
                filter[f"config.{key}"] = safe_dict_get(cfg, key)
        return filter
