import logging
import sys

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig, open_dict

from avr.config import register_omega_conf_resolvers
from avr.wandb import WandbClient
from avr.wandbapi.filter import Filters

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


@hydra.main(config_path="../../config", config_name="default", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)

    # cv2.setNumThreads(0)
    pl.seed_everything(cfg.seed)
    module = instantiate(cfg["avr"]["task"][cfg.avr.problem][cfg.avr.dataset], cfg)

    client = WandbClient(cfg.wandb_project)

    if "find_pretraining_checkpoint" in cfg:
        filter = Filters.from_find_pretraining_checkpoint(cfg)
        run_names = client.get_run_names(filter)
        if len(run_names) == 1:
            with open_dict(cfg):
                cfg.wandb_checkpoint = run_names[0]
            print(
                f"Found pretraining checkpoint {cfg.wandb_checkpoint} with filter {filter}"
            )
        else:
            raise ValueError(
                f"Query to find a pretraining checkpoint with filter {filter} returned more than 1 result: {run_names}"
            )

    if "wandb_checkpoint" in cfg:
        checkpoint_path = client.get_local_checkpoint_by_run_name(cfg.wandb_checkpoint, cfg.log_dir)
        if checkpoint_path is not None:
            print(f"Found checkpoint locally: {checkpoint_path}")

        if checkpoint_path is None:
            checkpoint_path = client.download_checkpoint_by_run_name(cfg.wandb_checkpoint)
            if checkpoint_path is not None:
                print(f"Downloaded checkpoint from wandb: {checkpoint_path}")

        if checkpoint_path is None:
            raise ValueError(f"Checkpoint for run name not found: {cfg.wandb_checkpoint}")

        module = module.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg,
            strict=False,
            **{
                k: v
                for k, v in cfg["avr"]["task"][cfg.avr.problem][cfg.avr.dataset].items()
                if k != "_target_"
            },
        )

    if cfg.torch.compile:
        module = torch.compile(module)

    data_module = instantiate(cfg.avr.datamodule, cfg)
    trainer: pl.Trainer = instantiate(cfg.pytorch_lightning.trainer)
    trainer.fit(module, data_module)
    trainer.test(module, data_module)


if __name__ == "__main__":
    register_omega_conf_resolvers()
    load_dotenv()
    main()
