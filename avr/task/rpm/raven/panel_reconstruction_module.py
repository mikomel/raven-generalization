from typing import Optional, Dict

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from avr.model.rule_predictor import RulePredictor, JointTargetRulePredictor
from avr.task.rpm.raven.module import RavenModule


class RavenPanelReconstructionModule(RavenModule):
    def __init__(
            self,
            cfg: DictConfig,
            rule_predictor: RulePredictor,
            joint_target_rule_predictor: JointTargetRulePredictor = JointTargetRulePredictor.LINEAR,
            rules_loss_scale: float = 1.0,
            reconstruction_loss_scale: float = 1.0,
    ):
        super().__init__(cfg, rule_predictor, joint_target_rule_predictor, rules_loss_scale)
        self.reconstruction_loss_scale = reconstruction_loss_scale

    def _step(
            self, split: str, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        rules_loss_scaling = 1.0
        context, answers, y, rules = batch

        (
            combined_reconstruction,
            reconstruction,
            mask,
            slots,
            attn,
            embedding,
        ) = self.model(context, answers)

        images = torch.cat([context, answers], dim=1)
        reconstruction_loss = F.mse_loss(combined_reconstruction, images)
        self.log_reconstruction_metrics(split, reconstruction_loss, dataloader_idx)

        y_hat = self.target_pred_head(embedding)
        target_loss = self.target_loss(y_hat, y)
        acc = self.log_target_metrics(split, y, y_hat, target_loss, dataloader_idx)

        rules_hat = self.rule_pred_head(embedding)
        rules_loss = F.binary_cross_entropy_with_logits(rules_hat, rules)
        self.log_auxiliary_metrics(split, rules, rules_hat, rules_loss, dataloader_idx)

        loss = (
                self.reconstruction_loss_scale * reconstruction_loss
                + target_loss
                + rules_loss_scaling * rules_loss
        )
        if dataloader_idx is not None:
            configuration = self.get_test_configuration(dataloader_idx).short_name()
            self.logm_configuration(loss, "loss", split, configuration)
        else:
            self.logm(loss, "loss", split)
        return {"loss": loss, "acc": acc}

    def log_reconstruction_metrics(
            self,
            split: str,
            loss: torch.Tensor,
            dataloader_idx: Optional[int] = None,
    ):
        if dataloader_idx is not None:
            configuration = self.get_configuration(split, dataloader_idx).short_name()
            self.logm_configuration_type(
                loss, "loss", split, configuration, "reconstruction"
            )
        else:
            self.logm_type(loss, "loss", split, "reconstruction")
