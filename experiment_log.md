# Experiment log

This file provides commands to reproduce all experiments from the paper in single-task learning (STL) and transfer learning (TL) settings.

## STL: Context-blind ResNet on I-RAVEN, I-RAVEN-Mesh and Attributeless-I-RAVEN

```bash
for seed in 42 123 1001; do
  for ds_name in "I-RAVEN" "I-RAVEN-Mesh" "I-RAVEN-attributeless-color" "I-RAVEN-attributeless-position" "I-RAVEN-attributeless-size" "I-RAVEN-attributeless-type"; do
    for model in "cb_resnet18" "cb_resnet50"; do
      sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="${model}" seed="${seed}" raven_dataset_name="${ds_name}"
    done
  done
done
```

## STL: Attributeless-I-RAVEN

```bash
for seed in 42 123 1001; do
  for ds_name in "I-RAVEN-attributeless-color" "I-RAVEN-attributeless-position" "I-RAVEN-attributeless-size" "I-RAVEN-attributeless-type"; do
    for model in "cnnlstm" "copinet" "cpcnet" "predrnet" "relbase" "scl" "sran" "wren"; do
      sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="${model}" seed="${seed}" raven_dataset_name="${ds_name}"
    done
    sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="stsn" seed="${seed}" raven_dataset_name="${ds_name}" avr/task/rpm/raven=panel_reconstruction batch_size=64 +pytorch_lightning.trainer.accumulate_grad_batches=2
    sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="pong" seed="${seed}" raven_dataset_name="${ds_name}" avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=5
  done
done
```

## STL: I-RAVEN-Mesh

```bash
ds_name="I-RAVEN-Mesh"
for seed in 42 123 1001; do
  # Baselines without STSN
  for model in "cnnlstm" "copinet" "cpcnet" "predrnet" "relbase" "scl" "sran" "wren"; do
    for limit_train_batches in 1.0 0.5 0.25 0.125 0.0625; do
      sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="${model}" seed="${seed}" raven_dataset_name="${ds_name}" +pytorch_lightning.trainer.limit_train_batches="${limit_train_batches}"
    done
    sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="${model}" seed="${seed}" raven_dataset_name="${ds_name}" +pytorch_lightning.trainer.limit_train_batches=0.03125 num_workers=2
    sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="${model}" seed="${seed}" raven_dataset_name="${ds_name}" +pytorch_lightning.trainer.limit_train_batches=0.015625 num_workers=1
  done
  # STSN
  for limit_train_batches in 1.0 0.5 0.25 0.125 0.0625; do
    sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="stsn" seed="${seed}" raven_dataset_name="${ds_name}" avr/task/rpm/raven=panel_reconstruction batch_size=64 +pytorch_lightning.trainer.accumulate_grad_batches=2 +pytorch_lightning.trainer.limit_train_batches="${limit_train_batches}"
  done
  sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="stsn" seed="${seed}" raven_dataset_name="${ds_name}" avr/task/rpm/raven=panel_reconstruction batch_size=64 +pytorch_lightning.trainer.accumulate_grad_batches=2 +pytorch_lightning.trainer.limit_train_batches=0.03125 num_workers=2
  sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="stsn" seed="${seed}" raven_dataset_name="${ds_name}" avr/task/rpm/raven=panel_reconstruction batch_size=64 +pytorch_lightning.trainer.accumulate_grad_batches=2 +pytorch_lightning.trainer.limit_train_batches=0.015625 num_workers=1
  # PoNG
  for limit_train_batches in 1.0 0.5 0.25 0.125 0.0625; do
    sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="pong" seed="${seed}" raven_dataset_name="${ds_name}" avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=5 +pytorch_lightning.trainer.limit_train_batches="${limit_train_batches}"
  done
  sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="pong" seed="${seed}" raven_dataset_name="${ds_name}" avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=5 +pytorch_lightning.trainer.limit_train_batches=0.03125 num_workers=2
  sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="pong" seed="${seed}" raven_dataset_name="${ds_name}" avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=5 +pytorch_lightning.trainer.limit_train_batches=0.015625 num_workers=1
done
```

## TL: I-RAVEN to I-RAVEN-Mesh

Note: This experiment also provides the results of STL on I-RAVEN.

```bash
for seed in 42 123 1001; do
  for model in "cnnlstm" "copinet" "cpcnet" "predrnet" "relbase" "scl" "sran" "wren"; do
    ./scripts/remote/transfer_iraven_to_mesh.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="${model}" seed="${seed}"
  done
  ./scripts/remote/transfer_iraven_to_mesh.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="stsn" seed="${seed}" avr/task/rpm/raven=panel_reconstruction batch_size=64 +pytorch_lightning.trainer.accumulate_grad_batches=2
  ./scripts/remote/transfer_iraven_to_mesh.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="pong" seed="${seed}" avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=5
done
```

## STL: PoNG ablation study

```bash
run() {
  for raven_dataset_name in "I-RAVEN" "I-RAVEN-Mesh" "I-RAVEN-attributeless-color" "I-RAVEN-attributeless-position" "I-RAVEN-attributeless-size" "I-RAVEN-attributeless-type"; do
    for seed in 402 502 602; do
      sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=raven avr/model="pong" seed="${seed}" raven_dataset_name="${raven_dataset_name}" ${@}
    done
  done
}

# w/o P3 and P4
run avr.model.reasoner_use_row_group_conv=False avr.model.reasoner_use_row_pair_group_conv=False avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=5
# w/o TCN
run avr.model.reasoner_group_conv_use_norm=False avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=5
# $\gamma=0$
run avr.task.rpm.raven.rules_loss_scale=25 avr.task.rpm.raven.joint_target_rule_loss_scale=0
# $\beta=0$
run avr.task.rpm.raven.rules_loss_scale=0 avr.task.rpm.raven.joint_target_rule_loss_scale=5
# $\gamma=0 \land \beta=0$
run avr.task.rpm.raven.rules_loss_scale=0 avr.task.rpm.raven.joint_target_rule_loss_scale=0
# union
run avr.model.reasoner_use_row_group_conv=False avr.model.reasoner_use_row_pair_group_conv=False avr.model.reasoner_group_conv_use_norm=False avr.task.rpm.raven.rules_loss_scale=0 avr.task.rpm.raven.joint_target_rule_loss_scale=0
```

## STL: PoNG on PGM

```bash
seed=402
for regime in "neutral" "extrapolation" "interpolation" "attr.rel.pairs" "attr.rels" "attrs.pairs" "attrs.line.type" "attrs.shape.color"; do
  sbatch ./scripts/remote/train.sh pytorch_lightning/trainer=with_wandb_logger +experiment=pgm avr/model="pong" seed="${seed}" avr.task.rpm.pgm.rules_loss_scale=25 avr.task.rpm.pgm.joint_target_rule_loss_scale=5 avr.data.rpm.pgm.regime="${regime}"
done
```
