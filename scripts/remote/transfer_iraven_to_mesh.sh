#!/usr/bin/env bash

set -e

pretrain () {
  sbatch --parsable ./scripts/remote/train.sh raven_dataset_name=I-RAVEN wandb_log_model=True "${@}"
}

finetune () {
  PRETRAINING_JOB_ID="${1}"
  for limit_train_batches in 1.0 0.5 0.25 0.125 0.0625; do
    PRETRAINING_JOB_ID="${PRETRAINING_JOB_ID}" sbatch --parsable --dependency=afterok:${PRETRAINING_JOB_ID} ./scripts/remote/train.sh raven_dataset_name=I-RAVEN-Mesh wandb_log_model=False +pytorch_lightning.trainer.limit_train_batches=${limit_train_batches} "${@:2}"
  done
  PRETRAINING_JOB_ID="${PRETRAINING_JOB_ID}" sbatch --parsable --dependency=afterok:${PRETRAINING_JOB_ID} ./scripts/remote/train.sh raven_dataset_name=I-RAVEN-Mesh wandb_log_model=False +pytorch_lightning.trainer.limit_train_batches=0.03125 num_workers=2 "${@:2}"
  PRETRAINING_JOB_ID="${PRETRAINING_JOB_ID}" sbatch --parsable --dependency=afterok:${PRETRAINING_JOB_ID} ./scripts/remote/train.sh raven_dataset_name=I-RAVEN-Mesh wandb_log_model=False +pytorch_lightning.trainer.limit_train_batches=0.015625 num_workers=1 "${@:2}"
}

PRETRAINING_JOB_ID=$(pretrain "${@}")
echo "Pretraining job id: ${PRETRAINING_JOB_ID}"

FINETUNING_JOB_ID=$(finetune "${PRETRAINING_JOB_ID}" "${@}")
echo "Finetuning job id: ${FINETUNING_JOB_ID}"
