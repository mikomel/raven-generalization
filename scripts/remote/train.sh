#!/usr/bin/env bash

#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=short
#SBATCH --export=ALL,HYDRA_FULL_ERROR=1

date
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "singularity version: $(singularity version)"
echo "nvidia-container-toolkit version: $(nvidia-container-toolkit -version)"
echo "nvidia-container-cli info: $(nvidia-container-cli info)"

if [[ -v PRETRAINING_JOB_ID ]]; then
  echo "PRETRAINING_JOB_ID was set to: ${PRETRAINING_JOB_ID}"
  wandb_run_name=$(cat "slurm-${PRETRAINING_JOB_ID}.out" | grep -oP 'wandb: Syncing run \K.+$')
  echo "Will load checkpoint from wandb run: ${wandb_run_name}"
  train_args="${@} +wandb_checkpoint=${wandb_run_name}"
else
  train_args="${@}"
fi

echo "Training command: python avr/experiment/train.py ${train_args}"

singularity run \
  --nv \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --bind /home/mikomel/projects/raven-generalization/config:/app/config:ro \
  --bind /home/mikomel/projects/raven-generalization/avr:/app/avr:ro \
  --bind /home/mikomel/datasets:/app/data:ro \
  --bind /home/mikomel/logs:/app/logs:rw \
  ~/singularity/mikomel-raven-generalization-latest.sif \
  python avr/experiment/train.py $train_args
date
