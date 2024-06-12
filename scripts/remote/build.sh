#!/bin/bash

#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0-2:00:00
#SBATCH --export=ALL

set -ex

timestamp="$(date +'%Y-%m-%d_%H-%M-%S')"
DOCKER_BUILD_DIR='/vagrant'
DOCKER_FILE_PATH='/vagrant/docker/nvidia.Dockerfile'
DOCKER_IMAGE_URI='mikomel/raven-generalization:latest'
OUTPUT_DIR_HOST='/raid/shared/mikomel'
OUTPUT_DIR_GUEST='/output'
OUTPUT_FILENAME="mikomel-raven-generalization_${timestamp}.tar"
SINGULARITY_CONTAINER_PATH="/home/mikomel/singularity/mikomel-raven-generalization_${timestamp}.sif"

env \
  VAGRANT_EXPERIMENTAL=1 \
  DOCKER_BUILD_DIR="${DOCKER_BUILD_DIR}" \
  DOCKER_FILE_PATH="${DOCKER_FILE_PATH}" \
  DOCKER_IMAGE_URI="${DOCKER_IMAGE_URI}" \
  OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST}" \
  OUTPUT_DIR_GUEST="${OUTPUT_DIR_GUEST}" \
  OUTPUT_FILENAME="${OUTPUT_FILENAME}" \
  vagrant up --provision
echo "Provisioned virtual machine and saved docker image to: ${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"

vagrant suspend
echo "Suspended virtual machine"

singularity build "${SINGULARITY_CONTAINER_PATH}" "docker-archive://${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"
echo "Created new singularity container ${SINGULARITY_CONTAINER_PATH} from ${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"
