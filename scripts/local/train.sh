#!/usr/bin/env bash

date
echo "Training command: python avr/experiment/train.py" "${@}"
docker run \
  -v ~/Projects/raven-generalization:/app \
  -v ~/Datasets:/app/data:ro \
  --rm \
  --ipc host \
  --gpus all \
  --entrypoint python \
  mikomel/raven-generalization:latest \
  "avr/experiment/train.py" "${@}"
date
