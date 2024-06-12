#!/usr/bin/env bash

date
echo "Command: python avr/experiment/test.py" "${@}"
docker run \
  -v ~/Projects/raven-generalization:/app \
  --rm \
  --ipc host \
  --gpus all \
  mikomel/raven-generalization:latest \
  "pytest" "${@}"
date
