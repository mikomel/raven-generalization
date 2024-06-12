#!/usr/bin/env bash

IMAGE_URI="mikomel/raven-generalization:latest"

docker build -t ${IMAGE_URI} -f docker/nvidia.Dockerfile .
