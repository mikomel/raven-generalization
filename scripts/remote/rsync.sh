#!/usr/bin/env bash

rsync -vazP --exclude-from .dockerignore . remote:~/projects/raven-generalization
