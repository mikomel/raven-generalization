# Generalization and Knowledge Transfer in Abstract Visual Reasoning Models

This repository provides implementation of methods and experiments from:
Małkiński, Mikołaj, and Jacek Mańdziuk. "Generalization and Knowledge Transfer in Abstract Visual Reasoning Models." Preprint. Under review. (2024).

Relevant links:
* Main project page: https://github.com/mikomel/raven
* Dataset generator: https://github.com/mikomel/raven-datasets
* Attributeless-I-RAVEN: https://huggingface.co/datasets/mikmal/attributeless-i-raven
* I-RAVEN-Mesh: https://huggingface.co/datasets/mikmal/i-raven-mesh

## Local setup

The project was implemented in Python 3.9.
Dependencies are listed in `requirements.txt`.

The recommended way to run scripts from the repository is to use Docker.
The Docker image can be built locally with:
```bash
./scripts/local/build.sh
```

Code was formatted with [Black](https://black.readthedocs.io/en/stable/).
It can be installed with:
```bash
pip install black==24.4.2
```

## Cluster setup

Experiments reported in the paper were run on a computing cluster.
The `scripts/remote/` directory provides a set of scripts that may be used to run experiments on a cluster.
Paths to files and directories in the corresponding scripts may have to be adjusted.

Sync project files with a remote host:
```bash
./scripts/remote/rsync.sh
```

Build [singularity](https://github.com/sylabs/singularity) container:
```bash
sbatch ./scripts/remote/build.sh
```

Run a training job:
```bash
sbatch ./scripts/remote/train.sh
```

## Tests

Unit tests can be run locally in the Docker container with:
```bash
./scripts/local/test.sh
```
