#!/bin/bash -l

# Copyright 2021-2025 Axel Huebl
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -J napac25
#    note: <proj> must end on _g
#SBATCH -A m4272_g
#S BATCH -q regular
#SBATCH -q premium  # note: needs premium QOS for the project -A in IRIS
# A100 40GB (most nodes, can include 80GB GPUs)
#S BATCH -C gpu
# A100 80GB (256 nodes)
#SBATCH -C gpu&hbm80g
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
# ideally single:1, but NERSC cgroups issue
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-node=1  # note: actually 4
#SBATCH --gpus-per-node=1    # note: actually 4
#SBATCH -o bench_logs/permutter.o%j
#SBATCH -e bench_logs/permutter.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=axelhuebl@lbl.gov

module load conda/Miniforge3-24.11.3-0

conda env remove --solver libmamba -q -y -n benchmark-base || true
conda env create --solver libmamba -q -y -n benchmark-base -f benchmark-base.yaml
conda activate benchmark-base

# do not use ccache
export CCACHE_DISABLE=1

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun --cpu-bind=cores ./bench_local.sh
