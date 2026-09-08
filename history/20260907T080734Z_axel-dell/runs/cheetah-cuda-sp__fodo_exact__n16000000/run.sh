#!/usr/bin/env bash
# FAITHFUL run manifest -- cheetah-cuda-sp / fodo_exact / n=16000000
#   status: supported   layout: 1r x 1t   env: cheetah-gpu   device: cuda   precision: single
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export BENCH_FASTMATH=0
export CUDA_VISIBLE_DEVICES=0
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/axel/src/impactx-benchmarks

exec pixi run --environment cheetah-gpu python runs/axel-dell/cheetah-cuda-sp__fodo_exact__n16000000/cheetah__fodo_exact.py
