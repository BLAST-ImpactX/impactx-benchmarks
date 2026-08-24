#!/usr/bin/env bash
# FAITHFUL run manifest -- xsuite-cuda-dp-fm / fodo_exact / n=4000000
#   status: supported   layout: 1r x 1t   env: xsuite-gpu   device: cuda   precision: double
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export BENCH_FASTMATH=1
export CUDA_VISIBLE_DEVICES=0
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/axel/src/impactx-benchmarks

exec pixi run --environment xsuite-gpu python runs/axel-dell/xsuite-cuda-dp-fm__fodo_exact__n4000000/xsuite__fodo_exact.py
