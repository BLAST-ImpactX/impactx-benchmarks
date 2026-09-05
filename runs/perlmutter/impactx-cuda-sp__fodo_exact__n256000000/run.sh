#!/usr/bin/env bash
# FAITHFUL run manifest -- impactx-cuda-sp / fodo_exact / n=256000000
#   status: failed   layout: 1r x 1t   env: impactx-cuda-sp   device: cuda   precision: single
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
export PYTHONPATH=/pscratch/sd/a/ahuebl/impactx-benchmarks:/opt/nersc/pymon

exec pixi run --environment impactx-cuda-sp python runs/perlmutter/impactx-cuda-sp__fodo_exact__n256000000/impactx__fodo_exact.py
