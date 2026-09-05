#!/usr/bin/env bash
# FAITHFUL run manifest -- impactx-cuda-dp / fodo_chromatic / n=16000000
#   status: failed   layout: 1r x 1t   env: impactx-cuda-dp   device: cuda   precision: double
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

exec pixi run --environment impactx-cuda-dp python runs/perlmutter/impactx-cuda-dp__fodo_chromatic__n16000000/impactx__fodo_chromatic.py
