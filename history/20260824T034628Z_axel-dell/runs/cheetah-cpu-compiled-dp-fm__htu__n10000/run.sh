#!/usr/bin/env bash
# FAITHFUL run manifest -- cheetah-cpu-compiled-dp-fm / htu / n=10000
#   status: supported   layout: 1r x 4t   env: cheetah   device: cpu   precision: double
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export BENCH_FASTMATH=1
export CUDA_VISIBLE_DEVICES=''
export JULIA_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH=/home/axel/src/impactx-benchmarks
export TORCHINDUCTOR_USE_FAST_MATH=1

exec pixi run --environment cheetah taskset -c 0,2,4,6 python runs/axel-dell/cheetah-cpu-compiled-dp-fm__htu__n10000/cheetah__htu.py
