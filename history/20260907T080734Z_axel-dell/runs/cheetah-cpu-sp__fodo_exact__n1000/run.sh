#!/usr/bin/env bash
# FAITHFUL run manifest -- cheetah-cpu-sp / fodo_exact / n=1000
#   status: supported   layout: 1r x 1t   env: cheetah   device: cpu   precision: single
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export BENCH_FASTMATH=0
export CUDA_VISIBLE_DEVICES=''
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/axel/src/impactx-benchmarks

exec pixi run --environment cheetah taskset -c 0,2,4,6 python runs/axel-dell/cheetah-cpu-sp__fodo_exact__n1000/cheetah__fodo_exact.py
