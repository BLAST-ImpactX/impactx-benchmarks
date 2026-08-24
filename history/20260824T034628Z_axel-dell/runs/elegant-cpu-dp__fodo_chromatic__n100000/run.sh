#!/usr/bin/env bash
# FAITHFUL run manifest -- elegant-cpu-dp / fodo_chromatic / n=100000
#   status: supported   layout: 4r x 1t   env: elegant   device: cpu   precision: double
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

exec pixi run --environment elegant python /home/axel/src/impactx-benchmarks/codes/elegant/driver.py runs/axel-dell/elegant-cpu-dp__fodo_chromatic__n100000/elegant__fodo_chromatic.ele --ranks 4 --threads 1 --cpus 0,2,4,6 --device cpu
