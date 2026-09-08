#!/usr/bin/env bash
# FAITHFUL run manifest -- impactz-cpu-dp / fodo_chromatic / n=1000000
#   status: supported   layout: 4r x 1t   env: impactz   device: cpu   precision: double
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

exec pixi run --environment impactz python /home/axel/src/impactx-benchmarks/codes/impactz/driver.py runs/axel-dell/impactz-cpu-dp__fodo_chromatic__n1000000/impactz__fodo_chromatic.in --ranks 4 --threads 1 --cpus 0,2,4,6 --device cpu
