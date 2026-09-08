#!/usr/bin/env bash
# FAITHFUL run manifest -- impactx-cpu-simd-sp-fm / htu_spin / n=10000
#   status: supported   layout: 4r x 1t   env: impactx-sp-fm   device: cpu   precision: single
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export BENCH_FASTMATH=1
export CUDA_VISIBLE_DEVICES=''
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/axel/src/impactx-benchmarks

exec pixi run --environment impactx-sp-fm mpirun -np 4 -bind-to none /home/axel/src/impactx-benchmarks/codes/pin_rank.sh 1 0,2,4,6 python runs/axel-dell/impactx-cpu-simd-sp-fm__htu_spin__n10000/impactx__htu_spin.py
