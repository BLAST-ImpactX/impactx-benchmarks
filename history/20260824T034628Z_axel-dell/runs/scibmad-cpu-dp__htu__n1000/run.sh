#!/usr/bin/env bash
# FAITHFUL run manifest -- scibmad-cpu-dp / htu / n=1000
#   status: supported   layout: 1r x 2t   env: scibmad   device: cpu   precision: double
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export BENCH_FASTMATH=0
export CUDA_VISIBLE_DEVICES=''
export JULIA_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export PYTHONPATH=/home/axel/src/impactx-benchmarks

exec pixi run --environment scibmad taskset -c 0,2,4,6 julia -t2 --project=codes/scibmad runs/axel-dell/scibmad-cpu-dp__htu__n1000/scibmad__htu.jl
