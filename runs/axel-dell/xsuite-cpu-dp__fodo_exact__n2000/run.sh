#!/usr/bin/env bash
# FAITHFUL run manifest -- xsuite-cpu-dp / fodo_exact / n=2000
#   status: supported   layout: 1r x 20t   env: xsuite   device: cpu   precision: double
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export CUDA_VISIBLE_DEVICES=''
export JULIA_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OMP_NUM_THREADS=20
export PYTHONPATH=/home/axel/src/impactx-benchmarks

exec pixi run --environment xsuite taskset -c 0,2,4,6 python runs/axel-dell/xsuite-cpu-dp__fodo_exact__n2000/xsuite__fodo_exact.py
