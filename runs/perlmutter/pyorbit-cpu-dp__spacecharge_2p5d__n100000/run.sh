#!/usr/bin/env bash
# FAITHFUL run manifest -- pyorbit-cpu-dp / spacecharge_2p5d / n=100000
#   status: supported   layout: 16r x 1t   env: pyorbit   device: cpu   precision: double
# The input file(s) in this folder are the template-resolved script that ran; the
# command below is exactly how the harness launched it (auto-written by
# benchmarks/runner.py for author review / reproduction). Run from the repo root.
# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.
set -eu

export BENCH_FASTMATH=0
export CUDA_VISIBLE_DEVICES=''
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=1
export PYTHONPATH=/pscratch/sd/a/ahuebl/impactx-benchmarks:/opt/nersc/pymon

exec pixi run --environment pyorbit mpirun -np 16 -bind-to none /pscratch/sd/a/ahuebl/impactx-benchmarks/codes/pin_rank.sh 1 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 python runs/perlmutter/pyorbit-cpu-dp__spacecharge_2p5d__n100000/pyorbit__spacecharge_2p5d.py
