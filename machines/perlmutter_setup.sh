#!/bin/bash -l
# One-time Perlmutter (NERSC) setup: build ALL benchmark environments (CPU + GPU) with pixi.
#
# Run this on a LOGIN node -- compute nodes have no outbound internet for the conda / pip / git
# downloads (source clones of ImpactX, PyORBIT, Bmad, SciBmad). It is a good login-node citizen:
# build parallelism is capped (BUILD_NPROC). The from-source codes are compiled for Perlmutter's
# Zen3 microarchitecture (-march=znver3) with fast-math ON (the harness default); the ImpactX CUDA
# build auto-detects sm_80 (A100) even without a GPU present.
#
#   bash machines/perlmutter_setup.sh
#
# Then submit the run jobs (see machines/PERLMUTTER.md):
#   cpu=$(sbatch --parsable machines/perlmutter_cpu.sbatch)
#   sbatch --dependency=afterok:$cpu machines/perlmutter_gpu.sbatch
set -eu
cd "$(git rev-parse --show-toplevel)"

# Build-time knobs (see codes/*/build.sh). znver3 = AMD EPYC 7763 (Milan/Zen3); fast-math ON.
export BENCH_ARCH="${BENCH_ARCH:-znver3}"
export BENCH_FASTMATH="${BENCH_FASTMATH:-1}"
export BUILD_NPROC="${BUILD_NPROC:-16}"   # cap parallel compiles on the shared login node

echo "== pixi install (materialize all environments) =="
pixi install

echo "== build from-source codes for CPU (impactx OMP+SIMD, pyat, pyorbit, bmad, scibmad, ...) =="
pixi run -e default python -m benchmarks.build --device cpu

echo "== build from-source codes for GPU (impactx CUDA sm_80 dp/sp, scibmad-gpu; pip envs materialized) =="
pixi run -e default python -m benchmarks.build --device cuda

echo
echo "Perlmutter environments built (CPU + GPU), arch=${BENCH_ARCH}, fast-math=${BENCH_FASTMATH}."
echo "Next: submit machines/perlmutter_cpu.sbatch then machines/perlmutter_gpu.sbatch (see PERLMUTTER.md)."
