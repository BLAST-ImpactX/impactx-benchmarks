#!/bin/bash -l
# Perlmutter (NERSC) CPU benchmark job: one socket of a 2× AMD EPYC 7763 node, driven by pixi.
# Submit from the repo root AFTER machines/perlmutter_setup.sh has built the envs on a login node.
#   sbatch machines/perlmutter_cpu.sbatch
#
# NERSC allocation (the "_g" GPU-project suffix is optional/skippable on Perlmutter now).
#SBATCH -A m5125
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -J bench-cpu
#SBATCH -o bench-cpu.o%j
#SBATCH -e bench-cpu.e%j
#SBATCH --mail-type=END,FAIL
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# One EPYC socket = 64 physical cores. The harness pins every MPI/OpenMP layout to socket 0
# (BENCH_SOCKET) and auto-tunes the best split within the 64-core budget -- no hyperthread
# oversubscription, no cross-socket spill. Arch/fast-math were baked at build time (setup.sh);
# this is a --skip-build run. Same results file ("perlmutter") as the GPU job, so the GPU-FP32
# comparison later sees the CPU fallbacks (pyAT/PyORBIT/Bmad).
export BENCH_MACHINE_SLUG=perlmutter
export BENCH_NCORES=64
export BENCH_SOCKET=0

pixi run -e default bench --device cpu --ncores 64 --skip-build
pixi run -e default validate
pixi run -e default plot
echo "CPU benchmarks done -> results/perlmutter.yaml"
