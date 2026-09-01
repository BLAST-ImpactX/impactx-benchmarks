#!/bin/bash -l
# Perlmutter (NERSC) CPU benchmark job: one socket of a 2× AMD EPYC 7763 node, driven by pixi.
# Submit from the repo root AFTER machines/perlmutter_setup.sh has built the envs on a login node.
#   sbatch machines/perlmutter_cpu.sbatch
#
# NERSC allocation: bare project name (the "_g" GPU suffix is optional these days). The fix here is
# the TASK LAYOUT -- mirror WarpX's Perlmutter template (16 ranks x 16 cpus = the full 256-thread
# node) so NERSC's submit-policy filter accepts the job; without an explicit layout an --exclusive
# job is rejected as "Job request does not match any supported policy". The harness still pins its
# OWN split to socket 0 (BENCH_SOCKET/BENCH_NCORES) via its own mpirun, ignoring these SLURM values.
#SBATCH -A m5125
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH -t 06:00:00
#SBATCH -J bench-cpu
#SBATCH -o bench-cpu.o%j
#SBATCH -e bench-cpu.e%j
#SBATCH --mail-type=END,FAIL
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$PSCRATCH/pixi-cache}"   # same scratch cache as setup.sh (off the $HOME quota)

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
