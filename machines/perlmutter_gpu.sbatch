#!/bin/bash -l
# Perlmutter (NERSC) GPU benchmark job: a single A100-80GB, driven by pixi.
# Submit AFTER the CPU job so the GPU-FP32 comparison plot sees the CPU fallbacks:
#   cpu=$(sbatch --parsable machines/perlmutter_cpu.sbatch)
#   sbatch --dependency=afterok:$cpu machines/perlmutter_gpu.sbatch
#
# NERSC allocation. The "_g" GPU-project suffix is optional/skippable on Perlmutter now, so the
# bare project name works for both the CPU and GPU jobs.
#SBATCH -A m5125
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -G 4
#SBATCH -t 06:00:00
#SBATCH -J bench-gpu
#SBATCH -o bench-gpu.o%j
#SBATCH -e bench-gpu.e%j
#SBATCH --mail-type=END,FAIL
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Exclusive node (4× A100) but the benchmark uses ONE GPU (CUDA_VISIBLE_DEVICES=0); GPU runs use
# the larger nparts_gpu sweep. MPICH_GPU_SUPPORT_ENABLED=1 is the Cray-MPICH GPU-aware knob (moot at
# 1 GPU / no MPI, but correct). Same results file ("perlmutter") as the CPU job -> merged.
export BENCH_MACHINE_SLUG=perlmutter
export BENCH_CUDA_DEVICES=0
export MPICH_GPU_SUPPORT_ENABLED=1

pixi run -e default bench --device cuda --skip-build
pixi run -e default validate
pixi run -e default plot
pixi run -e default python -m benchmarks.plotting --gpu   # GPU-FP32 cross-code comparison
echo "GPU benchmarks done -> results/perlmutter.yaml ; plots/ + plots/gpu/"
