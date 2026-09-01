#!/bin/bash -l
# Perlmutter (NERSC) GPU benchmark job: a single A100-80GB, driven by pixi.
# Submit AFTER the CPU job so the GPU-FP32 comparison plot sees the CPU fallbacks:
#   cpu=$(sbatch --parsable machines/perlmutter_cpu.sbatch)
#   sbatch --dependency=afterok:$cpu machines/perlmutter_gpu.sbatch
#
# NERSC allocation: bare project name (the "_g" GPU suffix is optional these days). Task/GPU layout
# mirrors WarpX's Perlmutter GPU template (4 ranks x 4 GPUs = full node) so NERSC's submit-policy
# filter accepts the job -- an --exclusive job with no explicit layout is rejected as "Job request
# does not match any supported policy". The harness then uses ONE GPU (BENCH_CUDA_DEVICES=0).
#SBATCH -A m5125
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 24:00:00
#SBATCH -J bench-gpu
#SBATCH -o bench-gpu.o%j
#SBATCH -e bench-gpu.e%j
#SBATCH --mail-type=END,FAIL
set -eu
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$PSCRATCH/pixi-cache}"   # same scratch cache as setup.sh (off the $HOME quota)

# Exclusive node (4× A100) but the benchmark uses ONE GPU (CUDA_VISIBLE_DEVICES=0).
# MPICH_GPU_SUPPORT_ENABLED=1 is the Cray-MPICH GPU-aware knob (moot at 1 GPU / no MPI, but correct).
# Same results file ("perlmutter") as the CPU job -> merged.
export BENCH_MACHINE_SLUG=perlmutter
export BENCH_CUDA_DEVICES=0
export MPICH_GPU_SUPPORT_ENABLED=1

# 80GB sweep. The default nparts_gpu tops at 16M -- sized for an 8GB card, where 16M is already the
# ceiling (it OOM'd the PyTorch codes). An A100-80GB has ~10x the memory, so extend the sweep with
# --nparts (which overrides the per-scenario default) up to 1e9 to reach the compute-bound plateau.
# Expect a spread at the top: lean AMReX codes (ImpactX) scale furthest; memory-heavy codes
# (Cheetah/HELIX, and space charge at the very top) OOM gracefully (classified 'oom', not failed).
# NOTE: 1e9 x runs=5 across all GPU configs is heavy and its runtime is uncertain, so -t is 24h as
# insurance against a timeout wasting the run. QOS is NOT the constraint: gpu_regular allows 48h
# (2 days) and the gpu_ss11 partition 7 days -- bump -t up to 48h freely if 1e9 is slow. To cap
# runtime instead: drop the 1e9 point, split >=256M into a follow-on job, or lower --runs.
export BENCH_NPARTS_GPU="100000,1000000,4000000,16000000,64000000,256000000,1000000000"
pixi run -e default bench --device cuda --skip-build --nparts "$BENCH_NPARTS_GPU"
pixi run -e default validate
pixi run -e default plot
pixi run -e default python -m benchmarks.plotting --gpu   # GPU-FP32 cross-code comparison
echo "GPU benchmarks done -> results/perlmutter.yaml ; plots/ + plots/gpu/"
