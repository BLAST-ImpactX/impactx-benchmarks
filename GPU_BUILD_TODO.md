# GPU (CUDA) variant — build status & notes

Default target: **1 GPU**, FP32 (+ FP64 where a code is FP64-only). All via pixi CUDA envs.

## Local build DONE — RTX A2000 (laptop), compute capability 8.6

Built and validated the **GPU FP32** variant of every GPU-capable code on the local A2000
(sm_86, CUDA 12.9 toolkit, driver 610). Results merged into `results/axel-dell.yaml`; GPU FP32
cross-code comparison plots in `plots/gpu/` (`pixi run plot --gpu`).

| code    | GPU FP32 | notes |
|---------|----------|-------|
| ImpactX | ✅ built (impactx-cuda-sp) | sm_86 auto-detected, `--use_fast_math`; cuFFT/cuRAND linked |
| Cheetah | ✅ env (cheetah-gpu) | FP32 = runtime dtype; eager **and** `torch.compile` (inductor) both work |
| SciBmad | ✅ env (scibmad-gpu) | CuArray + KernelAbstractions CUDA backend; `CUDA.functional()` true |
| Xsuite  | ⚠️ FP64 only | no FP32 build → comparison shows GPU-FP64 with an asterisk |
| pyAT / PyORBIT / Bmad | ✗ no GPU | CPU is the next-best → CPU bar + asterisk in the comparison |

### Measured (100k particles, particles/sec) — physics all validated `correct`
- **ImpactX FP32**: fodo ~4.7–5.0e8, htu ~1.7e7, htu_spin ~1.4e7, 3D-SC ~2.2e7  (≈6× its CPU FP32).
- **Cheetah FP32**: compiled fodo ~1.1e8 (≈4× eager ~3e7), 3D-SC ~8.3e6.
- **SciBmad FP32**: fodo ~3.5e6, htu_spin ~2.5e5 (KA-CUDA launch overhead dominates at 100k →
  *slower than its CPU FP32* here; expected to cross over at larger N).
- **Xsuite FP64** (fallback): fodo ~3.6e7, 3D-SC ~8.7e6, 2.5D-SC ~1.3e7.

### Three fixes needed during the build (all committed in pixi.toml / build.sh)
1. **ImpactX NVTX header** — AMReX_GpuDevice.cpp needs `nvtx3/nvToolsExt.h`; `cuda-toolkit` ships
   only the NVTX *runtime*. Fix: add `cuda-nvtx-dev` to `[feature.cuda]`.
2. **Cheetah pytorch was the CPU build** — pixi doesn't probe the driver, so conda-forge picked
   `pytorch=*cpu_mkl*`. Fix: `[feature.cheetah-gpu.system-requirements] cuda = "12"` (declares the
   `__cuda` virtual package) → selects `cuda129_mkl`. Also **pinned python=3.13** (no py3.14 CUDA
   pytorch build exists yet; the CPU envs stay on 3.14).
3. **Cheetah `torch.compile` on GPU** — Inductor/Triton codegens a `cuda_utils.c` launcher needing
   `cuda.h` + a host C compiler. Fix: add `cuda-driver-dev cuda-cudart-dev cuda-nvcc c-compiler`
   and `CUDA_HOME` to `[feature.cheetah-gpu]`.
   Also: the Xsuite GPU readout passed cupy arrays into the numpy observables → fixed centrally in
   `scenarios/_obs.py` (`_host()` pulls cupy/torch arrays to host via `.get()`/`.cpu()`).

### KNOWN LIMITATION — ImpactX 2.5D space charge segfaults on GPU
`impactx-cuda-sp spacecharge_2p5d` → SIGSEGV in
`impactx::particles::spacecharge::GatherAndPush` (→ `HandleSpacecharge` → `track_particles`).
ImpactX **3D**-SC runs fine on GPU and **2.5D**-SC runs fine on CPU, so this is an upstream
ImpactX issue in the 2.5D-SC gather/push CUDA path (build pinned to PR #1521, ed88d69). The
comparison plot for `spacecharge_2p5d` therefore has no GPU-FP32 bar (Xsuite GPU-FP64 only) and
is intentionally not emitted. **Re-check when ImpactX is moved off PR #1521 to development.**

## GPU-aware MPI (device pointers straight to MPI; multi-GPU only)

Only **ImpactX/AMReX** does MPI+GPU in this suite (Cheetah/SciBmad/Xsuite run single-process on
GPU; PyORBIT is CPU-only). AMReX **auto-detects** GPU-aware MPI at runtime via
`MPIX_GPU_query_support()` and enables `amrex.use_gpu_aware_mpi` **only when the MPI reports
support** — i.e. it already does exactly "on where supported", per platform. We must NOT force it:
forcing it on a non-GPU-aware MPI makes AMReX pass device pointers to MPI → segfault in multi-rank
runs (verified below).

The runner sets `MPICH_GPU_SUPPORT_ENABLED=1` for every cuda run (`benchmarks/runner.py`
`_runtime_env`). This is the **Cray-MPICH** knob that enables GPU support on Perlmutter (AMReX then
auto-detects it and turns on device-pointer MPI). It is a harmless no-op on our conda-forge MPICH.

Local finding (RTX A2000, conda-forge MPICH 5.0.1 + UCX 1.20.1):
- **UCX is CUDA-aware** (`cuda_copy`/`cuda_ipc` transports, `HAVE_CUDA=1`), BUT
- **MPICH is NOT**: `MPIX_Query_cuda_support()=0` (also with `MPIR_CVAR_ENABLE_GPU=1`), and a 2-rank
  `MPI_Send` of a `cudaMalloc` device buffer **segfaults**. conda-forge MPICH was built without GPU
  support, so its ch4 layer never routes device pointers to UCX's CUDA path. No CUDA-aware
  mpich/openmpi exists on conda-forge (openmpi "external" builds need a system CUDA-aware MPI).
- So AMReX correctly keeps `use_gpu_aware_mpi=0` locally. This only matters for multi-GPU anyway;
  the 1-GPU benchmark layout (1 rank) does no MPI communication.

**Perlmutter:** build ImpactX against **cray-mpich** (the system GPU-aware MPI), not conda mpich;
with `MPICH_GPU_SUPPORT_ENABLED=1` (the runner sets it), AMReX auto-detects and uses GPU-aware MPI.

## Still TODO
- **ImpactX GPU FP64** (`impactx-cuda-dp`) is wired but not yet built locally (only FP32 was asked
  for). Build with `pixi run -e impactx-cuda-dp build-impactx-cuda-dp` when an FP64 GPU comparison
  is wanted.
- **Perlmutter / multi-GPU** (≤4× A100, sm_80): `CUDA_ARCH` auto-detects to 80 there; the runner's
  multi-GPU mpirun branch is stubbed and needs per-rank `CUDA_VISIBLE_DEVICES` binding.
- File the ImpactX 2.5D-SC GPU segfault upstream (or confirm it's fixed post-#1521).
