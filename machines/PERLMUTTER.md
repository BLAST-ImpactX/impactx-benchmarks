# Running the benchmarks on Perlmutter (NERSC)

Clone-and-run recipe for the CPU and GPU partitions, driven entirely by **pixi** (no Cray
PrgEnv modules). The matrix is **{CPU, GPU} × {SP, DP}** per code, **fast-math ON** (the
`validate` step gates physics, so relaxed FP is caught rather than silently wrong).

**Layout:** CPU uses **one socket** of a 2× AMD EPYC 7763 node (64 physical cores, best
MPI/OpenMP split, no hyperthread oversubscription). GPU uses **one A100-80GB** on an exclusive
node, with a larger particle sweep. Builds target Zen3 (`-march=znver3`) and Perlmutter's
glibc 2.38 (see `pixi.toml [system-requirements]`).

## 1. Clone

```bash
git clone https://github.com/BLAST-ImpactX/impactx-benchmarks.git
cd impactx-benchmarks
```
(pixi must be on PATH — `module load pixi` or a user install.)

## 2. Build the environments — on a LOGIN node (once)

Compute nodes have no internet, so all conda/pip/git downloads + compiles happen here:

```bash
bash machines/perlmutter_setup.sh     # builds all CPU + GPU envs (znver3, fast-math ON, sm_80)
```

## 3. Submit the run jobs

Edit the `#SBATCH -A` account line in each `.sbatch` first (CPU project; GPU project ends in
`_g`). Submit CPU first, then GPU **dependent on it** so the GPU-FP32 comparison sees the CPU
fallbacks (both jobs write the same `results/perlmutter.yaml`):

```bash
cpu=$(sbatch --parsable machines/perlmutter_cpu.sbatch)
sbatch --dependency=afterok:$cpu machines/perlmutter_gpu.sbatch
```

Each job runs `bench` (with `--skip-build`), then `validate` + `plot`; the GPU job also emits the
GPU-FP32 comparison (`plots/gpu/`).

## 4. Publish the results — back on a LOGIN node

`publish` pushes to the `benchmarks` branch (needs internet + git credentials), so run it on a
login node after both jobs finish:

```bash
pixi run -e default python -m benchmarks.publish --push
```

## Knobs (already set by the scripts)

| var | where | meaning |
|---|---|---|
| `BENCH_ARCH=znver3` | setup.sh (build) | CPU microarch for the source builds |
| `BENCH_FASTMATH=1` | setup.sh (build) | fast-math ON (default; set `0` for an IEEE build) |
| `BENCH_NCORES=64`, `BENCH_SOCKET=0` | cpu.sbatch (run) | pin to one 64-core EPYC socket |
| `BENCH_CUDA_DEVICES=0` | gpu.sbatch (run) | single A100 |
| `BENCH_MACHINE_SLUG=perlmutter` | both | one merged results file + plots |

## Notes

- **Fast-math coverage:** ON for ImpactX (CPU+GPU), Cheetah-compiled, pyAT, PyORBIT, Bmad, and
  Xsuite (CPU + GPU). **SciBmad** and **Cheetah-eager** have no fast-math knob and run IEEE
  (asterisked). See the fast-math table in the repo plan / `codes/*/build.sh`.
- **Rebuild off-fast-math** (future IEEE axis): re-run `perlmutter_setup.sh` with `BENCH_FASTMATH=0`.
- The harness pins layouts itself (taskset / per-rank), so the jobs call `pixi run bench` directly
  (its own `mpirun`), not `srun`.
