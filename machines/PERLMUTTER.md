# Running the benchmarks on Perlmutter (NERSC)

Clone-and-run recipe for the CPU and GPU partitions, driven entirely by **pixi** (no Cray
PrgEnv modules). The matrix is **{CPU, GPU} × {SP, DP}** per code, **fast-math ON** (the
`validate` step gates physics, so relaxed FP is caught rather than silently wrong).

**Layout:** CPU uses **one socket** of a 2× AMD EPYC 7763 node (64 physical cores, best
MPI/OpenMP split, no hyperthread oversubscription). GPU uses **one A100-80GB** (`sm_80`) on an
exclusive node, with the particle sweep **extended to 10⁹** to exploit the 80GB (the default tops at
16M, sized for an 8GB card — see *GPU particle sweep* below). Builds target Zen3 (`-march=znver3`), Perlmutter's
glibc 2.38, and conda **CUDA 13.2** — the exact-fit match to Perlmutter's default `cudatoolkit`
(Aug 2026); a 13.2-built binary is guaranteed to run on its 13.2 driver (see `pixi.toml
[system-requirements]` / `[feature.cuda]`).

## 1. Clone (into `$PSCRATCH`) + pixi

Work from scratch — pixi materializes many-file envs; `$HOME` inode limits bite, and compute nodes
can read `$PSCRATCH`. (Use `$CFS/m5125/...` instead if you need it to survive the scratch purge.)

```bash
cd "$PSCRATCH"
git clone https://github.com/BLAST-ImpactX/impactx-benchmarks.git
cd impactx-benchmarks
```

pixi on PATH — NERSC has no pixi module, so user-install once (skips if already present):

```bash
command -v pixi >/dev/null || { curl -fsSL https://pixi.sh/install.sh | bash && source ~/.bashrc; }
pixi --version
```

## 2. Build the environments — on a LOGIN node (once)

Compute nodes have no internet, so all conda/pip/git downloads + compiles happen here:

```bash
bash machines/perlmutter_setup.sh     # builds all CPU + GPU envs (znver3, fast-math ON, sm_80)
```

This is the long step (from-source Bmad/SciBmad/PyORBIT/ImpactX/Elegant/HELIX). When it finishes,
smoke-test the GPU build on a short interactive node **before** committing the multi-hour jobs — a
broken env should not be discovered 12 h into a `regular` allocation:

```bash
salloc -A m5125 -C gpu -q interactive -N 1 -G 1 -t 00:20:00
BENCH_MACHINE_SLUG=smoke pixi run -e default bench --codes impactx --scenarios fodo_exact \
    --device cuda --precision single --nparts 100000,1000000 --runs 1 --skip-build
rm -f results/smoke.yaml; exit          # discard the smoke result, drop the alloc
```

## 3. Submit the run jobs

The `#SBATCH -A` account is already set to `m5125` in both `.sbatch` files (the `_g` GPU-project
suffix is optional on Perlmutter now, so the bare project works for both jobs). Submit CPU first,
then GPU **dependent on it** so the GPU-FP32 comparison sees the CPU fallbacks (both jobs write the
same `results/perlmutter.yaml`):

```bash
cpu=$(sbatch --parsable machines/perlmutter_cpu.sbatch)
sbatch --dependency=afterok:$cpu machines/perlmutter_gpu.sbatch
squeue --me                              # watch state; logs stream to bench-{cpu,gpu}.o<jobid>
```

Each job runs `bench` (with `--skip-build`), then `validate` + `plot`; the GPU job also emits the
GPU-FP32 comparison (`plots/gpu/`). **Heads-up:** the GPU job sweeps to 10⁹ particles at `runs=5`
and requests `-t 12:00:00` — confirm that fits the current `regular` GPU QOS max walltime
(`sacctmgr show qos regular format=Name,MaxWall`); if it's tight, trim the sweep in
`perlmutter_gpu.sbatch` (drop the `1000000000` point) rather than risk a timeout.

## 4. Publish the results — back on a LOGIN node

`publish` pushes results + plots to the `benchmarks` branch (needs internet + push credentials), so
run it on a login node after both jobs finish. Set up an HTTPS token once (NERSC login nodes reach
github.com over HTTPS; a GitHub Personal Access Token with `repo` scope is the reliable path):

```bash
git config --global credential.helper store    # caches the token after the first push
pixi run publish --push                          # first push prompts: username + PAT (not password)
```

`--remote` is auto-detected (a fresh clone's `origin`). Re-run `publish` (no `--push`) first for a
dry run if you want to preview what will be committed. The results land as
`results/perlmutter.yaml` + `plots/` (incl. `plots/gpu/`) on the `benchmarks` branch, plus a
timestamped `history/<utc>_perlmutter/` archive.

## Knobs (already set by the scripts)

| var | where | meaning |
|---|---|---|
| `BENCH_ARCH=znver3` | setup.sh (build) | CPU microarch for the source builds |
| `BENCH_FASTMATH=1` | setup.sh (build) | fast-math ON (default; set `0` for an IEEE build) |
| `BENCH_NCORES=64`, `BENCH_SOCKET=0` | cpu.sbatch (run) | pin to one 64-core EPYC socket |
| `BENCH_CUDA_DEVICES=0` | gpu.sbatch (run) | single A100 |
| `BENCH_NPARTS_GPU=…,1000000000` | gpu.sbatch (run) | 80GB particle sweep, passed as `--nparts` (overrides the 16M default) |
| `BENCH_MACHINE_SLUG=perlmutter` | both | one merged results file + plots |

### GPU particle sweep (80GB)

The default GPU sweep (`registry._NPARTS_GPU`) tops at **16M** — that is the ceiling of an 8GB card
(where it already OOM'd the PyTorch codes), not of an A100-80GB. `perlmutter_gpu.sbatch` therefore
overrides it via `--nparts` up to **10⁹** to reach the compute-bound plateau. Expect a spread at the
top: lean AMReX codes (ImpactX) scale furthest, while memory-heavy codes (Cheetah/HELIX, and space
charge at the very top) OOM **gracefully** — classified `oom`, distinct from `failed`. The 10⁹ point
with `runs=5` is heavy; **12h may not suffice** — if the job times out, drop the 10⁹ point, split
`≥256M` into a follow-on job, or lower `--runs` (timing is stable at huge N). Verify `-t` against the
current NERSC GPU `regular` QOS max walltime before submitting.

## Notes

- **Fast-math coverage:** ON for ImpactX (CPU+GPU), Cheetah-compiled, pyAT, PyORBIT, Bmad, and
  Xsuite (CPU + GPU). **SciBmad**, **Cheetah-eager**, **Elegant** (released DP code) and **HELIX**
  (PyTorch `torch.fft`) have no fast-math knob and run IEEE. See `codes/*/build.sh`.
- **Codes:** 9 total — impactx, cheetah, pyat, pyorbit, xsuite, scibmad, bmad, elegant, helix.
  HELIX (PyTorch PIC) runs the 3D `spacecharge` scenario only, CPU+GPU × FP32/FP64 (same IGF model
  as ImpactX). Elegant covers the tracking scenarios (DP only; CPU MPI + GPU).
- **Rebuild off-fast-math** (future IEEE axis): re-run `perlmutter_setup.sh` with `BENCH_FASTMATH=0`.
- The harness pins layouts itself (taskset / per-rank), so the jobs call `pixi run bench` directly
  (its own `mpirun`), not `srun`.
