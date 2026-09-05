# Running the benchmarks on Perlmutter (NERSC)

Clone-and-run recipe for the CPU and GPU partitions, driven entirely by **pixi** (no Cray
PrgEnv modules). The matrix is **{CPU, GPU} × {SP, DP}** per code, **fast-math ON** (the
`validate` step gates physics, so relaxed FP is caught rather than silently wrong).

**Layout:** CPU uses **one socket** of a 2× AMD EPYC 7763 node (64 physical cores, best
MPI/OpenMP split, no hyperthread oversubscription). GPU uses **one A100** (`sm_80`) from
Perlmutter's general `gpu` pool — **40 or 80GB, whichever schedules first**; the exact card is
recorded in the results metadata (`metadata.gpu`, the GPU analogue of the CPU microarch), so a run
stays interpretable regardless of which it lands on. The particle sweep is **extended to 10⁹** (the
default tops at 16M, sized for an 8GB card — see *GPU particle sweep* below). Builds target Zen3 (`-march=znver3`), Perlmutter's
glibc 2.38, and conda **CUDA 13.2** — the exact-fit match to Perlmutter's default `cudatoolkit`
(Aug 2026); a 13.2-built binary is guaranteed to run on its 13.2 driver (see `pixi.toml
[system-requirements]` / `[feature.cuda]`).

## 1. Clone (onto scratch, NOT `$HOME`) + pixi

NERSC `$HOME` is only **40 GB** and inode-limited — pixi's many-file envs + package cache blow past
it. Work from a persistent scratch software area on `$PSCRATCH` (readable from compute nodes; use
`$CFS/m5125/...` if you need it to survive the scratch purge). Do **not** put the repo or cache in a
node-local `mktemp -d` / `/tmp` — those don't cross between login and compute nodes.

```bash
mkdir -p "$PSCRATCH/storage/sw" && cd "$PSCRATCH/storage/sw"      # your persistent scratch sw area
git clone git@github.com:BLAST-ImpactX/impactx-benchmarks.git    # SSH -> publish uses your GH key, no PAT
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

> **`$HOME` quota (this bit us once).** NERSC home is only **40 GB** + inode-limited; the default pixi
> cache `~/.cache/rattler` overflows it mid-install with `Quota exceeded (os error 122)` (unpacking
> `libxcrypt`, failing the `helix` env). `perlmutter_setup.sh` and both `.sbatch` files default
> `PIXI_CACHE_DIR` to **`$PSCRATCH/pixi-cache`** — override to your `$PSCRATCH/storage/sw/...` area if
> you prefer. It must be on a **shared** FS: the cache is written on the login node and read by
> `pixi run` on the compute nodes, so a node-local `mktemp -d`/`/tmp` won't work. Keeping it on the
> same FS as the repo also lets pixi hardlink packages into `.pixi/envs` rather than copy them.

**Long-running — use tmux** so it survives logout (it must stay on a login node; compute nodes
have no internet). Note the login node first — Perlmutter load-balances `perlmutter.nersc.gov`, so
you reattach only on the *same* node:

```bash
hostname                                        # e.g. login09 — remember it
tmux new -s bench
bash machines/perlmutter_setup.sh 2>&1 | tee setup.log
#   detach: Ctrl-b then d ; log off. Reattach later:
#   ssh perlmutter.nersc.gov && ssh login09 && tmux attach -t bench   (or: tail -f setup.log)
```
ccache + already-installed pixi envs make a restart cheap if you need to stop/resume. If NERSC's
login limiter throttles the 16-core build, re-run with `BUILD_NPROC=8`.

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

Each job runs `bench --resume` (with `--skip-build`), then `validate` + `plot`; the GPU job also
emits the GPU-FP32 comparison (`plots/gpu/`). Both request **`-t 24:00:00`**. **`--resume` makes a
wall-clock timeout non-fatal:** each cell is saved as it completes, so if a job hits the wall,
just **resubmit the same script** — it skips the finished cells in `results/perlmutter.yaml` and
continues. (The CPU matrix is heavy: the per-cell MPI×OMP layout sweep is ~7 layouts × `runs=5` on
64 cores, so it needs the full budget.) QOS is *not* the constraint — `gpu_regular`/CPU `regular`
both allow **48 h** and the partitions 7 days — so bump `-t` toward 48 h freely if a job is slow.

## 4. Publish the results — back on a LOGIN node

`publish` pushes results + plots to the `benchmarks` branch (needs internet), so run it on a login
node after both jobs finish. If you cloned via **SSH** (step 1) and your key is on your GitHub
profile, this just works — no token.

**Export the slug once** for the whole login session. Everything that reads the results file
(`validate`, `plot`, `plotting --gpu`, `publish`) resolves it as `results/<machine_slug>.yaml`, and
on a login node the slug otherwise defaults to the *login hostname* — so without this they operate on
a non-existent file and find nothing:

```bash
export BENCH_MACHINE_SLUG=perlmutter          # once; validate/plot/publish all inherit it

# If you re-ran a targeted cell on a compute node (e.g. a --codes pyorbit re-measure), refresh the
# derived artifacts on the login node before publishing:
pixi run publish --push                         # or: validate -> plot -> plotting --gpu, then publish
```

`pixi run validate` ends with a one-line status summary (`N cells: correct=… unconverged=… …` and a
`⚠` line if anything is `incorrect`/`failed`) — glance at it before publishing.

(If you cloned via HTTPS instead, either switch the remote —
`git remote set-url origin git@github.com:BLAST-ImpactX/impactx-benchmarks.git` — or use a GitHub
PAT with `repo` scope: `git config --global credential.helper store` then push once.)

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

### GPU pool, QOS, and the particle sweep

**Any-A100 pool + recorded card.** The job requests `-C gpu` (the full A100 pool), **not**
`-C gpu&hbm80g`: the 80GB-only subset is small, so pinning to it means a long queue. Instead we take
whatever card (40 or 80GB) schedules first and record it in `metadata.gpu` via `nvidia-smi` (name +
memory + driver; the name encodes the tier, `A100-SXM4-80GB` vs `-40GB`). On a **40GB** card the top
of the sweep OOMs earlier than on 80GB — that's fine, it's classified `oom` (not `failed`) and the
recorded card explains exactly why. To force reproducible 80GB instead, put back `&hbm80g` and expect
a longer wait. **QOS:** `perlmutter_gpu.sbatch` uses `-q premium` (priority boost, **2× GPU-hour
charge**) to clear the queue; switch to `-q regular` (1×) to avoid the premium — relaxing the
constraint above already helps a lot on its own.

**Sweep to 10⁹.** The default GPU sweep (`registry._NPARTS_GPU`) tops at **16M** — the ceiling of an
8GB card (where it already OOM'd the PyTorch codes). `perlmutter_gpu.sbatch` overrides it via
`--nparts` up to **10⁹** to reach the compute-bound plateau. Expect a spread at the top: lean AMReX
codes (ImpactX) scale furthest, while memory-heavy codes (Cheetah/HELIX, and space charge at the very
top) OOM **gracefully**. The 10⁹ point with `runs=5` is heavy, so the job requests **`-t 24:00:00`**;
`regular`/`premium` both cap at **48 h**, so raise `-t` freely if 10⁹ is slow. To cap runtime instead,
drop the 10⁹ point, split `≥256M` into a follow-on job, or lower `--runs` (timing is stable at huge N).

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
