# FIXME: harness pins neither MPI nor OpenMP — the "4-core budget" is only logical

Investigation of **ImpactX issue
[#1522](https://github.com/BLAST-ImpactX/impactx/issues/1522)** ("OpenMP particle
push scales poorly vs MPI on shared cores").

**TL;DR — the reported OpenMP-vs-MPI gap is mostly a core-placement measurement
artifact, not an OpenMP defect in ImpactX.** When OpenMP threads and MPI ranks are
pinned to the *identical* physical CPUs (or both left unpinned, as this harness
actually does), they scale the same. The "drift is flat" effect is genuine memory
bandwidth saturation and affects MPI too. The one real, NUMA-independent lever for
the bandwidth wall is **element/kernel fusion** (issue task #5).

- Machine: 12th Gen Intel Core i9-12900H (6 P-cores + 8 E-cores = 14 physical
  cores, 20 logical), Ubuntu 24.04, `powersave` governor + turbo on.
- ImpactX `build/` config: `ImpactX_COMPUTE=OMP ImpactX_MPI=ON ImpactX_SIMD=ON
  ImpactX_FFT=ON`, double precision (= the issue's config).
- MPI here is **MPICH** (matches this harness's `pixi.toml`).
- Throughput = particles/second, min-of-rounds; for MPI, global_npart /
  max-over-ranks time.

---

## 1. The decisive fact: `taskset -c 0-3` is only 2 physical cores

On this CPU the first four *logical* CPUs are two *physical* P-cores plus their
hyperthread siblings (verified via
`/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`):

```
cpu0,cpu1 -> physical core 0   (HT pair)
cpu2,cpu3 -> physical core 1   (HT pair)
cpu4,cpu5 -> physical core 2
cpu6,cpu7 -> physical core 3
...
cpu12..19 -> the 8 E-cores (1 thread each, max 3.8 GHz)
```

So the issue's `taskset -c 0-3` confines a run to **2 physical cores + HT**, not
"4 cores". Any compute-bound code is then capped near ~2× regardless of
parallelism model.

---

## 2. ChrQuad (compute-bound) scales near-linearly *per physical core*

`elements.ChrQuad(ds=1.0, k=1.0, nslice=1)`, 1e6 particles, single-element
`el.push`. Speedups are relative to the 1-core baseline (~1.4e7 part/s).

| placement                                            | OpenMP        | MPI           |
|------------------------------------------------------|---------------|---------------|
| 1 core                                               | 1.0×          | 1.0×          |
| 2 physical, no HT (cpus 0,2)                          | ~2.0×         | ~2.0×         |
| **2 physical + HT (cpus 0-3) = the issue's setup**   | **~2.3×**     | **~2.3×**     |
| **4 distinct physical (cpus 0,2,4,6)**               | **~3.8×**     | **~3.9×**     |
| **unpinned (what this harness actually does)**       | **3.68×**     | **≈ MPI 4r**  |

Key point: **at matched core counts, OpenMP ≈ MPI.** There is no OpenMP-specific
scaling defect for compute-bound elements. The "plateau at ~2×" is the
2-physical-core / hyperthreading ceiling of `taskset -c 0-3`.

Representative back-to-back unpinned run (same thermal state, so turbo-fair):

```
ChrQuad, npart=1e6, UNPINNED (free to use all 14 cores):
  OMP 1T : 1.371e7 part/s
  OMP 4T : 5.050e7 part/s   -> 3.68x
  MPI 4r : 4.965e7 part/s   -> comparable
```

---

## 3. The flat drift is memory-bandwidth saturation (physics), not OpenMP

`elements.ExactDrift(ds=0.25, nslice=1)`, 1e6 particles. The kernel is already
minimal — it reads `px,py,pt` and writes `x,y,t` (momenta untouched). At 4 threads
it reaches ~60 GB/s, saturating the memory controller:

| placement                          | OpenMP drift  | MPI drift     |
|------------------------------------|---------------|---------------|
| 1 core                             | ~4.9e8        | ~5.0e8        |
| 2 physical + HT (cpus 0-3)         | ~1.1×         | ~1.26×        |
| 4 distinct physical (cpus 0,2,4,6) | ~1.25×        | ~1.36×        |

Both models are flat; MPI's small edge (~5-10%) is a first-touch / contiguous
per-rank-array locality effect, not a fixable OpenMP bug. No scheduling or affinity
trick changes a bandwidth-bound kernel.

---

## 4. How the reported "MPI 3.62× vs OpenMP 2.0×" happened

Two compounding placement issues:

1. **This harness pins nothing.** `benchmarks/runner.py` builds the launch command
   as plain `mpirun -np N python …` (line ~82-84) and the OpenMP path as plain
   `python` with only `OMP_NUM_THREADS` set (no `taskset`, no `--bind-to`). The
   "`ranks × threads ≤ ncores`" invariant in `_run_layout` is **logical only** —
   on a 14-core box neither mode is physically constrained to the budget. Verified:

   ```
   # harness-style: mpirun -np 4 (no taskset, no -bind-to)
   rank 0..3 -> cpus [0,1,2,...,19]   # full machine; OS spreads ranks onto 4 fast P-cores
   ```

   So both modes float onto distinct fast P-cores and OpenMP scales fine (3.68×).
   The headline gap does **not** appear in the actual harness on this machine.

2. **The manual `taskset -c 0-3 … --bind-to core` repro in the issue text is the
   trap.** `taskset` genuinely confines the OpenMP process to 2 physical cores, but
   OpenMPI's `--bind-to core` *escapes the cpuset* and binds the 4 ranks to 4
   *distinct* physical cores. Net: OpenMP got 2 physical cores (→2×), MPI got 4
   (→~3.8×) — never the same cores. (MPICH `-bind-to` does respect the taskset
   mask; with MPICH the two would match.)

When MPI is genuinely confined to the same 2 physical cores (cpus 0-3 via explicit
per-rank pinning), it gets ~2.3×, the same as OpenMP.

---

## 5. "Improve" — levers tried, with measurements

### Per-push OpenMP overhead ≈ 15 µs (real, but narrow)
Measured directly with a tiny push (npart=2000): a drift push is 3 µs at 1T but
**18 µs at 4T** — the delta is per-push fork/join + AMReX
`critical(gettilearray)` + the implicit barrier. Setting
`impactx.do_dynamic_scheduling=0` cuts it to ~11 µs (~8 µs overhead). This only
matters for **thin-element / low-npart** lattices; at 1e6 particles it is <2%.

### Negative result — static scheduling is NOT a safe default
It *regresses* realistic multi-element FODO tracking, because dynamic scheduling
load-balances the unequal tiles `ImpactXParticleContainer::prepare()` creates:

```
FODO track, 4 distinct cores, 4 threads (part-elem/s):
  npart=1e5   dynamic 9.77e7   vs  static 7.16e7   (static -37%)
  npart=2e4   dynamic 9.56e7   vs  static 7.89e7   (static -21%)
  npart=1e6   dynamic 6.49e7   vs  static 6.79e7   (static  +5%)
```

The dynamic default is well-chosen for multi-element tracking.

### Negative result — forcing OpenMP affinity hurts on this P/E machine
`OMP_PROC_BIND=spread OMP_PLACES=cores` *dropped* unpinned ChrQuad from
5.05e7 → 2.39e7 by binding threads onto the slow E-cores, fighting the OS
scheduler's good default P-core placement.

### The real fix for the bandwidth wall: element/kernel fusion (4× ceiling)
Pushing a tile through N elements while it is hot in cache turns N memory passes
into 1. Measured ceiling (drifts compose exactly, so this is apples-to-apples):

```
1e6 particles, 4 distinct cores:
  4x ExactDrift(ds=0.25).push  : 6.18 ms   (4 memory passes)
  1x ExactDrift(ds=1.0).push   : 1.54 ms   (1 memory pass)   -> 4.01x speedup
```

This is the only lever that both fixes the flat drift *and* lets threads help on
bandwidth-bound transport (it raises arithmetic intensity). It is issue #1522
task #5 and is an architectural change in the ImpactX tracking loop (group
consecutive collective-effect-free elements inside one `ParIter` pass).

---

## 6. Recommendations for this harness

1. **Physically pin both modes to the same CPU set** so the 4-core budget is real
   on multi-core workstations. Without it, results are dominated by OS placement
   and frequency scaling, not by the code. Concretely:
   - OpenMP run: wrap the process in `taskset -c <list>` and set
     `OMP_NUM_THREADS` to the list length.
   - MPI run: pin **each rank** to one CPU from the same list. MPICH `mpirun
     -bind-to user:<list>` or a per-rank `taskset` wrapper keyed on `$PMI_RANK`
     works; do **not** rely on `--bind-to core` semantics (OpenMPI can escape an
     outer `taskset`).
   - Choose the list to be **distinct physical cores** (e.g. `0,2,4,6` here) to
     emulate a real N-vCPU runner, and document that hosted 4-vCPU CI runners
     usually expose 4 distinct cores, not 2 cores + HT.
2. **Record placement in results metadata** (per-rank/thread affinity, governor,
   turbo state). The `powersave` governor + turbo on this box swings frequency
   from 0.4→5 GHz and confounds absolute throughput; pin the governor to
   `performance` (or disable turbo) for stable numbers, and prefer turbo-fair
   metrics (OpenMP vs MPI at matched core counts; ChrQuad-vs-drift scaling ratios
   within one config).
3. **Do not "fix" ImpactX OpenMP with** static scheduling or forced affinity —
   both measured net-negative above.
4. **Track the fusion work** as the substantive ImpactX-side follow-up; it is the
   real win for bandwidth-bound transport (~4× ceiling).

---

## 7. Reproducer scripts

Standalone scripts used for the numbers above live in `/tmp/impactx_1522/`:

- `bench_elem.py` — single-element `el.push` throughput; env knobs `NPART`,
  `ROUNDS`, `ELEMS`, `TILE`, `DYN` (=`impactx.do_dynamic_scheduling`).
- `bench_elem_mpi.py` — MPI variant (global npart split across ranks).
- `bench_track.py` — FODO multi-element track (per-push path; `PERIODS`, `NSLICE`).
- `pin.sh` — per-rank pinning wrapper (`CORES=0,2,4,6 mpirun -np 4 -bind-to none
  ./pin.sh …`).
- `fuse_probe.py` — the drift fusion-ceiling measurement.

Example fair comparison on 4 distinct physical cores:

```bash
# OpenMP, pinned to 4 distinct physical cores
taskset -c 0,2,4,6 env OMP_NUM_THREADS=4 NPART=1000000 python bench_elem.py

# MPI, one rank per the same 4 physical cores
CORES=0,2,4,6 mpirun -np 4 -bind-to none ./pin.sh \
  env OMP_NUM_THREADS=1 NPART=1000000 python bench_elem_mpi.py
```
