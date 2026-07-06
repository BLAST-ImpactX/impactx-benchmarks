# ImpactX Benchmarks

Automated **correctness & performance** benchmarks of [ImpactX](https://github.com/BLAST-ImpactX/impactx)
against other community beam-dynamics codes. Each run compiles/installs the codes the
best-performing supported way for the host (native `-march=native -mtune=native` where it
beats prebuilt packages), tracks identical physics scenarios, **validates the physics**, and
records throughput (particles/second).

The same harness runs **locally** and in **GitHub Actions** (Ubuntu 24.04, 4 vCPU — the
single public, CPU-only source of truth). Results and plots are published to the
[`benchmarks`](../../tree/benchmarks) branch and embedded below.

## Benchmark rules

**You get 4 CPU cores. Use them however is fastest.** Each code may spend that budget any
way its design allows — all MPI ranks, all OpenMP/threads, or a mix — but the total must
satisfy **`MPI_ranks × threads ≤ 4`**. The budget is **physically pinned**: every run is
confined to the *same* set of 4 distinct physical cores (`taskset` for the threaded process,
one core per rank for MPI), so OpenMP and MPI layouts compete on identical hardware rather than
whatever the OS scheduler hands out. For every `(code, scenario)` the harness *sweeps* the
layouts a code supports (e.g. `4r×1t`, `2r×2t`, `1r×4t`) and reports the **fastest** one; the
winning layout is printed under each bar. This keeps the comparison about implementation
quality, not who was allowed more cores — and it lets each code shine where its parallelism
works best (e.g. ImpactX chromatic FODO is fastest at `2r×2t` but exact FODO at `1r×4t`; PyORBIT
tracking scales with ranks but its space charge is fastest at `1r×1t`). Codes are also built the
best-performing supported way (native `-march=native -mtune=native`, SIMD, the right solver),
and physics is validated every run.

## Latest results

<!-- These images live on the `benchmarks` branch and update on every run. -->
![fodo_chromatic](https://raw.githubusercontent.com/BLAST-ImpactX/impactx-benchmarks/benchmarks/plots/fodo_chromatic.svg)
![fodo_exact](https://raw.githubusercontent.com/BLAST-ImpactX/impactx-benchmarks/benchmarks/plots/fodo_exact.svg)
![htu](https://raw.githubusercontent.com/BLAST-ImpactX/impactx-benchmarks/benchmarks/plots/htu.svg)
![htu_spin](https://raw.githubusercontent.com/BLAST-ImpactX/impactx-benchmarks/benchmarks/plots/htu_spin.svg)
![spacecharge](https://raw.githubusercontent.com/BLAST-ImpactX/impactx-benchmarks/benchmarks/plots/spacecharge.svg)
![spacecharge_2p5d](https://raw.githubusercontent.com/BLAST-ImpactX/impactx-benchmarks/benchmarks/plots/spacecharge_2p5d.svg)

> Images appear once the first run has published to the `benchmarks` branch.
> (GitHub's CDN caches raw images for a few minutes, so updates may lag slightly.)

**How to read the bars** (taller = faster, particles/second; the y-axis always scales
to fully show the fastest code):

| appearance | meaning |
|---|---|
| solid bar | physics **validated correct** vs the reference |
| value with **`*`** | shown, but the code runs a *non-tuned* model for this problem — either a costlier one (exact where chromatic suffices) or a cheaper/cruder one (point-1/r vs the integrated Green function); see the per-plot footnote |
| **dashed** bar + marker | plotted but physics flagged: `incorrect` (`physics ✗`) or `unconverged` |
| grey "unsupported" | the **code** cannot do this physics (e.g. no space charge) |
| grey "not in harness" | the code can, but a run template hasn't been added **here** yet |
| grey "OOM" / "failed" | ran out of memory / errored |

## Codes & capabilities

Scenarios: **fodo_chromatic** (chromatic-paraxial FODO), **fodo_exact** (exact non-paraxial
FODO), **htu** (tracking), **htu_spin** (htu + Thomas-BMT spin), **spacecharge** (3D PIC),
**spacecharge_2p5d** (2.5D PIC). ✓ = implemented and physics-validated; ✓\* = shown but runs a
*non-tuned* model (see footnotes); ✗ = the code cannot do that physics. (All current
numbers are **FP64** — shown in each plot title.)

| code | fodo_chr¹ | fodo_exact² | htu³ | htu_spin⁵ | 3D SC⁴ | 2.5D SC⁴ | FP32⁶ | parallelism | install |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|------|------|
| [ImpactX](https://github.com/BLAST-ImpactX/impactx) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | OpenMP+SIMD(+MPI) | source (CMake) |
| [Cheetah](https://github.com/desy-ml/cheetah) | ✓\* | ✓ | ✓\* | ✗ | ✓ | ✗ | ✓ | torch / `torch.compile` | pip |
| [pyAT](https://github.com/atcollab/at) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | OpenMP | source (native+OpenMP) |
| [PyORBIT3](https://github.com/PyORBIT-Collaboration/PyORBIT3) | ✓ | ✗ | ✓ | ✗ | ✓\* | ✓\* | ✗ | MPI | source (Meson) |
| [Xsuite](https://github.com/xsuite/xsuite) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | OpenMP | pip |
| [SciBmad.jl](https://github.com/bmad-sim/SciBmad.jl) | ✓\* | ✓ | ✓\* | ✓ | ✗ | ✗ | ✓ | Julia threads | Julia pkg |
| [Bmad](https://github.com/bmad-sim/bmad-ecosystem) | ✓\* | ✓ | ✓\* | ✓ | ✗ | ✗ | ✗ | OpenMP | source (Fortran, libbmad) |

The FODO and space-charge benchmarks split on the element/solver *model* so the comparison is
always like-for-like:
¹ **fodo_chr** — chromatic-paraxial FODO: `ChrQuad` + `ChrDrift` (`x' = px/(1+δ)`), the fast
workhorse model. ImpactX/pyAT/Xsuite/PyORBIT run it natively. Cheetah (no chromatic quad) and
SciBmad (only exact `MatrixKick`+exact drift) run the costlier **exact** map as a stand-in → **\***.
² **fodo_exact** — exact, non-paraxial FODO: `ExactQuad` + `ExactDrift` (nonlinear, sqrt-based):
ImpactX, Cheetah (`drift_kick_drift`), pyAT (`ExactMultipolePass`), Xsuite (`drift-kick-drift-exact`),
SciBmad (`MatrixKick` — a symplectic integrator of the exact quad Hamiltonian — + `exact_drift`).
PyORBIT has no exact quad/drift (TEAPOT is chromatic-paraxial) → ✗. Chromatic vs exact agree to O(angle²).
³ **htu** — the BELLA HTU beamline, chromatic-paraxial model. Cheetah/SciBmad run the costlier
exact map (no chromatic quad/drift) → **\***.
⁵ **htu_spin** — the same HTU lattice with **Thomas-BMT spin tracking**; the beam starts fully
spin-aligned (+z) and the dispersive chicane differentially precesses spins → depolarization
(RMS spin spread, validated on sigma_sx/sy). Run by ImpactX (`SpinvMF`), Xsuite (`configure_spin`)
and SciBmad (`Bunch(spin=true)` quaternions); cross-code agreement vs ImpactX is ~0.5% (SciBmad)
to ~11% (Xsuite) — depolarization is bend-model-sensitive, hence a 15% tolerance. Cheetah/pyAT/
PyORBIT have no spin tracking (✗).
⁴ **space charge** — open-boundary FFT PIC. ImpactX/Cheetah/Xsuite use the rigorous **integrated
Green function (IGF)**; PyORBIT uses a cheaper, less-rigorous **point-1/r** Green function (agrees
here on a round beam) → **\*** (its speed is partly model-driven). pyAT/SciBmad have no self-
consistent space charge; Cheetah is 3D-only.

> **Xsuite FFT threading (important for SC perf).** The xfields PIC solve is FFT-bound (~77% of a
> 3D SC step). On CPU, xobjects only threads the FFT through **pyfftw**, which is **broken with
> xfields** — its `FFTCpu` binds the plan to one buffer (`assert data is self.data`) while xfields
> reuses one plan across arrays, so installing pyfftw raises `AssertionError`. Left at the default,
> xfields falls back to **single-threaded numpy/pocketfft**, making Xsuite SC several-fold slower
> than ImpactX (`fftw3_omp`) / PyORBIT. `workers=N` is **not** exposed by Xsuite/xobjects, but
> `Context.plan_FFT()` is — so we inject a **multithreaded `scipy.fft` plan (workers = OMP threads)**
> via a `ContextCpu` subclass (`scenarios/_xsuite_threaded_fft.py`), giving Xsuite a fair threaded
> FFT (~1.8× faster SC, bit-identical results) like the other codes.

> **SC kick splitting (solve count is matched).** The SC *field solve* dominates the cost, so a
> fair comparison requires each code to do the **same number of solves**. Source-verified (against
> the *built* versions) + counted: for the single-drift benchmark **all SC codes do exactly one SC
> solve per track, via a 1st-order asymmetric single kick** — apples-to-apples and consistent.
> Order within the element: **ImpactX** (ed88d69) and **PyORBIT** apply the SC kick at the element
> entrance (kick→drift); **Cheetah** and **Xsuite** apply the thin kick after the drift (drift→kick).
> None does a symmetric Strang split today. Watch: ImpactX is developing a symmetric Strang split
> ([impactx#846](https://github.com/BLAST-ImpactX/impactx/issues/846), not yet landed) whose
> kick-outer form would do **N+1** solves on *non-drift* elements — revisit when it ships or if SC
> is benchmarked on a non-drift element / N>1 slices.

⁶ **FP32** — single-precision variant, run **side-by-side** with FP64 in each scenario plot
(`<code>-cpu-sp` bars). Only ImpactX, Cheetah and SciBmad have FP32 tracking; pyAT/PyORBIT/Xsuite
track in FP64 only (Xsuite's kernels stay double even given float32 input) and Bmad's `rp` is
compile-fixed to double. ImpactX FP32 is a **separate compiled build** in its own `impactx-sp`
env (a clean build dir + `-DImpactX_PRECISION=SINGLE` sets fields *and* particles to single; a
pip install of SP would otherwise overwrite the DP one) and covers all scenarios incl. space
charge. FP32 results are validated against the FP64 ImpactX reference and dashed if outside tolerance.
On CPU only ImpactX (compiled SIMD) gets an FP32 speedup; the Python/Julia codes are FP32-neutral or slower.

**Fair, matched models.** To make the comparison about implementation, not algorithm, every
code is pinned to the *same* physical model per scenario; where a model splits into a fast and an
exact variant, we run **two separate benchmarks** so it stays like-for-like:
* **FODO chromatic** — chromatic-paraxial thick quad (`k1 → k1/(1+δ)`) + chromatic-paraxial drift:
  ImpactX `ChrQuad`+`ChrDrift`, pyAT `QuadLinearPass`+`DriftPass`, Xsuite `mat-kick-mat`+`expanded`,
  PyORBIT `QuadTEAPOT`+`DriftTEAPOT`. Cheetah/SciBmad/Bmad lack a tuned chromatic model and run
  the exact map (marked `*`).
* **FODO exact** — exact nonlinear quad + exact non-paraxial drift: ImpactX `ExactQuad`+`ExactDrift`,
  Cheetah `drift_kick_drift`, pyAT `ExactMultipolePass`+`ExactDriftPass`, Xsuite
  `drift-kick-drift-exact`+`exact`, SciBmad `MatrixKick`+`exact_drift`, Bmad `bmad_standard`
  (the exact map Cheetah/SciBmad port). PyORBIT has no exact quad/drift → unsupported.
* **htu** — the BELLA HTU beamline (chromatic-paraxial quads + drifts; the few weak chicane
  bends use each code's thick bend). Cheetah/SciBmad/Bmad run the costlier exact map (marked `*`).
* **htu_spin** — the same HTU lattice with **Thomas-BMT spin tracking**: the beam starts fully
  spin-aligned (+z) and the dispersive chicane differentially precesses spins → depolarization
  (RMS spin spread). ImpactX, Xsuite (`configure_spin`), SciBmad (`Bunch(spin=true)`) and Bmad
  (`bmad_com%spin_tracking_on`) all run it; Cheetah/pyAT/PyORBIT have no spin tracking.

> **Bmad** is driven by a small standalone Fortran driver linked directly to `libbmad`
> (no Tao/pytao) — see `codes/bmad/driver/` — and is exact-native like Cheetah/SciBmad.
* **space charge** — open-boundary FFT PIC, kept **separate** for 3D and 2.5D (a code without the
  matching solver is *unsupported* there). ImpactX/Cheetah/Xsuite use the rigorous integrated
  Green function; PyORBIT uses a cheaper point-1/r Green function (agrees here) → marked `*`.

## Run it locally

Requires [pixi](https://pixi.sh).

```bash
# build the from-source codes (ImpactX, pyAT, PyORBIT, SciBmad); pip codes need no build
pixi run build

# run the full matrix (no args = all 6 codes x 5 scenarios); or filter freely
pixi run bench
pixi run bench --codes cheetah,pyat,xsuite --scenarios fodo_chromatic,fodo_exact,htu --runs 5

# validate physics & (re)generate plots
pixi run validate
pixi run plot            # writes plots/<scenario>.svg

# host/CPU/compiler/version metadata
pixi run metadata
```

Results are written to `results/<machine>.yaml`; plots to `plots/`. To commit them to the
`benchmarks` branch (CI does this automatically): `pixi run publish --push`.

## How it works

```
benchmarks/      driver: registry, runner, render, validate, metadata, results, plotting, publish
scenarios/<s>/   physics params per scenario: fodo_chromatic, fodo_exact, htu,
                 htu_spin, spacecharge, spacecharge_2p5d
codes/<c>/<s>.<ext>.jinja   per-(code, scenario) run template (python / julia) + build.sh|build.jl
pixi.toml        one isolated environment per code + a driver environment
```

For each `config × scenario × particle-count` the runner **sweeps the code's parallel layouts**
(`ranks × threads ≤ ncores`; see *Benchmark rules*), rendering and running each in the code's pixi
environment with the matching launcher (`mpirun -np <ranks>` + `OMP_NUM_THREADS=<threads>`, or
`julia -t <threads>`), and keeps the **fastest** layout (recorded as `cores`, e.g. `4r x 1t`). Each
run parses `Track: <ns>ns` (wall-clock, min over runs) and `Validate: {json}` (beam observables),
classifies failures (`oom`/`failed`) and physics (`correct`/`incorrect`/`unconverged`), and saves
incrementally. `Code.core_configs()` only ever emits layouts with `ranks*threads ≤ ncores`, and the
runner asserts it before launching. Validation compares each result to the **ImpactX** result for
that scenario (each FODO variant has its own ImpactX reference), with a sampling-aware tolerance
(looser for the dispersive htu and the space-charge scenarios).

**JIT fairness:** every run template does one untimed **warm-up** call before timing, so JIT
codes (Cheetah `torch.compile`, Xsuite cffi kernels, SciBmad/Julia) are measured at
steady state, never including compilation.

## Extending

* **New scenario** — add `scenarios/<name>/params.py` (a `PARAMS` dict; optional
  `analytic_observables()`), register it in `benchmarks/registry.py:SCENARIOS`, and add
  `codes/<code>/<name>.<ext>.jinja` for each code that supports it.
* **New code** — add a `[feature.<code>]` environment in `pixi.toml`, an entry in
  `benchmarks/registry.py:CODES` (capabilities, parallelism, launcher), configs in `CONFIGS`,
  a `codes/<code>/build.*` if it builds from source, and the run templates.

Missing `(code, scenario)` templates show up honestly as **"not in harness"** placeholders, so
coverage can grow incrementally.
