"""Orchestration loop: build envs, run the matrix, classify, save.

For every ``config x scenario x npart`` the runner:

1. checks support (skips with ``unsupported_physics`` if the code can't do it),
2. renders the run template,
3. launches it in the code's pixi environment with the right launcher
   (``python`` / ``mpirun -np N`` / ``julia -t N``) and thread settings,
4. parses ``Track: <ns>ns`` (min over ``--runs``) and ``Validate: {json}``,
5. classifies failures (``oom`` / ``failed``) without aborting the matrix,
6. records the measurement and saves incrementally.

After the matrix, physics classification runs over the full result set.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from . import build as build_mod
from . import metadata as meta_mod
from . import render as render_mod
from . import results as results_mod
from . import validate as validate_mod
from .affinity import budget_cpus
from .registry import CODES, CONFIGS, SCENARIOS, Config, Scenario, configs_for, support_status

REPO_ROOT = Path(__file__).resolve().parent.parent
PIN_SCRIPT = REPO_ROOT / "codes" / "pin_rank.sh"
BMAD_DRIVER = REPO_ROOT / "codes" / "bmad" / "bmad_driver"  # compiled by codes/bmad/build.sh
# Per-run manifests: the template-resolved input file(s) + a runnable `run.sh` (exact launch
# command + harness env) for each benchmark cell -- so the LLM-generated templates can be
# scrutinized by the codes' authors. Gitignored on main; published to the `benchmarks` branch
# alongside results/plots. Run OUTPUTS (stdout/diags/data) are deliberately NOT persisted here.
RUNS_DIR = REPO_ROOT / "runs"
# env vars the harness itself controls -- listed in every run.sh so the launch is reproducible
_HARNESS_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "JULIA_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES", "MPICH_GPU_SUPPORT_ENABLED",
    "TORCHINDUCTOR_USE_FAST_MATH", "PYTHONPATH",
)

OOM_MARKERS = (
    "MemoryError",
    "std::bad_alloc",
    "out of memory",
    "OutOfMemory",
    "CUDA out of memory",
    "Killed",
    "Cannot allocate memory",
    "bad_alloc",
)


def default_ncores() -> int:
    env = os.environ.get("BENCH_NCORES")
    if env and env.isdigit():
        return int(env)
    return os.cpu_count() or 1


# --------------------------------------------------------------------------- #
# Launch a single rendered run script
# --------------------------------------------------------------------------- #
def _runtime_env(cfg: Config, threads: int) -> dict:
    """Environment variables for the launched process (per-rank thread count)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # OpenMP/MKL threads per process (= per MPI rank). MPI rank count is set by mpirun.
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["JULIA_NUM_THREADS"] = str(threads)
    if cfg.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif cfg.device == "cuda":
        # 1 GPU by default (override with BENCH_CUDA_DEVICES; multi-GPU mapping added later)
        env["CUDA_VISIBLE_DEVICES"] = env.get("BENCH_CUDA_DEVICES", "0")
        # GPU-aware MPI: enable Cray-MPICH's GPU support so AMReX (ImpactX) passes device
        # pointers straight to MPI on multi-GPU runs (avoids d2h staging copies). AMReX
        # auto-detects GPU-aware MPI via MPIX_GPU_query_support and only turns it on when the
        # MPI reports support -- so this is the right "on where supported" knob: honored by Cray
        # MPICH (Perlmutter), a harmless no-op on our conda-forge MPICH (which is NOT GPU-aware:
        # its query stays 0, so AMReX correctly keeps it off and never passes device ptrs to it).
        env.setdefault("MPICH_GPU_SUPPORT_ENABLED", "1")
    # Fast-math per config. Build-time codes (ImpactX/pyAT/PyORBIT/Bmad) already carry it in their
    # env; this only drives the RUNTIME-toggle codes: Xsuite (make_context reads BENCH_FASTMATH for
    # the cffi/cupy compile) and Cheetah/Inductor (TORCHINDUCTOR_USE_FAST_MATH).
    env["BENCH_FASTMATH"] = "1" if cfg.fast_math else "0"
    if cfg.fast_math and cfg.code == "cheetah":
        env["TORCHINDUCTOR_USE_FAST_MATH"] = "1"
    return env


def _launch_cmd(cfg: Config, script: Path, ranks: int, threads: int, cpus: list) -> list[str]:
    """Build the launch command for a (ranks, threads) layout, physically pinned to the
    shared ``cpus`` budget set so MPI and OpenMP layouts compete on identical cores."""
    code = CODES[cfg.code]
    base = ["pixi", "run", "--environment", cfg.pixi_env]
    cpu_csv = ",".join(str(c) for c in cpus)
    if code.launcher == "julia":
        # GPU uses the scibmad-gpu project (adds CUDA.jl); CPU uses the threaded scibmad project
        project = "codes/scibmad-gpu" if cfg.device == "cuda" else "codes/scibmad"
        inner = ["julia", f"-t{threads}", f"--project={project}", str(script)]
    elif code.launcher == "bmad":
        # compiled Fortran driver (links conda libbmad); `script` is the .in namelist
        inner = [str(BMAD_DRIVER), str(script)]
    else:
        inner = ["python", str(script)]
    # GPU: no CPU taskset pinning. 1 GPU = 1 process; multi-GPU (mpirun + per-rank GPU binding
    # via CUDA_VISIBLE_DEVICES) will be added when we test >1 GPU on Perlmutter.
    if cfg.device == "cuda":
        if ranks > 1 or code.launcher == "mpirun":
            return base + ["mpirun", "-np", str(ranks), "-bind-to", "none"] + inner
        return base + inner
    # Use mpirun for >1 rank, or whenever the code launches via MPI (always-MPI codes).
    if ranks > 1 or code.launcher == "mpirun":
        # disable mpirun's own binding; pin each rank ourselves to its slice of `cpus`
        pin = [str(PIN_SCRIPT), str(threads), cpu_csv]
        return base + ["mpirun", "-np", str(ranks), "-bind-to", "none"] + pin + inner
    # single process (OpenMP / Julia threads): confine to the whole budget set
    return base + ["taskset", "-c", cpu_csv] + inner


def _classify_failure(returncode: int, stderr: str) -> tuple[str, str]:
    blob = stderr or ""
    if returncode in (137, 139) or any(m in blob for m in OOM_MARKERS):
        return "oom", _last_error_line(blob)
    return "failed", _last_error_line(blob)


def _last_error_line(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""


def _run_layout(cfg, sc, npart, ranks, threads, nruns, ncores, cpus):
    """Run one (ranks, threads) layout nruns times. Returns (track_ns|None, obs, status, reason)."""
    # Hard fairness invariant: MPI ranks * OMP threads must never exceed the core budget.
    if ranks * threads > ncores:
        raise ValueError(
            f"layout {ranks}r x {threads}t = {ranks * threads} cores exceeds budget "
            f"{ncores} (MPI ranks * OMP threads must be <= ncores)"
        )
    context = _render_context(cfg, sc, npart, threads)
    with tempfile.TemporaryDirectory(prefix="benchrun_") as tmp:
        script = render_mod.render_run_script(cfg.code, sc.name, context, Path(tmp))
        cmd = _launch_cmd(cfg, script, ranks, threads, cpus)
        env = _runtime_env(cfg, threads)
        best_ns = None
        observables = None
        for _ in range(nruns):
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                status, reason = _classify_failure(proc.returncode, proc.stderr)
                return None, None, status, reason
            ns = results_mod.parse_track_ns(proc.stdout)
            if ns is None:
                return None, None, "failed", "no Track line"
            best_ns = ns if best_ns is None else min(best_ns, ns)
            obs = results_mod.parse_observables(proc.stdout)
            if obs is not None:
                observables = obs
    return best_ns, observables, "supported", ""


def _write_run_manifest(cfg: Config, sc: Scenario, npart: int, ranks: int, threads: int,
                        cpus: list, status: str, slug: str) -> None:
    """Persist the template-resolved input file(s) + a runnable ``run.sh`` (exact launch command
    + harness env vars) for one benchmark cell, under ``runs/<machine>/``. Faithful to the winning
    layout (or the last attempt on failure). Never raises -- writing a manifest must not break a
    run. Run OUTPUTS (stdout/diags/data files) are deliberately NOT persisted."""
    try:
        cell = RUNS_DIR / slug / f"{cfg.name}__{sc.name}__n{npart}"
        cell.mkdir(parents=True, exist_ok=True)
        for old in cell.glob("*"):  # drop stale files so a re-run reflects the current template
            if old.is_file():
                old.unlink()
        context = _render_context(cfg, sc, npart, threads)
        script = render_mod.render_run_script(cfg.code, sc.name, context, cell)
        # repo-relative script path keeps the reproduction command readable (not machine-absolute)
        cmd = _launch_cmd(cfg, script.relative_to(REPO_ROOT), ranks, threads, cpus)
        env = _runtime_env(cfg, threads)
        keys = sorted(k for k in set(_HARNESS_ENV_KEYS) | set(env)
                      if k in env and (k in _HARNESS_ENV_KEYS or os.environ.get(k) != env[k]))
        head = [
            "#!/usr/bin/env bash",
            f"# FAITHFUL run manifest -- {cfg.name} / {sc.name} / n={npart}",
            f"#   status: {status}   layout: {ranks}r x {threads}t   env: {cfg.pixi_env}"
            f"   device: {cfg.device}   precision: {cfg.precision}",
            "# The input file(s) in this folder are the template-resolved script that ran; the",
            "# command below is exactly how the harness launched it (auto-written by",
            "# benchmarks/runner.py for author review / reproduction). Run from the repo root.",
            "# Run OUTPUTS are NOT persisted -- only the inputs + this launch script.",
            "set -eu",
        ]
        body = [f"export {k}={shlex.quote(env[k])}" for k in keys]
        run_sh = cell / "run.sh"
        run_sh.write_text("\n".join(head + [""] + body + ["", "exec " + shlex.join(cmd)]) + "\n")
        run_sh.chmod(0o755)
    except Exception as exc:  # noqa: BLE001 -- logging must never abort the matrix
        print(f"   (run manifest skipped for {cfg.name}/{sc.name} n={npart}: {exc})", flush=True)


def run_one(cfg: Config, sc: Scenario, npart: int, ncores: int, nruns: int, slug: str) -> dict:
    """Sweep the code's (ranks, threads) layouts (<= ncores) and keep the fastest."""
    code = CODES[cfg.code]
    # GPU: a single 1-GPU layout (no rank/thread sweep); CPU: the code's (ranks, threads) sweep.
    layouts = [(1, 1)] if cfg.device == "cuda" else code.core_configs(ncores)
    cpus = budget_cpus(ncores)  # same physical-core set for every CPU layout (fair)
    best = None  # (track_ns, obs, ranks, threads)
    last_status, last_reason = "failed", "no layout ran"
    last_layout = layouts[0] if layouts else (1, 1)
    for ranks, threads in layouts:
        last_layout = (ranks, threads)
        ns, obs, status, reason = _run_layout(cfg, sc, npart, ranks, threads, nruns, ncores, cpus)
        if status != "supported" or ns is None:
            last_status, last_reason = status, reason
            print(f"   [{cfg.name}] {sc.name} n={npart} [{ranks}r x {threads}t]: "
                  f"{status} ({reason})", flush=True)
            continue
        if best is None or ns < best[0]:
            best = (ns, obs, ranks, threads)

    if best is None:
        # faithful manifest of the last attempt, so the failure is reproducible/reviewable
        _write_run_manifest(cfg, sc, npart, last_layout[0], last_layout[1], cpus, last_status, slug)
        return {"status": last_status, "reason": last_reason, "physics": None,
                "model": cfg.sc_model, "track_ns": None, "push_per_sec": None,
                "observables": None, "cores": None}

    track_ns, observables, ranks, threads = best
    _write_run_manifest(cfg, sc, npart, ranks, threads, cpus, "supported", slug)
    push_per_sec = npart / track_ns * 1e9
    cores = f"{ranks}r x {threads}t"
    print(f"   [{cfg.name}] {sc.name} n={npart}: {track_ns} ns "
          f"({push_per_sec:.3e} part/s)  best layout {cores}", flush=True)
    return {"status": "supported", "reason": "", "physics": None,
            "model": cfg.sc_model, "track_ns": track_ns,
            "push_per_sec": push_per_sec, "observables": observables, "cores": cores}


def _render_context(cfg: Config, sc: Scenario, npart: int, threads: int) -> dict:
    """Build the Jinja context: scenario params + config knobs + runtime."""
    import importlib

    params = {}
    try:
        mod = importlib.import_module(f"scenarios.{sc.name}.params")
        params = dict(getattr(mod, "PARAMS", {}))
    except ModuleNotFoundError:
        pass

    context = {
        "code": cfg.code,
        "scenario": sc.name,
        "npart": npart,
        "threads": threads,   # per-process thread count for this layout
        "ncores": threads,    # backward-compat alias for templates
        "precision": cfg.precision,
        "dtype": "float64" if cfg.precision == "double" else "float32",
        "device": cfg.device,
        "fast_math": cfg.fast_math,   # Cheetah/Inductor template reads this (runtime toggle)
        "params": params,
        **cfg.options,
    }
    # For htu/htu_spin in non-Python codes (e.g. SciBmad/Julia), bake a code-agnostic
    # element spec (physics computed by the shared Python htu_lattice) into the template.
    if sc.name in ("htu", "htu_spin"):
        try:
            mod = importlib.import_module("scenarios.htu.htu_lattice")
            context["htu_spec"] = mod.get_lattice("spec", screens_as_markers=True)
        except Exception:
            context["htu_spec"] = []
    return context


# --------------------------------------------------------------------------- #
# Matrix driver
# --------------------------------------------------------------------------- #
def run_matrix(args) -> Path:
    ncores = args.ncores or default_ncores()
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else list(CODES)
    scenarios = (
        [s.strip() for s in args.scenarios.split(",")] if args.scenarios else list(SCENARIOS)
    )
    nparts_override = (
        [int(n) for n in args.nparts.split(",")] if args.nparts else None
    )
    precisions = (
        {p.strip() for p in args.precision.split(",")} if args.precision else None
    )

    # metadata + (optionally) build -- device-aware: a cuda run builds the GPU envs
    if not args.skip_build:
        build_mod.ensure_built(codes, device=args.device)

    host = meta_mod.host_metadata()
    # record package versions from the envs actually used this run (GPU envs on a cuda run)
    if args.device == "cuda":
        version_envs = sorted({cfg.pixi_env for cfg in configs_for(codes)
                               if cfg.device == "cuda"})
    else:
        version_envs = [CODES[c].pixi_env for c in codes]
    versions = meta_mod.code_versions(version_envs)
    # capture ALL codes' versions (not just this run's subset) so a filtered run does not
    # wipe the version footer for the other codes already in the results file.
    code_version = meta_mod.code_version_labels(list(CODES))
    slug = host["machine_slug"]
    path = results_mod.results_path(slug)
    data = results_mod.load(path)
    data["machine"] = slug
    data["metadata"] = {"host": host, "versions": versions, "code_version": code_version}
    results_mod.save(path, data)

    for cfg in configs_for(codes):
        if cfg.device != args.device:
            continue
        if precisions and cfg.precision not in precisions:
            continue
        for sc_name in scenarios:
            sc = SCENARIOS[sc_name]
            status, reason = support_status(cfg, sc)
            # GPU runs sweep the larger nparts_gpu (single A100); CPU keeps sc.nparts. --nparts wins.
            sweep = nparts_override or list(sc.sweep(cfg.device))
            for npart in sweep:
                if status != "supported":
                    entry = {"status": status, "reason": reason, "physics": None,
                             "model": cfg.sc_model, "track_ns": None,
                             "push_per_sec": None, "observables": None}
                    print(f"   [{cfg.name}] {sc.name} n={npart}: {status} ({reason})",
                          flush=True)
                else:
                    if not render_mod.template_exists(cfg.code, sc.name):
                        # the code CAN do this physics, but THIS harness has no run
                        # template for (code, scenario) yet -- distinct from the code
                        # being physically incapable ("unsupported_physics").
                        entry = {"status": "not_in_harness",
                                 "reason": "run template not yet added to harness",
                                 "physics": None, "model": cfg.sc_model,
                                 "track_ns": None, "push_per_sec": None,
                                 "observables": None}
                    else:
                        entry = run_one(cfg, sc, npart, ncores, args.runs, slug)
                results_mod.record(data, cfg.name, sc.name, npart, entry)
                results_mod.save(path, data)

    # physics classification over the full result set
    validate_mod.classify_results(data)
    results_mod.save(path, data)
    print(f"\nResults written to {path}")
    return path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the beam-dynamics benchmark matrix.")
    p.add_argument("--codes", default="", help="comma-separated codes (default: all)")
    p.add_argument("--scenarios", default="", help="comma-separated scenarios (default: all)")
    p.add_argument("--nparts", default="", help="comma-separated particle counts (override sweep)")
    p.add_argument("--runs", type=int, default=5, help="repetitions per measurement (min kept)")
    p.add_argument("--ncores", type=int, default=0, help="cores to use (default: nproc / $BENCH_NCORES)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device to benchmark")
    p.add_argument("--precision", default="", help="comma-separated: single,double (default: both)")
    p.add_argument("--skip-build", action="store_true", help="assume environments already built")
    p.add_argument("--write-manifests", action="store_true",
                   help="don't run: (re)write runs/<machine>/ manifests for the stored results "
                        "(faithful to each cell's recorded winning layout)")
    return p


def write_manifests(ncores: int | None) -> int:
    """Backfill the ``runs/<machine>/`` manifests from the stored results file, without running:
    re-render each recorded cell at its winning ``cores`` layout. Faithful, and lets the resolved
    templates be reviewed even for runs already completed."""
    import re
    ncores = ncores or default_ncores()
    cpus = budget_cpus(ncores)
    slug = meta_mod.host_metadata()["machine_slug"]
    data = results_mod.load(results_mod.results_path(slug))
    n = 0
    for cfg_name, measurements in data.get("results", {}).items():
        cfg = CONFIGS.get(cfg_name)
        if cfg is None:
            continue
        for entry in measurements.values():
            sc_name, npart = entry.get("scenario"), entry.get("npart")
            sc = SCENARIOS.get(sc_name) if sc_name else None
            if sc is None or npart is None or not render_mod.template_exists(cfg.code, sc_name):
                continue  # only cells whose template actually gets rendered/run
            if entry.get("status") not in ("supported", "failed", "oom"):
                continue  # skip unsupported_physics / not_in_harness (nothing was launched)
            m = re.match(r"(\d+)r x (\d+)t", entry.get("cores") or "")
            ranks, threads = (int(m[1]), int(m[2])) if m else ((1, 1) if cfg.device == "cuda"
                                                               else (1, ncores))
            _write_run_manifest(cfg, sc, npart, ranks, threads, cpus, entry.get("status"), slug)
            n += 1
    print(f"wrote {n} run manifests under runs/{slug}/")
    return 0


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.ncores == 0:
        args.ncores = None
    if args.write_manifests:
        return write_manifests(args.ncores)
    run_matrix(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
