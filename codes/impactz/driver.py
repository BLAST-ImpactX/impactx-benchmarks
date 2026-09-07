#!/usr/bin/env python3
"""Harness driver for IMPACT-Z (compiled Fortran/MPI binary, input-file driven).

Invoked once per (ranks, threads) layout by the runner (``benchmarks/runner.py``,
``launcher == "impactz"``)::

    python codes/impactz/driver.py <deck.in> --ranks N --threads T --cpus 0,1,.. --device cpu

Like the Elegant driver, IMPACT-Z's MPI lives INSIDE the binary, so the runner launches
this driver un-wrapped and we build the ``mpirun``/``taskset`` command ourselves.

What we do:
  1. Recover the scenario from the deck filename (``impactz__<scenario>.in``).
  2. Stage the rendered deck as ``ImpactZ.in`` in a work dir (the binary reads that
     hard-wired name from its CWD), rewriting the first data line -- the ``npcol nprow``
     processor grid -- so its product equals ``--ranks`` (IMPACT-Z requires this).
  3. Run ``ImpactZexe-mpi`` under mpirun (>1 rank) or ``taskset`` (1 rank), CWD = work dir.
  4. Read the final beam moments from fort.24/25/26 (see read_fort.py) -> observables.
  5. Print the two harness contract lines: ``Track: <ns>ns`` and ``Validate: {json}``.

Precision/parallelism note (verified from source): IMPACT-Z is double-precision only,
MPI-only (2D domain decomposition; no OpenMP), CPU-only (no GPU path), so ``--threads``
is unused and ``--device cuda`` is rejected.

Timing note: IMPACT-Z prints its own wall time (``time: <sec>``, from MPI_Wtime) and a
per-phase breakdown. We report (total - init) as the track time when available (excludes
lattice/beam setup, comparable to ImpactX timing ``track_particles()``), else the process
wall time. Absolute timings are only meaningful on a quiet machine (a follow-up perf run);
this integration targets correctness.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = REPO_ROOT / "codes" / "impactz" / "bin"
PIN_SCRIPT = REPO_ROOT / "codes" / "pin_rank.sh"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_fort  # noqa: E402


def _proc_grid(ranks: int) -> tuple[int, int]:
    """(npcol, nprow) with npcol*nprow == ranks. We decompose the longitudinal (Z)
    direction (nprow = ranks, npcol = 1): natural and load-balanced for a bunch, and
    the transverse Y stays local so the per-slice FFT is undisturbed. Nz (>= 32 in our
    decks) comfortably exceeds the small rank budget."""
    return 1, max(1, int(ranks))


def _stage_deck(deck: Path, workdir: Path, ranks: int) -> Path:
    """Copy the rendered deck to ``workdir/ImpactZ.in``, rewriting the processor-grid
    line (the first non-comment line) to match ``ranks``."""
    npcol, nprow = _proc_grid(ranks)
    dst = workdir / "ImpactZ.in"
    out, replaced = [], False
    for line in deck.read_text().splitlines():
        if not replaced and line.strip() and not line.strip().startswith("!"):
            out.append(f"{npcol} {nprow}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise ValueError(f"{deck}: no data line to place the processor grid")
    dst.write_text("\n".join(out) + "\n")
    return dst


def _binary(ranks: int) -> Path:
    """The staged IMPACT-Z binary. Prefer the MPI build; fall back to the serial
    (mpistub) build when running a single rank."""
    mpi = BIN_DIR / "ImpactZexe-mpi"
    ser = BIN_DIR / "ImpactZexe"
    if mpi.exists():
        return mpi
    if ranks == 1 and ser.exists():
        return ser
    return mpi  # report the missing MPI binary in the error below


_TOTAL_RE = re.compile(r"time:\s*([0-9.DEed+-]+)")
_PHASE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s+([0-9.DEed+-]+)\s+seconds\s*$")


def _track_ns(stdout: str, wall_ns: int) -> int:
    """Tracking-only time in ns: IMPACT-Z's total wall (MPI_Wtime) minus the ``init``
    (setup) phase when both parse and are non-zero; else the total; else process wall."""
    def _f(s):
        try:
            return float(s.replace("D", "E").replace("d", "e"))
        except ValueError:
            return None

    total = None
    for m in _TOTAL_RE.findall(stdout):
        v = _f(m)
        if v is not None:
            total = v
    init = 0.0
    for name, val in _PHASE_RE.findall(stdout):
        if name == "init":
            v = _f(val)
            if v is not None:
                init = v
    if total and total > 0:
        return int(round(max(total - init, 0.0) * 1e9)) or wall_ns
    return wall_ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)  # IMPACT-Z is MPI-only; unused
    ap.add_argument("--cpus", default="")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.device == "cuda":
        sys.stderr.write("IMPACT-Z has no GPU build (CPU Fortran/MPI only)\n")
        return 2

    deck = Path(args.deck).resolve()
    workdir = deck.parent
    scenario = deck.stem.split("__", 1)[-1]  # impactz__spacecharge -> spacecharge

    binary = _binary(args.ranks)
    if not binary.exists():
        sys.stderr.write(f"IMPACT-Z binary not found: {binary} (run `pixi run build-impactz`)\n")
        return 2

    impactz_in = _stage_deck(deck, workdir, args.ranks)

    if args.ranks > 1:
        cmd = ["mpirun", "-np", str(args.ranks), "-bind-to", "none",
               str(PIN_SCRIPT), "1", args.cpus, str(binary)]
    else:
        cmd = (["taskset", "-c", args.cpus] if args.cpus else []) + [str(binary)]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
    wall_ns = int((time.perf_counter() - t0) * 1e9)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
        return proc.returncode

    try:
        obs = read_fort.observables(workdir, impactz_in)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to the runner
        sys.stderr.write(f"IMPACT-Z output parse failed ({scenario}): {exc}\n")
        sys.stderr.write(proc.stdout[-1500:])
        return 3

    ns = _track_ns(proc.stdout, wall_ns)
    print(f"Track: {ns}ns")
    print("Validate: " + json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
