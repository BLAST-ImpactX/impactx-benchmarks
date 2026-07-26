#!/usr/bin/env python3
"""Harness driver for Elegant: run one rendered ``.ele`` deck and emit the harness contract lines.

The runner launches this once per measurement (it handles repetition/min-keeping) as::

    python codes/elegant/driver.py <deck.ele> --ranks N --threads T --cpus 0,1,.. --device cpu|cuda

It selects the right binary (serial ``elegant`` / MPI ``Pelegant`` / ``gpu-elegant``), runs it,
then prints the two machine-readable lines the harness parses::

    Track: <ns>ns
    Validate: {"sigma_x": ..., "sigma_y": ..., "emit_x": ..., "emit_y": ...}

Timing fairness: we do NOT wall-time the whole process (that would include Elegant's startup,
input parsing and Gaussian beam generation -- work the other codes exclude from their track timer).
Instead we parse Elegant's own ``Tracking step completed ... ET:`` elapsed-time line(s), the closest
analogue to the other codes' track-loop timer. Falls back to subprocess wall time if unparseable.

Observables come from the SDDS ``final`` file (``<deck>.fin``) read by ``read_sdds`` (Elegant's own
SDDS dialect isn't parseable by the PyPI ``sdds`` package, and ``sdds2stream`` isn't built).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_sdds import read_parameters, read_columns  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN = REPO_ROOT / "codes" / "elegant" / "bin"
PIN_SCRIPT = REPO_ROOT / "codes" / "pin_rank.sh"
RPN_DEFNS = BIN / "defns.rpn"  # Elegant aborts without RPN_DEFNS pointing at the SDDS defns file

# "Tracking step completed   ET:  00:00:0.142 CP:    1.87 ..." -> elapsed HH:MM:SS.mmm
_ET_RE = re.compile(r"Tracking step completed\s+ET:\s+(\d+):(\d+):([\d.]+)")


def _track_ns(stdout: str) -> int | None:
    """Sum the elapsed time of all tracking steps (n_passes/pages), in nanoseconds."""
    total = 0.0
    found = False
    for h, m, s in _ET_RE.findall(stdout):
        total += int(h) * 3600 + int(m) * 60 + float(s)
        found = True
    return int(round(total * 1e9)) if found else None


def _binary(device: str, ranks: int) -> Path:
    if device == "cuda":
        return BIN / ("gpu-Pelegant" if ranks > 1 else "gpu-elegant")
    return BIN / ("Pelegant" if ranks > 1 else "elegant")


def _std(vals: list) -> float:
    """Centered population standard deviation (matches ImpactX's rbc 'sigma_s*' convention)."""
    n = len(vals)
    m = sum(vals) / n
    return (sum((v - m) ** 2 for v in vals) / n) ** 0.5


def _observables(fin: Path, scenario: str) -> dict:
    """Map Elegant output onto the harness observable keys.

    Tracking scenarios: from the ``final`` SDDS parameters -- Sx/Sy -> sigma_x/sigma_y (RMS beam
    size, m); ex/ey -> emit_x/emit_y (geometric projected emittance incl. dispersion, matching
    ImpactX's projected emittance, NOT the ecx/ecy dispersion-corrected pair).

    Spin (htu_spin): the ``final`` file carries only spin CENTROIDS, so the RMS spin spread
    sigma_sx/sigma_sy is computed from the per-particle spin columns (spx/spy) of the phase-space
    ``output`` (%s.out). Beam starts fully +z-aligned; the chicane depolarizes it.
    """
    if scenario == "htu_spin":
        out = fin.with_suffix(".out")
        cols = read_columns(str(out), ["spx", "spy"])
        return {"sigma_sx": _std(cols["spx"]), "sigma_sy": _std(cols["spy"])}
    p = read_parameters(str(fin))
    return {
        "sigma_x": p["Sx"], "sigma_y": p["Sy"],
        "emit_x": p["ex"], "emit_y": p["ey"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)  # Elegant tracking is MPI-only; threads unused
    ap.add_argument("--cpus", default="")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    deck = Path(args.deck).resolve()
    workdir = deck.parent
    scenario = deck.stem.split("__", 1)[-1]  # elegant__<scenario>.ele
    binary = _binary(args.device, args.ranks)
    if not binary.is_file():
        print(f"Elegant binary not found: {binary} (build the elegant[-gpu] env)", file=sys.stderr)
        return 2

    # Build the command. CPU serial: taskset to the budget. CPU MPI: mpirun + per-rank pinning
    # (1 thread each). CUDA: single GPU (CUDA_VISIBLE_DEVICES set by the runner).
    if args.device == "cuda":
        cmd = [str(binary), deck.name]
    elif args.ranks > 1:
        cmd = ["mpirun", "-np", str(args.ranks), "-bind-to", "none",
               str(PIN_SCRIPT), "1", args.cpus, str(binary), deck.name]
    else:
        cmd = (["taskset", "-c", args.cpus] if args.cpus else []) + [str(binary), deck.name]

    import os
    env = dict(os.environ)
    env["RPN_DEFNS"] = str(RPN_DEFNS)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(workdir), env=env, capture_output=True, text=True)
    wall_ns = int((time.perf_counter() - t0) * 1e9)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
        return proc.returncode

    ns = _track_ns(proc.stdout)
    if ns is None:  # fall back to wall clock (should not happen for a normal track)
        sys.stderr.write("warning: could not parse Elegant tracking ET; using wall time\n")
        ns = wall_ns

    fin = workdir / f"{deck.stem}.fin"
    if not fin.is_file():
        sys.stderr.write(f"Elegant final file not produced: {fin}\n" + proc.stdout[-1500:])
        return 3
    obs = _observables(fin, scenario)

    print(f"Track: {ns}ns")
    print("Validate: " + json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
