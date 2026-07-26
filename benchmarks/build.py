"""Build/install the code environments via pixi.

* pip codes (cheetah, xsuite) -- ``pixi install`` materializes the environment.
* source codes (impactx, pyat, pyorbit) -- run the per-feature ``build-<code>`` task,
  which compiles with the native flags from ``[feature.<code>.activation.env]``.
* julia (scibmad) -- run ``build-scibmad`` to instantiate the Julia project.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .registry import CODES

REPO_ROOT = Path(__file__).resolve().parent.parent

# per-code build task name (None => pip-only, materialized by `pixi install`)
BUILD_TASKS = {
    "impactx": "build-impactx",
    "pyat": "build-pyat",
    "pyorbit": "build-pyorbit",
    "scibmad": "build-scibmad",
    "bmad": "build-bmad",
    "elegant": "build-elegant",
}

# extra CPU (env, build-task) pairs a code needs beyond its main env, e.g. a separate
# single-precision compiled build that lives in its own env.
# extra CPU (env, task) pairs beyond a code's main IEEE env: the SP build and the fast-math (-fm)
# overlay builds for the compile-time codes. (The main impactx env is the DP IEEE baseline.)
EXTRA_ENVS = {
    "impactx": [("impactx-sp", "build-impactx-sp"),
                ("impactx-fm", "build-impactx-fm"),
                ("impactx-sp-fm", "build-impactx-sp-fm")],
    "pyat": [("pyat-fm", "build-pyat-fm")],
    "pyorbit": [("pyorbit-fm", "build-pyorbit-fm")],
    "bmad": [("bmad-fm", "build-bmad-fm")],
}

# GPU (CUDA) envs per code -- built ONLY with --device cuda (never on CPU/CI runs). A code
# absent here has no GPU variant (pyAT/PyORBIT/Bmad stay CPU-only). task=None => pip/pixi-only env.
GPU_ENVS = {
    "impactx": [("impactx-cuda-dp", "build-impactx-cuda-dp"),
                ("impactx-cuda-sp", "build-impactx-cuda-sp"),
                ("impactx-cuda-dp-fm", "build-impactx-cuda-dp-fm"),
                ("impactx-cuda-sp-fm", "build-impactx-cuda-sp-fm")],
    "cheetah": [("cheetah-gpu", None)],
    "xsuite": [("xsuite-gpu", None)],
    "scibmad": [("scibmad-gpu", "build-scibmad-gpu")],
    "elegant": [("elegant-gpu", "build-elegant-gpu")],
}


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def _build_env(env: str, task: str | None) -> bool:
    ok = _run(["pixi", "install", "--environment", env]) == 0
    if ok and task:
        ok = _run(["pixi", "run", "--environment", env, task]) == 0
    return ok


def ensure_built(codes: list[str], device: str = "cpu") -> dict:
    """Install/compile the given codes for ``device``. Returns ``{code: ok_bool}``."""
    status: dict[str, bool] = {}
    for code in codes:
        if device == "cuda":
            envs = GPU_ENVS.get(code, [])
            if not envs:
                print(f"== build {code}: no GPU variant (CPU-only), skipped", flush=True)
                status[code] = True
                continue
            ok = all(_build_env(env, task) for env, task in envs)
        else:
            ok = _build_env(CODES[code].pixi_env, BUILD_TASKS.get(code))
            for env, task in EXTRA_ENVS.get(code, []):
                ok = _build_env(env, task) and ok
        status[code] = ok
        print(f"== build {code} ({device}): {'OK' if ok else 'FAILED'}", flush=True)
    return status


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build/install benchmark code environments.")
    parser.add_argument(
        "--codes",
        default=",".join(CODES),
        help="comma-separated code names (default: all)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="build CPU envs (default) or the CUDA GPU envs")
    args = parser.parse_args(argv)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    status = ensure_built(codes, device=args.device)
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
