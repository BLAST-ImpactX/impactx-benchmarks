"""Results schema and (de)serialization.

A results file is one YAML document per machine::

    machine: <slug>
    metadata: {...}                       # host/OS/CPU/compiler/versions
    results:
      <config_name>:
        <scenario>_<npart>:
          scenario: drift
          npart: 1000
          status: supported | unsupported_physics | oom | failed
          physics: correct | incorrect | unconverged | model_mismatch | null
          model: 3d-pic | 2.5d-pic | null
          track_ns: <int> | null
          push_per_sec: <float> | null
          observables: {sigma_x: ..., ...} | null
          reason: <str>                    # for unsupported/oom/failed/mismatch

Writes are incremental and atomic so a long run that is interrupted keeps every
measurement collected so far (mirrors the original ``save_timings`` behaviour).
"""

from __future__ import annotations

import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def measurement_key(scenario: str, npart: int) -> str:
    return f"{scenario}_{npart}"


def results_path(machine_slug: str, base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else RESULTS_DIR
    return base / f"{machine_slug}.yaml"


def load(path: Path) -> dict:
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("results", {})
    return data


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save(path: Path, data: dict) -> None:
    text = yaml.safe_dump(data, default_flow_style=False, sort_keys=True)
    _atomic_write(Path(path), text)


def record(data: dict, config: str, scenario: str, npart: int, entry: dict) -> dict:
    """Insert/replace a single measurement in ``data`` (mutates and returns it)."""
    data.setdefault("results", {})
    data["results"].setdefault(config, {})
    full = {"scenario": scenario, "npart": npart}
    full.update(entry)
    data["results"][config][measurement_key(scenario, npart)] = full
    return data


def merge(base: dict, new: dict) -> dict:
    """Deep-merge ``new`` into a copy of ``base`` (new keys win)."""
    from collections.abc import Mapping

    result = deepcopy(base)
    for key, value in new.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


_TRACK_RE = re.compile(r"Track:\s*([0-9]+)\s*ns")
_VALIDATE_RE = re.compile(r"Validate:\s*(\{.*\})")


def parse_track_ns(stdout: str) -> Optional[int]:
    matches = _TRACK_RE.findall(stdout)
    return int(matches[-1]) if matches else None


def parse_observables(stdout: str) -> Optional[dict]:
    import json

    matches = _VALIDATE_RE.findall(stdout)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
