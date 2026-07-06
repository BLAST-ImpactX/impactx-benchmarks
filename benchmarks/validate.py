"""Physics-correctness classification.

Every *supported* run that produced observables is classified as one of:

* ``correct``        -- observables within tolerance of the reference, model matches
* ``model_mismatch`` -- ran a different physics model than the scenario intends
                        (e.g. Xsuite 2.5D in the 3D space-charge scenario)
* ``incorrect``      -- converged but out of tolerance (wrong physics)
* ``unconverged``    -- out of tolerance but a sibling (usually the DP run) is correct,
                        i.e. a precision/convergence artefact rather than wrong physics

The reference is either analytic (provided by ``scenarios/<name>/params.py``) or a
designated reference code's double-precision result. Re-runnable standalone via
``pixi run validate`` so classification can evolve without re-benchmarking.
"""

from __future__ import annotations

import importlib
from typing import Optional

import math

from . import results as results_mod
from .registry import CONFIGS, SCENARIOS, Config, Scenario, model_status

# Sampling-aware tolerance: an RMS/emittance estimator from N particles fluctuates
# by O(1/sqrt(N)). Widen every tolerance by SAMPLING_C/sqrt(N) so a physically
# correct finite-N run is not flagged, while gross errors / wrong models still fail.
SAMPLING_C = 3.0


def _load_analytic(scenario: str):
    """Return the scenario's ``analytic_observables(npart) -> dict`` or ``None``."""
    try:
        mod = importlib.import_module(f"scenarios.{scenario}.params")
    except ModuleNotFoundError:
        return None
    return getattr(mod, "analytic_observables", None)


def _compare(obs: dict, ref: dict, keys, tolerances: dict, npart: int) -> tuple[bool, float, str]:
    """Return ``(all_within_tol, worst_rel_err, worst_key)`` over ``keys``.

    Each observable is checked against its own tolerance (falling back to the
    ``"default"`` entry), widened by a sampling term ``SAMPLING_C/sqrt(npart)``.
    """
    default_tol = tolerances.get("default", 1e-3)
    sampling = SAMPLING_C / math.sqrt(max(npart, 1))
    worst_err = 0.0
    worst_key = ""
    all_within = True
    compared = 0
    for key in keys:
        if key not in ref or key not in obs:
            continue
        try:
            ref_f = float(ref[key])
            obs_f = float(obs[key])
        except (TypeError, ValueError):
            continue
        compared += 1
        scale = max(abs(ref_f), 1e-30)
        rel = abs(obs_f - ref_f) / scale
        eff_tol = tolerances.get(key, default_tol) + sampling
        if rel > eff_tol:
            all_within = False
        if rel > worst_err:
            worst_err, worst_key = rel, key
    if compared == 0:
        return False, float("inf"), "no-shared-observables"
    return all_within, worst_err, worst_key


def reference_observables(data: dict, scenario: str, npart: int) -> Optional[dict]:
    """Resolve reference observables for a scenario/npart (analytic or ref code)."""
    sc = SCENARIOS[scenario]
    key = results_mod.measurement_key(scenario, npart)

    if sc.reference == "analytic":
        fn = _load_analytic(scenario)
        if fn is not None:
            ref = fn(npart)
            if ref:
                return ref

    ref_code = "impactx" if sc.reference == "analytic" else sc.reference
    for cfg in CONFIGS.values():
        if cfg.code != ref_code or cfg.precision != "double":
            continue
        # the reference must itself use the intended model
        if sc.intended_model and cfg.sc_model and cfg.sc_model != sc.intended_model:
            continue
        entry = data.get("results", {}).get(cfg.name, {}).get(key)
        if entry and entry.get("observables"):
            return entry["observables"]
    return None


def _dp_sibling_correct(data: dict, cfg: Config, key: str) -> bool:
    """True if the double-precision sibling of a single-precision config is correct."""
    for other in CONFIGS.values():
        if (
            other.code == cfg.code
            and other.precision == "double"
            and other.options == cfg.options
            and other.sc_model == cfg.sc_model
        ):
            entry = data.get("results", {}).get(other.name, {}).get(key)
            return bool(entry and entry.get("physics") == "correct")
    return False


def classify_measurement(data: dict, cfg: Config, sc: Scenario, entry: dict) -> tuple[str, str]:
    """Return ``(physics, reason)`` for one supported measurement with observables."""
    # model mismatch takes precedence and is independent of numeric agreement
    mm = model_status(cfg, sc)
    if mm is not None:
        return "model_mismatch", f"{mm} vs intended {sc.intended_model}"

    obs = entry.get("observables")
    if not obs:
        return "unknown", "no observables"

    ref = reference_observables(data, sc.name, entry["npart"])
    if not ref:
        return "unknown", "no reference"

    within, err, worst = _compare(obs, ref, sc.observables, sc.tolerances, entry["npart"])
    if within:
        return "correct", ""

    reason = f"{worst} off by {err:.2%}"
    if cfg.precision == "single" and _dp_sibling_correct(
        data, cfg, results_mod.measurement_key(sc.name, entry["npart"])
    ):
        return "unconverged", reason
    return "incorrect", reason


def classify_results(data: dict) -> dict:
    """Classify every supported measurement in ``data`` (mutates and returns it)."""
    # Double precision first so SP can reference its DP sibling's verdict.
    def sort_key(item):
        cfg_name = item[0]
        prec = CONFIGS[cfg_name].precision if cfg_name in CONFIGS else "double"
        return 0 if prec == "double" else 1

    for cfg_name, measurements in sorted(data.get("results", {}).items(), key=sort_key):
        if cfg_name not in CONFIGS:
            continue
        cfg = CONFIGS[cfg_name]
        for key, entry in measurements.items():
            if entry.get("status") != "supported":
                continue
            sc = SCENARIOS.get(entry.get("scenario"))
            if sc is None:
                continue
            physics, reason = classify_measurement(data, cfg, sc, entry)
            entry["physics"] = physics
            if reason:
                entry["reason"] = reason
    return data


def main() -> None:
    """Re-classify the stored results file for this machine and save in place."""
    from .metadata import machine_slug

    path = results_mod.results_path(machine_slug())
    data = results_mod.load(path)
    classify_results(data)
    results_mod.save(path, data)
    print(f"Re-validated {path}")


if __name__ == "__main__":
    main()
