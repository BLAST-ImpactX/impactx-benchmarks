"""Status/physics-aware bar charts (particles per second).

Rendering rules
---------------
* ``correct``                       -> solid bar, value label
* ``model_mismatch`` / ``incorrect`` / ``unconverged``
                                    -> bar drawn with a **dashed outline** and a
                                       clear marker ("physics x", "unconverged",
                                       or the model name e.g. "2.5D")
* ``unsupported_physics``           -> greyed placeholder labelled "unsupported"
* ``oom`` / ``failed``              -> greyed placeholder labelled "OOM" / "failed"

Y-axis: the metric is particles/second (higher = faster), so the fastest code is the
tallest bar. The y-axis always scales to fully show that winner (with headroom for its
value label) -- bars are never clipped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import results as results_mod  # noqa: E402
from .registry import CODES, CONFIGS, SCENARIOS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "plots"

CODE_COLORS = {
    "impactx": "tab:red",
    "cheetah": "tab:blue",
    "pyat": "tab:green",
    "pyorbit": "tab:purple",
    "xsuite": "tab:orange",
    "scibmad": "tab:brown",
    "elegant": "tab:cyan",
}

DASHED_PHYSICS = {"model_mismatch", "incorrect", "unconverged"}
# Placeholder statuses (no bar height). Note the deliberate distinction:
#   * unsupported_physics -> the CODE cannot do this physics
#   * not_in_harness      -> the code can, but this HARNESS lacks the run template
#   * sp_na                -> a DP-only code: a grey "no SP" stub next to its DP bar
PLACEHOLDER_STATUS = {"unsupported_physics", "oom", "failed", "not_in_harness", "sp_na"}
PLACEHOLDER_LABELS = {
    "unsupported_physics": "physics model\nunsupported",
    "oom": "OOM",
    "failed": "failed",
    "not_in_harness": "not in harness",
    "sp_na": "no SP",
}

YHEADROOM = 1.28  # extra y-axis space above the tallest bar for its value label


def _physics_marker(entry: dict) -> str:
    physics = entry.get("physics")
    if physics == "model_mismatch":
        model = (entry.get("model") or "").replace("-pic", "").upper()
        return model or "model"
    if physics == "unconverged":
        return "unconv."
    if physics == "incorrect":
        return "physics ✗"
    return ""


#: Published plots focus on 100k-particle beams and above. Smaller counts may still be
#: run (e.g. for scaling/crossover studies) but are not the headline published number,
#: because at small N some scenarios (esp. space charge) are grid/FFT-bound rather than
#: per-particle and so are unrepresentative.
PUBLISHED_MIN_NPART = 100_000


def _select_npart(data: dict, scenario: str, npart_override=None) -> int | None:
    """Pick the published particle count: among counts >= PUBLISHED_MIN_NPART, the one
    with the most configs measured (ties broken toward the larger count). Falls back to
    all counts if none reach the floor."""
    if npart_override:
        return npart_override
    counts: dict[int, int] = {}
    for measurements in data.get("results", {}).values():
        for entry in measurements.values():
            if entry.get("scenario") == scenario:
                n = entry.get("npart")
                counts[n] = counts.get(n, 0) + 1
    if not counts:
        return None
    pool = {n: c for n, c in counts.items() if n >= PUBLISHED_MIN_NPART} or counts
    return max(pool, key=lambda n: (pool[n], n))


def _entries_for(data: dict, scenario: str, npart: int, device: str | None = None) -> list[tuple]:
    """``(config_name|None, entry, label, code, fm_entry)`` per bar, grouped DP|SP per code.
    ``device`` "cpu"/"gpu" restricts to that device's configs; None = both (combined).

    Layout rules:
      * bars are the IEEE (non-fast-math) configs; each carries its fast-math sibling's entry
        (``fm_entry``, or None) which the plot draws as a lighter bar BEHIND it;
      * per config "base" (name minus ``-dp``/``-sp``) the DP bar is followed by the SP bar, with a
        grey "no SP" stub if the code has no SP build (so DP|SP pairs line up);
      * a code entirely unsupported for this scenario collapses to a SINGLE bar.
    ``code`` is the grouping key (x-spacing changes with it).
    """
    key = results_mod.measurement_key(scenario, npart)
    results = data.get("results", {})
    want_dev = {"cpu": "cpu", "gpu": "cuda"}.get(device)  # None => both devices (combined)
    per_code: dict[str, list[tuple]] = {}
    for cfg_name, cfg in CONFIGS.items():
        if cfg.fast_math:
            continue  # fast-math variants are overlays on their IEEE sibling, not standalone bars
        if want_dev is not None and cfg.device != want_dev:
            continue  # device-filtered (_cpu / _gpu) view
        entry = results.get(cfg_name, {}).get(key)
        if entry is not None:
            fm_entry = results.get(cfg_name + "-fm", {}).get(key) if (cfg_name + "-fm") in CONFIGS else None
            per_code.setdefault(cfg.code, []).append((cfg_name, cfg, entry, fm_entry))
    out: list[tuple] = []
    for code in CODES:
        items = per_code.get(code)
        if not items:
            continue
        items.sort(key=lambda t: t[0])
        if all(e.get("status") == "unsupported_physics" for _, _, e, _ in items):
            cfg_name, _, entry, fm_entry = items[0]
            out.append((cfg_name, entry, code, code, fm_entry))  # one collapsed bar, labelled by code
            continue
        pairs: dict[str, dict] = {}
        order: list[str] = []
        for cfg_name, cfg, entry, fm_entry in items:
            base = re.sub(r"-(dp|sp)$", "", cfg_name)
            if base not in pairs:
                pairs[base] = {}
                order.append(base)
            pairs[base][cfg.precision] = (cfg_name, entry, fm_entry)
        for base in order:
            pr = pairs[base]
            if "double" in pr:
                cn, en, fe = pr["double"]
                out.append((cn, en, cn, code, fe))
            if "single" in pr:
                cn, en, fe = pr["single"]
                out.append((cn, en, cn, code, fe))
            else:  # DP-only code -> grey "no SP" stub in the SP slot
                out.append((None, {"status": "sp_na", "physics": None}, f"{base}-sp", code, None))
    return out


# --------------------------------------------------------------------------- #
# GPU (FP32) cross-code comparison
# --------------------------------------------------------------------------- #
# Each code shows ONE bar: its best config along this preference ladder toward the headline
# "GPU FP32". A code whose best is below the top rung (no GPU-FP32 build) shows its next-best
# instead, marked with an asterisk + a per-code caveat in the footnote -- in analogy to the
# CPU plots' asterisk for codes running a costlier untuned model.
_GPU_LADDER = [
    ("cuda", "single"),   # 0: the headline config -- real GPU FP32 (no caveat)
    ("cuda", "double"),   # 1: GPU, but FP64 only (no FP32 build) -> Xsuite
    ("cpu", "single"),    # 2: no GPU; best is CPU FP32
    ("cpu", "double"),    # 3: no GPU; best is CPU FP64       -> pyAT / PyORBIT / Bmad
]
_GPU_RUNG = {dp: i for i, dp in enumerate(_GPU_LADDER)}
_GPU_CAVEAT = {
    1: "GPU FP64 (no FP32)",
    2: "CPU FP32 (no GPU)",
    3: "CPU FP64 (no GPU)",
}


def _gpu_entries_for(data: dict, scenario: str, npart: int) -> list[tuple[str, dict, str, int]]:
    """``(cfg_name, entry, code, rung)`` per code -- the best *supported* config toward
    GPU-FP32 (lowest ladder rung; ties within a rung broken by fastest). ``rung`` 0 means a
    real GPU-FP32 result; >0 means the next-best fallback shown (see ``_GPU_CAVEAT``)."""
    key = results_mod.measurement_key(scenario, npart)
    out: list[tuple[str, dict, str, int]] = []
    for code in CODES:
        best_sort = None            # (rung, -push, cfg_name) -- min() picks preferred+fastest
        best = None                 # (cfg_name, entry, rung)
        for cfg_name, cfg in CONFIGS.items():
            if cfg.code != code or cfg.fast_math:
                continue  # IEEE bars only (fast-math variants aren't shown in the GPU comparison)
            rung = _GPU_RUNG.get((cfg.device, cfg.precision))
            if rung is None:
                continue
            entry = data.get("results", {}).get(cfg_name, {}).get(key)
            if not entry or entry.get("status") != "supported":
                continue
            sort_key = (rung, -(entry.get("push_per_sec") or 0.0), cfg_name)
            if best_sort is None or sort_key < best_sort:
                best_sort, best = sort_key, (cfg_name, entry, rung)
        if best is not None:
            out.append((best[0], best[1], code, best[2]))
    return out


def _compute_ylim(heights: list[float]) -> float:
    """Y-axis top: fully show the fastest (tallest) bar, with label headroom."""
    pos = [h for h in heights if h and h > 0]
    return max(pos) * YHEADROOM if pos else 1.0


def plot_scenario(data: dict, scenario: str, npart=None, out_dir: Path = PLOTS_DIR,
                  device: str | None = None) -> Path | None:
    npart = _select_npart(data, scenario, npart)
    if npart is None:
        return None
    entries = _entries_for(data, scenario, npart, device=device)
    if not entries:
        return None  # nothing on this device for this scenario -> no _cpu/_gpu file

    sc = SCENARIOS.get(scenario)
    untuned = sc.untuned_codes if sc else {}
    any_untuned = False

    labels = [lbl for _, _, lbl, _, _ in entries]
    codes = [cd for _, _, _, cd, _ in entries]
    heights, fm_heights = [], []
    for _, e, _, _, fe in entries:
        h = e.get("push_per_sec") if e.get("status") == "supported" else 0.0
        heights.append(h or 0.0)
        fh = fe.get("push_per_sec") if (fe and fe.get("status") == "supported") else 0.0
        fm_heights.append(fh or 0.0)

    ymax = _compute_ylim(heights + fm_heights)  # the fast-math overlay is usually the tallest

    # x-positions: bars step by 1 within a code, with an extra gap when the code changes,
    # so each code's DP|SP bars group together with whitespace before the next code.
    GROUP_GAP = 0.4
    xs: list[float] = []
    xpos, prev = 0.0, None
    for cd in codes:
        if prev is not None and cd != prev:
            xpos += GROUP_GAP
        xs.append(xpos)
        xpos += 1.0
        prev = cd
    span = (xs[-1] + 1.0) if xs else 1.0

    fig, ax = plt.subplots(figsize=(max(5.0, 0.62 * span), 3.2))

    any_fm = False
    for i, (cfg_name, entry, label, code, fm_entry) in enumerate(entries):
        xi = xs[i]
        color = CODE_COLORS.get(code, "gray")
        status = entry.get("status")
        physics = entry.get("physics")
        h = heights[i]

        if status in PLACEHOLDER_STATUS:
            # placeholder: thin greyed bar near the floor with a status label
            ax.bar(xi, ymax * 0.02, color="lightgray", edgecolor="gray", width=0.8)
            ptext = PLACEHOLDER_LABELS.get(status, status)
            ax.text(xi, ymax * 0.03, ptext, ha="center", va="bottom",
                    fontsize=7, rotation=90, color="dimgray")
            continue

        # fast-math overlay: same-colour, lighter bar drawn BEHIND (lower zorder). Fast-math is
        # usually faster, so its extra height peeks above the solid IEEE bar = the speedup.
        fmh = fm_heights[i]
        if fmh > 0:
            # A fast-math run can be FASTER while being physically WRONG (seen: cheetah SP
            # space charge, emit_x off by ~8e9% under -ffast-math). Mark such an overlay with
            # the same dashed+hatched convention the solid bars use, so it can never read as a
            # credible speedup just because it is tall.
            fm_bad = (fm_entry or {}).get("physics") in DASHED_PHYSICS
            fmbar = ax.bar(xi, fmh, color=color, width=0.8, alpha=0.28,
                           linewidth=1.0 if fm_bad else 0, zorder=1,
                           edgecolor="black" if fm_bad else "none")[0]
            if fm_bad:
                fmbar.set_linestyle((0, (4, 2)))
                fmbar.set_hatch("//")
            # Annotate the fast-math top only when it clears the IEEE bar's own value label (a
            # 1-2 line 6.5pt block sitting directly above that bar) -- otherwise the two labels
            # overlap for the common case of a small speedup. The overlay bar still shows it.
            label_block = ymax * (0.02 + 0.055 * (2 if entry.get("cores") else 1))
            if fmh > h + label_block:
                ax.text(xi, fmh + ymax * 0.01, f"{fmh:.1e}", ha="center", va="bottom",
                        fontsize=5.5, color=color, alpha=0.9)
            any_fm = True

        dashed = physics in DASHED_PHYSICS
        # full height -- the fastest bar (winner) is always shown completely
        bar = ax.bar(xi, h, color=color, edgecolor="black", width=0.8,
                     linewidth=1.3, alpha=0.55 if dashed else 0.95, zorder=2)[0]
        if dashed:
            bar.set_linestyle((0, (4, 2)))
            bar.set_hatch("//")

        # value + winning core layout (ranks x threads, <= 4 cores) + physics marker.
        # an asterisk marks codes running a costlier, untuned model (see footnote).
        cores = entry.get("cores")
        star = " *" if code in untuned else ""
        if star:
            any_untuned = True
        vlabel = f"{h:.1e}{star}" + (f"\n{cores}" if cores else "")
        ax.text(xi, h + ymax * 0.01, vlabel, ha="center", va="bottom", fontsize=6.5)
        marker = _physics_marker(entry)
        if marker:
            # status marker as a small line in the empty space above the value label, rather
            # than over the bar where it collides with the hatch/label
            ax.text(xi, min(h + ymax * 0.22, ymax * 0.92), marker, ha="center", va="bottom",
                    fontsize=6.5, color="black", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("particles / second")
    ref = f" — ref: {sc.reference}" if sc else ""
    title = (sc.display_name or sc.name) if sc else scenario
    _ptag = {"double": "FP64", "single": "FP32"}
    precs = sorted({_ptag.get(CONFIGS[c].precision, CONFIGS[c].precision)
                    for c, e, _, _, _ in entries if e.get("status") == "supported" and c in CONFIGS})
    plabel = "/".join(precs) if precs else ""
    dev_label = {"cpu": " · CPU", "gpu": " · GPU"}.get(device, "")
    ax.set_title(f"{title}{dev_label}  (n = {npart:,} particles, {plabel}){ref}", fontsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # footers: code versions (always) + asterisk note (untuned) + fast-math-overlay note
    notes = []
    if any_untuned:
        notes.append(sc.untuned_note if sc and sc.untuned_note
                     else "*  lacks a tuned model for this problem; runs a costlier one")
    if any_fm:
        notes.append("lighter bar = fast-math (relaxed FP), drawn behind its IEEE bar")
    bottom = 0.08 + 0.045 * len(notes)
    fig.tight_layout(rect=[0, bottom, 1, 1])
    cv = (data.get("metadata") or {}).get("code_version") or {}
    seen, present = set(), []
    for cc in codes:
        if cc not in seen:
            seen.add(cc)
            present.append(f"{cc} {cv.get(cc, '?')}")
    if present:
        fig.text(0.01, 0.012, "versions:  " + "   ·   ".join(present),
                 fontsize=5.5, color="dimgray")
    for j, note in enumerate(notes):
        fig.text(0.01, 0.012 + 0.045 * (j + 1), note, fontsize=6.5, color="dimgray", style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario}_{device}" if device else scenario  # combined = no suffix; _cpu / _gpu split
    out_path = out_dir / f"{stem}.svg"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
    return out_path


def plot_scenario_gpu(data: dict, scenario: str, npart=None,
                      out_dir: Path = PLOTS_DIR / "gpu") -> Path | None:
    """GPU FP32 cross-code comparison: one bar per code = its best config toward GPU-FP32.
    Codes without a GPU-FP32 build show their next-best (GPU-FP64, or CPU) with an asterisk
    and a per-code caveat footnote -- analogous to the CPU plots' untuned-model asterisk."""
    npart = _select_npart(data, scenario, npart)
    if npart is None:
        return None
    entries = _gpu_entries_for(data, scenario, npart)
    if not entries or all(r > 0 for *_, r in entries):
        return None  # nothing actually ran on the GPU for this scenario -> no GPU plot

    sc = SCENARIOS.get(scenario)
    untuned = sc.untuned_codes if sc else {}

    heights = [(e.get("push_per_sec") or 0.0) if e.get("status") == "supported" else 0.0
               for _, e, _, _ in entries]
    ymax = _compute_ylim(heights)

    # caveats grouped BY REASON (compact footnote even with all 7 codes): device/precision
    # fallback (this plot) + costlier untuned model (existing). A code may appear in two groups.
    by_reason: dict[str, list[str]] = {}
    caveats: set[str] = set()  # codes that get an asterisk
    for _, _, code, rung in entries:
        reasons = ([_GPU_CAVEAT[rung]] if rung in _GPU_CAVEAT else []) + \
                  (["exact stand-in model"] if code in untuned else [])
        for r in reasons:
            by_reason.setdefault(r, []).append(code)
        if reasons:
            caveats.add(code)

    # width: keep a generous minimum so the long "— GPU FP32 (...) — ref:" title and the
    # per-code caveat footnote fit even for the narrow (few-bar) scenarios.
    fig, ax = plt.subplots(figsize=(max(6.8, 1.05 * len(entries) + 1.9), 3.4))
    for i, (cfg_name, entry, code, rung) in enumerate(entries):
        color = CODE_COLORS.get(code, "gray")
        physics = entry.get("physics")
        dashed = physics in DASHED_PHYSICS
        h = heights[i]
        bar = ax.bar(i, h, color=color, edgecolor="black", width=0.72,
                     linewidth=1.3, alpha=0.55 if dashed else 0.95)[0]
        if dashed:
            bar.set_linestyle((0, (4, 2)))
            bar.set_hatch("//")
        star = " *" if code in caveats else ""
        # for fallbacks, name the actual device/precision shown ("GPU FP64"/"CPU FP32"/"CPU FP64");
        # the headline GPU-FP32 bars need no sub-label (the title already says GPU FP32)
        sub = f"\n{_GPU_CAVEAT[rung].split(' (')[0]}" if rung > 0 else ""
        ax.text(i, h + ymax * 0.01, f"{h:.1e}{star}{sub}", ha="center", va="bottom", fontsize=6.5)
        marker = _physics_marker(entry)
        if marker:
            ax.text(i, min(h + ymax * 0.22, ymax * 0.92), marker, ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold")

    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([c for _, _, c, _ in entries], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("particles / second")
    ref = f" — ref: {sc.reference}" if sc else ""
    title = (sc.display_name or sc.name) if sc else scenario
    ax.set_title(f"{title}  —  GPU FP32  (n = {npart:,} particles){ref}", fontsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    bottom = 0.13 if caveats else 0.08
    fig.tight_layout(rect=[0, bottom, 1, 1])
    cv = (data.get("metadata") or {}).get("code_version") or {}
    seen, present = set(), []
    for _, _, cc, _ in entries:
        if cc not in seen:
            seen.add(cc)
            present.append(f"{cc} {cv.get(cc, '?')}")
    if present:
        fig.text(0.01, 0.012, "versions:  " + "   ·   ".join(present), fontsize=5.5, color="dimgray")
    if by_reason:
        note = "*  next-best shown:   " + ";   ".join(
            f"{', '.join(codes)} = {reason}" for reason, codes in by_reason.items())
        fig.text(0.01, 0.012 + 0.045, note, fontsize=5.5, color="dimgray", style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario}.svg"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{scenario}.pdf")
    fig.savefig(out_dir / f"{scenario}.png", dpi=150)
    plt.close(fig)
    return out_path


def plot_all(data: dict, out_dir: Path = PLOTS_DIR) -> list[Path]:
    made = []
    for scenario in SCENARIOS:
        # combined (both devices) + device-split _cpu / _gpu versions of the same chart
        for device in (None, "cpu", "gpu"):
            p = plot_scenario(data, scenario, out_dir=out_dir, device=device)
            if p:
                made.append(p)
                print(f"wrote {p}")
    return made


def plot_all_gpu(data: dict, out_dir: Path = PLOTS_DIR / "gpu") -> list[Path]:
    made = []
    for scenario in SCENARIOS:
        p = plot_scenario_gpu(data, scenario, out_dir=out_dir)
        if p:
            made.append(p)
            print(f"wrote {p}")
    return made


def main(argv=None) -> int:
    from .metadata import machine_slug

    parser = argparse.ArgumentParser(description="Plot stored benchmark results.")
    parser.add_argument("--machine", default=machine_slug(), help="machine slug to plot")
    parser.add_argument("--scenario", default="", help="single scenario (default: all)")
    parser.add_argument("--gpu", action="store_true",
                        help="GPU FP32 cross-code comparison (next-best+asterisk for no-GPU/no-FP32)")
    args = parser.parse_args(argv)

    data = results_mod.load(results_mod.results_path(args.machine))
    if not data.get("results"):
        print(f"No results found for machine '{args.machine}'.")
        return 1
    if args.scenario:
        if args.gpu:
            p = plot_scenario_gpu(data, args.scenario)
            if p:
                print(f"wrote {p}")
        else:  # combined + _cpu + _gpu, matching plot_all
            for device in (None, "cpu", "gpu"):
                p = plot_scenario(data, args.scenario, device=device)
                if p:
                    print(f"wrote {p}")
    else:
        (plot_all_gpu if args.gpu else plot_all)(data)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
