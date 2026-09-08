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
    "bmad": "black",         # was missing -> defaulted to gray, colliding with impactz's tab:gray
    "elegant": "tab:cyan",
    "helix": "tab:pink",
    "synergia": "tab:olive",
    "impactz": "tab:gray",   # ImpactX's predecessor (fairest SC comparison)
}

# Human-facing names for the x-axis of the per-code "best" summary (draft: as spelled in the
# request; canonical project casing is Bmad/SciBmad/Xsuite/PyORBIT/HELIX -- one edit here to change).
CODE_DISPLAY = {
    "impactx": "ImpactX",
    "cheetah": "Cheetah",
    "pyat": "PyAT",
    "pyorbit": "PyOrbit",
    "xsuite": "XSuite",
    "scibmad": "SciBMad",
    "bmad": "BMad",
    "elegant": "Elegant",
    "helix": "Helix",
    "synergia": "Synergia",
    "impactz": "IMPACT-Z",
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
# The value labels above bars are drawn at a STATIC 6.5pt; one text line is ~fontsize*linespacing
# points. Used to stack the fast-math value exactly one line above the (1-2 line) black value label
# via an "offset points" annotation, independent of the dynamic y-axis scale.
_VALUE_LINE_PT = 6.5 * 1.25


def _physics_marker(entry: dict) -> str:
    physics = entry.get("physics")
    if physics == "model_mismatch":
        # A lower-fidelity / approximate model vs the scenario's reference. The per-code specifics
        # (paraxial quad, geometric coords, ...) are named in the footnote legend. Deliberately do
        # NOT use entry["model"] here -- that is the SC model, meaningless (and misleading) for a
        # tracking scenario. The generic tag is deliberately soft; the FOOTNOTE names each code's
        # actual model precisely (incl. cases like Elegant's geometric-vs-canonical coords that
        # aren't literally "simpler"). Two lines so it fits over a narrow bar without overlapping.
        return "simpler\nmodel"
    if physics == "unconverged":
        return "unconv."
    if physics == "incorrect":
        return "physics ✗"
    return ""


def _marker_style(physics: str) -> tuple[str, str]:
    """(color, weight) for a bar's physics marker: soft grey (like the skipped-code placeholders)
    for the 'simpler model' / 'unconv.' caveats; keep 'physics ✗' (out of tolerance) prominent."""
    if physics == "incorrect":
        return "black", "bold"
    return "dimgray", "normal"


def _marker_legend(code_physics, sc=None) -> str:
    """Footnote explaining the physics markers actually drawn above the bars. For 'approx. model'
    bars it names EACH code's actual lower-fidelity model (from the scenario's
    ``model_mismatch_codes``) rather than a vague blanket word. Empty if nothing is marked."""
    pairs = list(code_physics)
    kinds = {ph for _, ph in pairs}
    parts = []
    if "model_mismatch" in kinds:
        mm = (getattr(sc, "model_mismatch_codes", None) or {}) if sc is not None else {}
        # entries may carry CONFIG names (per-scenario plot: 'cheetah-cpu-dp') or BASE codes (GPU
        # plot: 'cheetah'); model_mismatch_codes is keyed by base code, so resolve first -- else
        # every config name misses and falls back to a vague placeholder.
        codes = list(dict.fromkeys(
            (CONFIGS[c].code if c in CONFIGS else c) for c, ph in pairs if ph == "model_mismatch"))
        detail = ";   ".join(
            f"{c}: {mm[c].split(';')[0].strip()}" if mm.get(c)
            else f"{c}: uncurated model_mismatch (add to model_mismatch_codes)" for c in codes)
        parts.append(f"'simpler model' = a lower-fidelity model vs the reference —  {detail}")
    if "unconverged" in kinds:
        parts.append("'unconv.' = out of tolerance but its FP64 run is correct (convergence artefact)")
    if "incorrect" in kinds:
        parts.append("'physics ✗' = converged but out of tolerance")
    return "   |   ".join(parts) if parts else ""


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
# One ladder per target precision: FP32 (only ImpactX/Cheetah/SciBmad build it) and FP64 (the
# widely-supported GPU precision -- ImpactX/Synergia/Xsuite/Cheetah/HELIX/...). Each code shows one
# bar: its best config toward the headline GPU result at that precision; a code below the top rung
# (no GPU build at that precision) shows its next-best, asterisked with a per-code caveat.
_GPU_LADDERS = {
    "single": [           # headline: GPU FP32
        ("cuda", "single"),   # 0: real GPU FP32 (no caveat)
        ("cuda", "double"),   # 1: GPU, but FP64 only (no FP32 build)
        ("cpu", "single"),    # 2: no GPU; CPU FP32
        ("cpu", "double"),    # 3: no GPU; CPU FP64
    ],
    "double": [           # headline: GPU FP64
        ("cuda", "double"),   # 0: real GPU FP64 (no caveat)
        ("cpu", "double"),    # 1: no GPU; CPU FP64
        ("cuda", "single"),   # 2: GPU, but FP32 only (no FP64 build)
        ("cpu", "single"),    # 3: no GPU; CPU FP32
    ],
}
_GPU_RUNGS = {prec: {dp: i for i, dp in enumerate(ladder)} for prec, ladder in _GPU_LADDERS.items()}
_GPU_CAVEATS = {
    "single": {1: "GPU FP64 (no FP32)", 2: "CPU FP32 (no GPU)", 3: "CPU FP64 (no GPU)"},
    "double": {1: "CPU FP64 (no GPU)", 2: "GPU FP32 (no FP64)", 3: "CPU FP32 (no GPU)"},
}
_GPU_HEADLINE = {"single": "GPU FP32", "double": "GPU FP64"}


def _gpu_entries_for(data: dict, scenario: str, npart: int,
                     precision: str = "single") -> list[tuple[str, dict, str, int]]:
    """``(cfg_name, entry, code, rung)`` per code -- the best *supported* config toward the headline
    GPU result at ``precision`` (lowest ladder rung; ties within a rung broken by fastest). ``rung``
    0 means a real GPU result at that precision; >0 is the next-best fallback (see
    ``_GPU_CAVEATS[precision]``)."""
    rungs = _GPU_RUNGS[precision]
    key = results_mod.measurement_key(scenario, npart)
    out: list[tuple[str, dict, str, int]] = []
    for code in CODES:
        best_sort = None            # (rung, -push, cfg_name) -- min() picks preferred+fastest
        best = None                 # (cfg_name, entry, rung)
        for cfg_name, cfg in CONFIGS.items():
            if cfg.code != code or cfg.fast_math:
                continue  # IEEE bars only (fast-math variants aren't shown in the GPU comparison)
            rung = rungs.get((cfg.device, cfg.precision))
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


def _compute_ylim(heights: list[float], headroom: float = YHEADROOM) -> float:
    """Y-axis top: fully show the fastest (tallest) bar, with label headroom (pass a larger
    ``headroom`` when a physics marker sits above a bar, so its 2-line tag clears the value label)."""
    pos = [h for h in heights if h and h > 0]
    return max(pos) * headroom if pos else 1.0


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

    # extra headroom when any bar carries a physics marker, so the 2-line 'simpler model' tag sits
    # clearly above the value label rather than overlapping it.
    any_marker = any(_physics_marker(e) for _, e, _, _, _ in entries)
    ymax = _compute_ylim(heights + fm_heights, 1.6 if any_marker else YHEADROOM)

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
    fm_labels: list = []  # (xi, h, fmh, color, n_lines); drawn after tight_layout (final geometry)
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
            # Fast-math THROUGHPUT, a lighter tint of the code's colour, stacked EXACTLY one line
            # above the black IEEE value label. The y-axis is dynamic but the font is static, so
            # the offset MUST be in POINTS, not an ymax fraction: anchor to the black label's data
            # position (h + 0.01*ymax) and push up by the black block's height = n_lines text lines
            # (6.5pt at linespacing ~1.2 -> ~7.8pt/line). Shown only for a real >=5% fast-math win.
            r = fmh / h if h else 0.0
            if r >= 1.05 and not fm_bad:
                # Defer placement to after tight_layout: the value sits at the HIGHER of one line
                # above the black value label, or on top of the light fm bar (a big speedup, e.g.
                # fodo_exact CPU) -- and comparing those two needs the final data<->points scale.
                n_lines = 2 if entry.get("cores") else 1
                fm_labels.append((xi, h, fmh, color, n_lines))
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
            # status marker in the empty space above the value label -- soft grey + small (like the
            # skipped-code placeholders) so it doesn't shout or overlap neighbours; lower cap leaves
            # room for the 2-line "simpler\nmodel" tag.
            mcolor, mweight = _marker_style(physics)
            ax.text(xi, min(h + ymax * 0.24, ymax * 0.88), marker, ha="center", va="bottom",
                    fontsize=7, color=mcolor, fontweight=mweight, linespacing=0.9)

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
        notes.append("lighter bar behind + lighter value above = fast-math (relaxed FP), "
                     "shown when >=5% faster than the IEEE bar")
    ml = _marker_legend(((c, e.get("physics")) for c, e, _, _, _ in entries), sc)
    if ml:
        notes.append(ml)
    bottom = 0.08 + 0.045 * len(notes)
    fig.tight_layout(rect=[0, bottom, 1, 1])

    # Fast-math value labels: placed now that the axes geometry is final. Each sits at the HIGHER
    # of (a) one line above the black value label, or (b) on top of the light fm bar (a big speedup
    # like fodo_exact CPU). "One line" is static font -> convert points to data via the final axes
    # height (ymax spans axes_height_pt points), then max() with the fm bar top (data).
    if fm_labels:
        axes_h_pt = ax.get_position().height * fig.get_figheight() * 72.0
        data_per_pt = ymax / axes_h_pt if axes_h_pt else 0.0
        for xi, h, fmh, color, n_lines in fm_labels:
            one_line_above = h + ymax * 0.01 + n_lines * _VALUE_LINE_PT * data_per_pt
            y = max(one_line_above, fmh)
            ax.annotate(f"{fmh:.1e}", xy=(xi, y), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=color, alpha=0.6)

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
                      out_dir: Path = PLOTS_DIR / "gpu", precision: str = "single") -> Path | None:
    """GPU cross-code comparison at ``precision`` (FP32 or FP64): one bar per code = its best config
    toward the headline GPU result. Codes without a GPU build at that precision show their next-best
    (other-precision GPU, or CPU) with an asterisk + a per-code caveat footnote -- analogous to the
    CPU plots' untuned-model asterisk. FP32 -> ``<scenario>.svg``; FP64 -> ``<scenario>_fp64.svg``."""
    npart = _select_npart(data, scenario, npart)
    if npart is None:
        return None
    caveat = _GPU_CAVEATS[precision]
    entries = _gpu_entries_for(data, scenario, npart, precision)
    if not entries or all(r > 0 for *_, r in entries):
        return None  # nothing actually ran on the GPU for this scenario -> no GPU plot

    sc = SCENARIOS.get(scenario)
    untuned = sc.untuned_codes if sc else {}

    heights = [(e.get("push_per_sec") or 0.0) if e.get("status") == "supported" else 0.0
               for _, e, _, _ in entries]
    any_marker = any(_physics_marker(e) for _, e, _, _ in entries)
    ymax = _compute_ylim(heights, 1.6 if any_marker else YHEADROOM)

    # caveats grouped BY REASON (compact footnote even with all 7 codes): device/precision
    # fallback (this plot) + costlier untuned model (existing). A code may appear in two groups.
    by_reason: dict[str, list[str]] = {}
    caveats: set[str] = set()  # codes that get an asterisk
    for _, _, code, rung in entries:
        reasons = ([caveat[rung]] if rung in caveat else []) + \
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
        sub = f"\n{caveat[rung].split(' (')[0]}" if rung > 0 else ""
        ax.text(i, h + ymax * 0.01, f"{h:.1e}{star}{sub}", ha="center", va="bottom", fontsize=6.5)
        marker = _physics_marker(entry)
        if marker:
            mcolor, mweight = _marker_style(physics)
            ax.text(i, min(h + ymax * 0.24, ymax * 0.88), marker, ha="center", va="bottom",
                    fontsize=7, color=mcolor, fontweight=mweight, linespacing=0.9)

    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([c for _, _, c, _ in entries], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("particles / second")
    ref = f" — ref: {sc.reference}" if sc else ""
    title = (sc.display_name or sc.name) if sc else scenario
    ax.set_title(f"{title}  —  {_GPU_HEADLINE[precision]}  (n = {npart:,} particles){ref}", fontsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ml = _marker_legend(((c, e.get("physics")) for _, e, c, _ in entries), sc)
    bottom = 0.08 + 0.05 * (bool(by_reason) + bool(ml))
    fig.tight_layout(rect=[0, bottom, 1, 1])
    cv = (data.get("metadata") or {}).get("code_version") or {}
    seen, present = set(), []
    for _, _, cc, _ in entries:
        if cc not in seen:
            seen.add(cc)
            present.append(f"{cc} {cv.get(cc, '?')}")
    if present:
        fig.text(0.01, 0.012, "versions:  " + "   ·   ".join(present), fontsize=5.5, color="dimgray")
    y = 0.012 + 0.05
    if by_reason:
        note = "*  next-best shown:   " + ";   ".join(
            f"{', '.join(codes)} = {reason}" for reason, codes in by_reason.items())
        fig.text(0.01, y, note, fontsize=5.5, color="dimgray", style="italic")
        y += 0.05
    if ml:
        fig.text(0.01, y, ml, fontsize=5.5, color="dimgray", style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = scenario if precision == "single" else f"{scenario}_fp64"
    out_path = out_dir / f"{stem}.svg"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
    return out_path


def _fmt_npart(n) -> str:
    """Particle count in scientific notation, 2 significant digits: 1.0e6, 1.6e7, 1.0e9."""
    if not n:
        return ""
    return f"{n:.1e}".replace("e+0", "e").replace("e+", "e")


def _best_measurement(data: dict, scenario: str, code: str, device: str):
    """The fastest VALID config for ``(code, device)`` in ``scenario`` at that code's LARGEST measured
    particle count (>= ``PUBLISHED_MIN_NPART``), over all its configs (any precision, IEEE or
    fast-math). Returns ``(cfg_name, entry)`` or ``None``.

    Why the *largest* N and not the numerically highest throughput: it keeps the comparison fair. A
    small-N point can sit entirely in cache (L1/L2/L3) and post an unrepresentative rate, so we pin
    each code to the top of its own sweep -- CPU codes end up ~1M (streaming from RAM), while GPU
    codes sit at the large N (>=10M, up to 1e9) where the device is actually saturated and used
    efficiently. N therefore varies per bar (shown as the per-bar ``N=`` label). Ties at the same N
    (e.g. a DP vs an SP/fast-math build) are broken by throughput. 'Valid' = ``supported`` with a
    ``push_per_sec`` and not physics-wrong (excludes ``incorrect``)."""
    best = None  # (npart, push, cfg_name, entry)
    for cfg_name, cfg in CONFIGS.items():
        if cfg.code != code or cfg.device != device:
            continue
        for entry in data.get("results", {}).get(cfg_name, {}).values():
            if entry.get("scenario") != scenario or entry.get("status") != "supported":
                continue
            if entry.get("physics") == "incorrect":
                continue
            n, pps = entry.get("npart") or 0, entry.get("push_per_sec")
            if not pps or n < PUBLISHED_MIN_NPART:
                continue
            if best is None or (n, pps) > (best[0], best[1]):
                best = (n, pps, cfg_name, entry)
    return (best[2], best[3]) if best else None


def _machine_label(data: dict) -> str:
    """Human machine name from the results metadata (``perlmutter`` -> ``Perlmutter``)."""
    slug = ((data.get("metadata") or {}).get("host") or {}).get("machine_slug") or ""
    return slug.replace("-", " ").title()


def _cpu_core_budget(data: dict) -> int:
    """The benchmark's CPU core budget = the largest ranks x threads layout actually used across the
    run (the auto-tune explores up to the budget, so some cell hits it). Robust to the ``cores`` text
    format (e.g. ``8r x 8t``) by multiplying the integers found. 0 if none recorded."""
    import math
    best = 0
    for cells in (data.get("results") or {}).values():
        for e in (cells or {}).values():
            nums = re.findall(r"\d+", str(e.get("cores") or ""))
            if nums:
                best = max(best, math.prod(int(x) for x in nums))
    return best


def _hardware_note(data: dict, bars: list) -> str:
    """The benchmark's hardware constraint for the grey footer: CPU model + core budget, and the GPU
    model (only if a GPU bar is shown). Mirrors what the run was actually allowed to use."""
    host = (data.get("metadata") or {}).get("host") or {}
    parts = []
    model = host.get("cpu_model") or (host.get("cpu") or {}).get("Model name")
    if model and any(b[1] == "cpu" for b in bars):
        model = re.sub(r"\s*\d+-Core Processor$", "", model)          # drop redundant "64-Core Processor"
        model = re.sub(r"\s*(Processor|CPU)$", "", model).strip()
        n = _cpu_core_budget(data)
        parts.append(f"CPU: {model}" + (f" ({n} cores)" if n else ""))
    gpus = (data.get("metadata") or {}).get("gpu") or []
    if gpus and any(b[1] == "cuda" for b in bars):
        names = list(dict.fromkeys(g.get("name") for g in gpus if g.get("name")))  # dedupe identical cards
        if names:
            parts.append("GPU: " + " / ".join(names))
    return "hardware:  " + "     ".join(parts) if parts else ""


def plot_scenario_best(data: dict, scenario: str, out_dir: Path = PLOTS_DIR,
                       logy: bool = True) -> Path | None:
    """Per-code 'best' summary: ONE bar per code = its fastest measured config on CPU, then (after a
    gap) one bar per code = its fastest on GPU. 'Best' is the peak throughput across the whole sweep
    and all the code's configs on that device (see :func:`_best_measurement`), so a bar may be an SP
    or fast-math build. Bars keep the code colour; the x-label is ``<Code> (CPU|GPU)`` and the value
    label is throughput (+ a tiny grey ``@N`` = the particle count where it peaked). model_mismatch /
    unconverged bars keep the dashed convention; untuned codes keep the asterisk. ``logy`` (default)
    puts the y-axis on a log scale so the CPU group is readable next to the far-taller GPU bars (the
    CPU-vs-GPU span is several decades); ``logy=False`` gives the linear-axis companion. Written next
    to the per-scenario plots as ``<scenario>_best`` (log) or ``<scenario>_best_liny`` (linear); plot_all
    emits both. All value/marker labels use point offsets so placement is identical on linear or log."""
    sc = SCENARIOS.get(scenario)
    untuned = sc.untuned_codes if sc else {}

    # GPU section first, then CPU; within each section fastest -> slowest.
    bars = []  # (code, device, dev_label, cfg_name, entry)
    for device, dev_label in (("cuda", "GPU"), ("cpu", "CPU")):
        section = [(code, device, dev_label, *be)
                   for code in CODES
                   if (be := _best_measurement(data, scenario, code, device))]
        section.sort(key=lambda b: b[4].get("push_per_sec") or 0.0, reverse=True)
        bars.extend(section)
    if not bars:
        return None

    heights = [e.get("push_per_sec") or 0.0 for *_, e in bars]
    pos = [h for h in heights if h > 0]
    hi = max(pos) if pos else 1.0
    any_marker = any(_physics_marker(e) for *_, e in bars)
    if logy:
        lo = min(pos) if pos else 1.0
        ybot = lo / 3.0                       # a little air under the shortest bar
        ytop = hi * (300 if any_marker else 60)  # headroom for the stacked value / N= / marker labels
    else:
        ybot = 0.0
        ytop = _compute_ylim(heights, 1.6 if any_marker else YHEADROOM)

    # x: GPU block, a gap, then CPU block (gap inserted where the device changes)
    GROUP_GAP = 0.8
    xs, xpos, prev = [], 0.0, None
    for _, device, *_ in bars:
        if prev is not None and device != prev:
            xpos += GROUP_GAP
        xs.append(xpos)
        xpos += 1.0
        prev = device
    span = (xs[-1] + 1.0) if xs else 1.0

    fig, ax = plt.subplots(figsize=(max(6.0, 0.60 * span), 3.4))
    if logy:
        ax.set_yscale("log")
    labels, codes_present, any_untuned = [], [], False
    for i, (code, device, dev_label, cfg_name, entry) in enumerate(bars):
        xi = xs[i]
        color = CODE_COLORS.get(code, "gray")
        h = heights[i]
        physics = entry.get("physics")
        dashed = physics in DASHED_PHYSICS
        # default bottom=0 -> on a log axis matplotlib clips the bar base to the axis lower limit
        # (ybot), so the bar spans ybot..h with its top exactly at h (where the labels anchor).
        bar = ax.bar(xi, h, color=color, edgecolor="black", width=0.8,
                     linewidth=1.3, alpha=0.55 if dashed else 0.95, zorder=2)[0]
        if dashed:
            bar.set_linestyle((0, (4, 2)))
            bar.set_hatch("//")
        star = " *" if code in untuned else ""
        any_untuned = any_untuned or bool(star)
        # value label, then the grey N= (each code's particle count differs), then an optional
        # physics marker -- all stacked above the bar top with POINT offsets, so the line spacing is
        # the same on a linear or log axis.
        ax.annotate(f"{h:.1e}{star}", xy=(xi, h), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5)
        ax.annotate(f"N={_fmt_npart(entry.get('npart'))}", xy=(xi, h),
                    xytext=(0, 2 + _VALUE_LINE_PT), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.5, color="dimgray")
        marker = _physics_marker(entry)
        if marker:
            mcolor, mweight = _marker_style(physics)
            ax.annotate(marker, xy=(xi, h), xytext=(0, 2 + 2 * _VALUE_LINE_PT), fontsize=7,
                        textcoords="offset points", ha="center", va="bottom",
                        color=mcolor, fontweight=mweight, linespacing=0.9)
        labels.append(f"{CODE_DISPLAY.get(code, code)} ({dev_label})")
        if code not in codes_present:
            codes_present.append(code)

    # faint divider in the gap between the two contiguous blocks + group headers (axes-fraction y,
    # so scale-independent). Order-agnostic: the split is where the device label changes.
    split = next((i for i in range(1, len(bars)) if bars[i][1] != bars[i - 1][1]), None)
    if split is not None:
        ax.axvline((xs[split - 1] + xs[split]) / 2, color="lightgray", lw=0.8, ls=":", zorder=0)
    gpu_xs = [xs[i] for i, b in enumerate(bars) if b[1] == "cuda"]
    cpu_xs = [xs[i] for i, b in enumerate(bars) if b[1] == "cpu"]
    for block, name in ((gpu_xs, "GPU"), (cpu_xs, "CPU")):
        if block:
            ax.text(sum(block) / len(block), 0.97, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="dimgray", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(ybot if logy else 0, ytop)
    ax.set_ylabel("particles / second")
    ref = f" — ref: {sc.reference}" if sc else ""
    title = (sc.display_name or sc.name) if sc else scenario
    machine = _machine_label(data)
    prefix = f"{machine} · " if machine else ""
    ax.set_title(f"{prefix}{title} — best per code, GPU vs. CPU{ref}", fontsize=9)
    if not logy:  # sci ScalarFormatter is invalid on a log axis (LogFormatter already reads 10^n)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    notes = ["each bar = the fastest measured config for that code on that device (any precision / "
             "fast-math); N = the particle count that bar was measured at (top of each device's sweep)"]
    if any_untuned:
        notes.append(sc.untuned_note if sc and sc.untuned_note
                     else "*  lacks a tuned model for this problem; runs a costlier one")
    ml = _marker_legend(((c, e.get("physics")) for c, _, _, _, e in bars), sc)
    if ml:
        notes.append(ml)
    hw = _hardware_note(data, bars)
    # footer rows (bottom-up): versions, [hardware], then the notes -- reserve space for all of them
    rows = 1 + bool(hw) + len(notes)
    bottom = 0.055 + 0.045 * rows
    fig.tight_layout(rect=[0, bottom, 1, 1])

    cv = (data.get("metadata") or {}).get("code_version") or {}
    present = [f"{cc} {cv.get(cc, '?')}" for cc in codes_present]
    y = 0.012
    if present:
        fig.text(0.01, y, "versions:  " + "   ·   ".join(present), fontsize=5.5, color="dimgray")
        y += 0.045
    if hw:
        fig.text(0.01, y, hw, fontsize=6.0, color="dimgray")
        y += 0.045
    for note in notes:
        fig.text(0.01, y, note, fontsize=6.0, color="dimgray", style="italic")
        y += 0.045

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario}_best" if logy else f"{scenario}_best_liny"  # log (default) vs linear y
    out_path = out_dir / f"{stem}.svg"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
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
        # per-code best-CPU|best-GPU summary, in both log (<scenario>_best) and linear
        # (<scenario>_best_liny) y-axis variants
        for logy in (True, False):
            pb = plot_scenario_best(data, scenario, out_dir=out_dir, logy=logy)
            if pb:
                made.append(pb)
                print(f"wrote {pb}")
    return made


def plot_all_gpu(data: dict, out_dir: Path = PLOTS_DIR / "gpu") -> list[Path]:
    made = []
    for scenario in SCENARIOS:
        for precision in ("single", "double"):   # FP32 (<scenario>) + FP64 (<scenario>_fp64)
            p = plot_scenario_gpu(data, scenario, out_dir=out_dir, precision=precision)
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
    parser.add_argument("--out", default="",
                        help="output directory (default: plots/). Pass e.g. plots/<machine> to keep "
                             "per-machine plots separate instead of overwriting the shared plots/ dir")
    args = parser.parse_args(argv)

    data = results_mod.load(results_mod.results_path(args.machine))
    if not data.get("results"):
        print(f"No results found for machine '{args.machine}'.")
        return 1
    base = Path(args.out) if args.out else PLOTS_DIR
    if args.scenario:
        if args.gpu:
            for precision in ("single", "double"):
                p = plot_scenario_gpu(data, args.scenario, out_dir=base / "gpu", precision=precision)
                if p:
                    print(f"wrote {p}")
        else:  # combined + _cpu + _gpu, matching plot_all
            for device in (None, "cpu", "gpu"):
                p = plot_scenario(data, args.scenario, device=device, out_dir=base)
                if p:
                    print(f"wrote {p}")
    elif args.gpu:
        plot_all_gpu(data, out_dir=base / "gpu")
    else:
        plot_all(data, out_dir=base)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
