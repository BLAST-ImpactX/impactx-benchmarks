"""Render per-(code, scenario) run scripts from Jinja templates.

Templates live at ``codes/<code>/<scenario>.<ext>.jinja`` where ``<ext>`` is ``py``
for Python codes and ``jl`` for Julia codes. The rendered script must print exactly
two machine-readable lines on success::

    Track: <int>ns
    Validate: {"sigma_x": ..., "sigma_y": ..., ...}

The scenario's physics parameters are baked into the template context so the same
single source of truth feeds both Python and Julia codes without import gymnastics.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .registry import CODES

REPO_ROOT = Path(__file__).resolve().parent.parent


def template_ext(code: str) -> str:
    lang = CODES[code].language
    # For file-driven codes the "primary" per-scenario template is the lattice: bmad -> .bmad,
    # elegant -> .lte (each pairs with a shared command/namelist template rendered in render_run_script).
    return {"julia": "jl", "fortran": "bmad", "elegant": "lte"}.get(lang, "py")


def _env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_run_script(code: str, scenario: str, context: dict, out_dir: Path) -> Path:
    """Render the run script(s) for ``(code, scenario)`` into ``out_dir``; return the path
    the launcher runs. Python/Julia codes render one file; Bmad (Fortran) renders a ``.bmad``
    lattice plus a ``.in`` namelist (the launch target referencing the lattice)."""
    ext = template_ext(code)
    template_dir = REPO_ROOT / "codes" / code
    env = _env(template_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if CODES[code].language == "fortran":
        # 1) lattice file from <scenario>.bmad.jinja
        lat_path = out_dir / f"{code}__{scenario}.bmad"
        lat_path.write_text(env.get_template(f"{scenario}.bmad.jinja").render(**context))
        # 2) beam_init namelist (shared template), pointing at the lattice; spin per scenario
        nml_ctx = {**context, "lat_filename": str(lat_path),
                   "spin": "T" if scenario == "htu_spin" else "F"}
        in_path = out_dir / f"{code}__{scenario}.in"
        in_path.write_text(env.get_template("beam.in.jinja").render(**nml_ctx))
        return in_path  # the driver reads the namelist

    if CODES[code].language == "elegant":
        # 1) lattice from <scenario>.lte.jinja
        lat_path = out_dir / f"{code}__{scenario}.lte"
        lat_path.write_text(env.get_template(f"{scenario}.lte.jinja").render(**context))
        # 2) command file from the shared run.ele.jinja (references the lattice by basename --
        #    the driver runs elegant with out_dir as the working directory); spin per scenario.
        ele_ctx = {**context, "lat_filename": lat_path.name, "spin": scenario == "htu_spin"}
        ele_path = out_dir / f"{code}__{scenario}.ele"
        ele_path.write_text(env.get_template("run.ele.jinja").render(**ele_ctx))
        return ele_path  # the driver runs `elegant <this>.ele`

    template = env.get_template(f"{scenario}.{ext}.jinja")
    out_path = out_dir / f"{code}__{scenario}.{ext}"
    out_path.write_text(template.render(**context))
    return out_path


def template_exists(code: str, scenario: str) -> bool:
    ext = template_ext(code)
    return (REPO_ROOT / "codes" / code / f"{scenario}.{ext}.jinja").is_file()
