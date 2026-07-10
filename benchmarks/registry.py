"""Single source of truth: codes, scenarios, configs and the capability matrix.

Adding a new code, scenario or config is meant to be a small, local edit here plus
a matching run template under ``codes/<code>/<scenario>.<ext>.jinja``.

Capability model
----------------
Each :class:`Code` advertises a set of *capability tags*. Each :class:`Scenario`
declares the tags it requires. A :class:`Config` (a concrete build/runtime variant
of a code) is *supported* for a scenario iff the code provides all required tags,
the requested precision is available, and no explicit special-case rule excludes it.

Space-charge tags are layered so a coarse requirement is satisfied by any model:

* ``space_charge``       -- the scenario needs *some* self-consistent space charge
* ``space_charge_3d``    -- full 3D PIC / FFT-Poisson
* ``space_charge_2p5d``  -- 2.5D PIC (transverse Poisson, longitudinal slices)

A code that provides ``space_charge_3d`` also lists ``space_charge`` explicitly.
The *model* a config actually uses is recorded on the config so the validation
layer can flag a **model mismatch** (e.g. Xsuite's 2.5D model in a 3D scenario).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Capability tags
# --------------------------------------------------------------------------- #
TRACKING = "tracking"
SPACE_CHARGE = "space_charge"
SPACE_CHARGE_3D = "space_charge_3d"
SPACE_CHARGE_2P5D = "space_charge_2p5d"
SPIN = "spin"  # Thomas-BMT spin tracking

SINGLE = "single"
DOUBLE = "double"


# --------------------------------------------------------------------------- #
# Codes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Code:
    """A simulation code and how the harness drives it."""

    name: str
    repo: str
    pixi_env: str
    install: str  # "pip" | "source" | "julia"
    parallelism: str  # thread mechanism: "omp" | "torch" | "julia-threads" | "mpi"
    launcher: str  # "python" | "mpirun" | "julia"
    capabilities: frozenset
    precisions: frozenset
    language: str = "python"  # "python" | "julia"
    #: parallelism the code can use. A code with both can run all-MPI, all-threads, or a
    #: mix; the harness picks the fastest layout with ranks * threads <= ncores.
    mpi_capable: bool = False
    thread_capable: bool = True

    def core_configs(self, ncores: int) -> list:
        """(ranks, threads) layouts to try, each using ranks*threads <= ncores cores.

        The auto-tune sweeps 1, 2 and the powers of two up to ``ncores`` (plus ``ncores`` itself),
        so a big single socket (Perlmutter: 64 cores) actually explores NUMA-friendly splits like
        16 threads (one NUMA domain), not just {1, 2, 64} -- "the best way a code provides". On the
        4-core local/CI budget this collapses to the previous {1,2,4} behaviour.
        """
        n = max(1, int(ncores))
        pts = {1, 2, n}
        p = 4
        while p < n:
            pts.add(p)
            p *= 2
        pts = sorted(x for x in pts if 1 <= x <= n)
        if self.mpi_capable and self.thread_capable:  # hybrid: ranks × threads == n, sweep the split
            return sorted({(r, n // r) for r in pts if n % r == 0} | {(n, 1), (1, n)})
        if self.mpi_capable:  # MPI only: vary rank count (1 thread each)
            return [(r, 1) for r in pts]
        return [(1, t) for t in pts]  # threads only: vary thread count


CODES: dict[str, Code] = {
    "impactx": Code(
        name="impactx",
        repo="https://github.com/BLAST-ImpactX/impactx",
        pixi_env="impactx",
        install="source",
        parallelism="omp",
        launcher="python",
        capabilities=frozenset(
            {TRACKING, SPACE_CHARGE, SPACE_CHARGE_3D, SPACE_CHARGE_2P5D, SPIN}
        ),
        precisions=frozenset({SINGLE, DOUBLE}),
        # ImpactX is built with MPI + OpenMP and the templates are MPI-safe, so the
        # harness sweeps ranks, threads, and mixes (e.g. 4x1, 2x2, 1x4) and keeps the
        # fastest layout within the core budget.
        mpi_capable=True,
        thread_capable=True,
    ),
    "cheetah": Code(
        name="cheetah",
        repo="https://github.com/desy-ml/cheetah",
        pixi_env="cheetah",
        install="pip",
        parallelism="torch",
        launcher="python",
        capabilities=frozenset({TRACKING, SPACE_CHARGE, SPACE_CHARGE_3D}),
        precisions=frozenset({SINGLE, DOUBLE}),
    ),
    "pyat": Code(
        name="pyat",
        repo="https://github.com/atcollab/at",
        pixi_env="pyat",
        install="source",
        parallelism="omp",
        launcher="python",
        capabilities=frozenset({TRACKING}),  # wakefields only; no self-consistent SC
        precisions=frozenset({DOUBLE}),
    ),
    "pyorbit": Code(
        name="pyorbit",
        repo="https://github.com/PyORBIT-Collaboration/PyORBIT3",
        pixi_env="pyorbit",
        install="source",
        parallelism="mpi",
        launcher="mpirun",
        capabilities=frozenset(
            {TRACKING, SPACE_CHARGE, SPACE_CHARGE_3D, SPACE_CHARGE_2P5D}
        ),
        precisions=frozenset({DOUBLE}),
        mpi_capable=True,    # MPI only (no OpenMP) -> vary rank count
        thread_capable=False,
    ),
    "xsuite": Code(
        name="xsuite",
        repo="https://github.com/xsuite/xsuite",
        pixi_env="xsuite",
        install="pip",
        parallelism="omp",
        launcher="python",
        # xfields SpaceCharge3D provides both FFTSolver3D (genuine 3D IGF) and
        # FFTSolver2p5D (2.5D) PIC solvers -> Xsuite does both 3D and 2.5D space charge.
        # Particles carry spin_x/y/z -> spin tracking capable.
        capabilities=frozenset(
            {TRACKING, SPACE_CHARGE, SPACE_CHARGE_3D, SPACE_CHARGE_2P5D, SPIN}
        ),
        precisions=frozenset({DOUBLE}),
    ),
    "scibmad": Code(
        name="scibmad",
        repo="https://github.com/bmad-sim/SciBmad.jl",
        pixi_env="scibmad",
        install="julia",
        parallelism="julia-threads",
        launcher="julia",
        # BeamTracking.jl has a spin kernel (spin.jl) -> spin tracking capable.
        capabilities=frozenset({TRACKING, SPIN}),  # space charge planned, not yet available
        precisions=frozenset({SINGLE, DOUBLE}),
        language="julia",
    ),
    "bmad": Code(
        name="bmad",
        repo="https://github.com/bmad-sim/bmad-ecosystem",
        pixi_env="bmad",
        install="source",
        parallelism="omp",
        # driven by our standalone Fortran driver linked to libbmad (NO Tao/pytao)
        launcher="bmad",
        # bmad_standard tracking is the EXACT map (Cheetah's drift_kick_drift & SciBmad are
        # ports of it) -> exact-native; full Thomas-BMT spin tracking. (Space charge later.)
        capabilities=frozenset({TRACKING, SPIN}),
        precisions=frozenset({DOUBLE}),
        language="fortran",
        mpi_capable=False,
        thread_capable=True,
    ),
}


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
# Default particle sweeps (override per run via the CLI --nparts). CPU tops out at 1M; the GPU
# sweep goes higher (single A100-80GB) to saturate the device and amortize kernel-launch overhead.
_NPARTS_SMALL = (1_000, 10_000, 100_000, 1_000_000)
_NPARTS_SC = (1_000, 10_000, 100_000, 1_000_000)
_NPARTS_GPU = (100_000, 1_000_000, 4_000_000, 16_000_000)


@dataclass(frozen=True)
class Scenario:
    """A physics problem to benchmark across codes."""

    name: str
    required_capabilities: frozenset
    nparts: tuple
    #: observables emitted by the run templates and compared during validation
    observables: tuple
    #: relative tolerance per observable (fallback ``"default"`` key allowed)
    tolerances: dict
    #: "analytic" or the name of a reference code whose output defines truth
    reference: str
    #: GPU-specific particle sweep (a single A100 handles more particles than the CPU core budget);
    #: used for device="cuda" runs (see ``sweep()``), falls back to ``nparts`` if empty.
    nparts_gpu: tuple = _NPARTS_GPU
    #: human-facing title for plots/headings (falls back to ``name`` if unset)
    display_name: Optional[str] = None
    #: physics model the scenario *intends* (configs using another are flagged)
    intended_model: Optional[str] = None
    #: codes that cannot express this scenario despite owning the base capability
    #: (e.g. missing element types); reason kept for the plot label
    unsupported_codes: dict = field(default_factory=dict)
    #: codes that ARE shown (real, validated result) but run a non-tuned model for this
    #: problem (a costlier one where a fast one suffices, OR a cheaper/cruder one than the
    #: reference); marked with an asterisk in the plot. reason kept for the label.
    untuned_codes: dict = field(default_factory=dict)
    #: plot footnote explaining the asterisk (falls back to the costlier-model default)
    untuned_note: Optional[str] = None
    #: extension of the run template (python "py" or Julia "jl")
    template_for_language: dict = field(
        default_factory=lambda: {"python": "py", "julia": "jl"}
    )

    def sweep(self, device: str) -> tuple:
        """Particle-count sweep for ``device``: the larger ``nparts_gpu`` on GPU, else ``nparts``."""
        return self.nparts_gpu if (device == "cuda" and self.nparts_gpu) else self.nparts


# Observables that drive validation. Near-zero means (mean_x/mean_y) may also be
# emitted by templates for the record, but are not compared (relative error of ~0
# is dominated by sampling noise).
_OBS_TRACKING = ("sigma_x", "sigma_y", "emit_x", "emit_y")
_OBS_SC = ("sigma_x", "sigma_y", "sigma_t", "emit_x", "emit_y")
# spin depolarization: RMS spread of the spin components (beam starts fully aligned +z).
# Validate only sx/sy -- the in-plane tilt carries the depolarization signal; sigma_sz is a
# 2nd-order tiny quantity (spins barely leave +z) and is noise-dominated. Templates still
# emit sigma_sz for the record.
_OBS_SPIN = ("sigma_sx", "sigma_sy")

SCENARIOS: dict[str, Scenario] = {
    "fodo_chromatic": Scenario(
        name="fodo_chromatic",
        display_name="chromatic FODO",
        required_capabilities=frozenset({TRACKING}),
        nparts=_NPARTS_SMALL,
        observables=_OBS_TRACKING,
        # chromatic models from different integrators agree to ~1e-3; allow a bit more
        tolerances={"default": 5e-3},
        reference="impactx",  # ImpactX ChrQuad+ChrDrift defines truth
        # Consistent chromatic-paraxial model: ChrQuad + ChrDrift (x' = px/(1+delta)).
        # Cheetah and SciBmad lack a tuned chromatic model and run the costlier *exact*
        # map as a stand-in -> shown with an asterisk (physically fine, just not tuned).
        untuned_codes={
            "cheetah": "no chromatic quad; runs the exact drift_kick_drift map",
            "scibmad": "only exact models (MatrixKick symplectic quad + exact drift); no chromatic-paraxial",
            "bmad": "only exact bmad_standard tracking; no chromatic-paraxial model",
        },
    ),
    "fodo_exact": Scenario(
        name="fodo_exact",
        display_name="exact FODO",
        required_capabilities=frozenset({TRACKING}),
        nparts=_NPARTS_SMALL,
        observables=_OBS_TRACKING,
        tolerances={"default": 5e-3},
        reference="impactx",  # ImpactX ExactQuad+ExactDrift defines truth
        # Consistent exact (non-paraxial, nonlinear) model: ExactQuad + ExactDrift.
        # PyORBIT has no exact nonlinear quad and no exact drift (TEAPOT is chromatic-
        # paraxial only), so it cannot express this model. (SciBmad CAN: MatrixKick is a
        # symplectic integrator of the exact quad Hamiltonian + exact_drift.)
        unsupported_codes={
            "pyorbit": "no exact quad/drift (TEAPOT is chromatic-paraxial only)",
        },
    ),
    "htu": Scenario(
        name="htu",
        display_name="HTU beamline (chromatic)",
        required_capabilities=frozenset({TRACKING}),
        nparts=_NPARTS_SMALL,
        observables=_OBS_TRACKING,
        # HTU is a dispersive line (chicane) with 2.5% energy spread: different chromatic
        # bend/drift models legitimately differ at the few-% level, so use a 3% tolerance
        # (cf. fodo 5e-3). Observed cross-code spread vs ImpactX is <=~2.9%.
        tolerances={"default": 3e-2},
        reference="impactx",
        # Chromatic-paraxial model (quads + drifts). Cheetah lacks a chromatic quad/drift and
        # SciBmad lacks a chromatic drift, so they run the costlier exact map -> asterisk.
        untuned_codes={
            "cheetah": "no chromatic quad/drift; runs the exact drift_kick_drift map",
            "scibmad": "drift is exact-only; runs an exact (non-paraxial) drift",
            "bmad": "only exact bmad_standard tracking; no chromatic-paraxial model",
        },
    ),
    "htu_spin": Scenario(
        name="htu_spin",
        display_name="HTU beamline (spin depolarization)",
        # same lattice as htu, but with Thomas-BMT spin tracking
        required_capabilities=frozenset({TRACKING, SPIN}),
        nparts=_NPARTS_SMALL,
        observables=_OBS_SPIN,
        # beam starts fully spin-aligned (+z); the dispersive chicane differentially
        # precesses spins -> depolarization (sigma_s). Spin depolarization is a derivative-
        # like, model-sensitive observable: different bend models amplify the ~3% orbital
        # htu spread to ~10% in the spin spread (SciBmad ~2.5% vs ImpactX, Xsuite ~11%), so
        # use a 15% tolerance.
        tolerances={"default": 1.5e-1},
        reference="impactx",
        # Cheetah/pyAT/PyORBIT have no spin tracking -> unsupported via the SPIN capability.
        # Xsuite/SciBmad are SPIN-capable; their htu_spin templates are pending (not_in_harness).
    ),
    # Two separate space-charge benchmarks: 3D for the codes that do full 3D PIC, and
    # 2.5D for those that do 2.5D. A code without the matching model is simply
    # "unsupported" in that benchmark (no dashed model-mismatch needed).
    "spacecharge": Scenario(
        name="spacecharge",
        display_name="3D space charge",
        required_capabilities=frozenset({SPACE_CHARGE_3D}),
        nparts=_NPARTS_SC,
        observables=_OBS_SC,
        tolerances={"default": 5e-2},  # space charge: looser; implementations differ
        reference="impactx",
        intended_model="3d-pic",
        # ImpactX/Cheetah/Xsuite use the rigorous integrated Green function (IGF); PyORBIT
        # uses a cheaper, less-rigorous point-1/r grid Green function (agrees here, on a
        # round beam / fine grid) -> asterisk so its model-driven speed edge is visible.
        untuned_codes={
            "pyorbit": "cheaper, less-rigorous point-1/r Green function (not the integrated GF)",
        },
        untuned_note="*  uses a cheaper, less-rigorous Green function (point-1/r vs the integrated GF)",
    ),
    "spacecharge_2p5d": Scenario(
        name="spacecharge_2p5d",
        display_name="2.5D space charge",
        required_capabilities=frozenset({SPACE_CHARGE_2P5D}),
        nparts=_NPARTS_SC,
        observables=_OBS_SC,
        tolerances={"default": 5e-2},
        reference="impactx",  # ImpactX 2.5D PIC is the 2.5D reference
        intended_model="2.5d-pic",
        untuned_codes={
            "pyorbit": "cheaper, less-rigorous point-1/r Green function (not the integrated GF)",
        },
        untuned_note="*  uses a cheaper, less-rigorous Green function (point-1/r vs the integrated GF)",
    ),
}


# --------------------------------------------------------------------------- #
# Configs (build / runtime variants of a code)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """A concrete build/runtime variant of a code."""

    name: str
    code: str
    precision: str  # "single" | "double"
    device: str = "cpu"  # "cpu" | "cuda"
    ncores: Optional[int] = None  # None -> harness default (nproc locally, 4 in CI)
    #: space-charge model this variant uses, for model-mismatch detection
    sc_model: Optional[str] = None
    #: free-form knobs forwarded into the run template's Jinja context
    options: dict = field(default_factory=dict)
    #: override the pixi env (e.g. impactx-sp: a separate single-precision compiled build)
    env_override: Optional[str] = None
    #: the verification baseline (ImpactX DP, IEEE/no-fast-math): its observables are the truth
    #: that validation compares every run against (it is a normal, shown IEEE bar too).
    is_reference: bool = False
    #: fast-math variant. False = IEEE (the solid front bar); True = fast-math (drawn as a lighter
    #: bar *behind* its IEEE sibling). Compile-time codes get a separate env (see fm_companions);
    #: runtime-toggle codes (cheetah-compiled, xsuite) reuse the env and flip it via BENCH_FASTMATH.
    fast_math: bool = False

    @property
    def pixi_env(self) -> str:
        return self.env_override or CODES[self.code].pixi_env

    @property
    def base_name(self) -> str:
        """Name of the IEEE sibling this config pairs with in plots (self if IEEE)."""
        return self.name[:-3] if self.name.endswith("-fm") else self.name


def _cfg(name, code, precision, **kw) -> Config:
    return Config(name=name, code=code, precision=precision, **kw)


# Compile-time fast-math codes need a SECOND build+env for the fast-math variant; the map is
# base-env -> fast-math-env. Runtime-toggle codes (cheetah-compiled, xsuite) are absent here and
# reuse their env (the runner sets BENCH_FASTMATH / TORCHINDUCTOR_USE_FAST_MATH per config).
_FM_BUILD_ENV = {
    "impactx": "impactx-fm", "impactx-sp": "impactx-sp-fm",
    "impactx-cuda-dp": "impactx-cuda-dp-fm", "impactx-cuda-sp": "impactx-cuda-sp-fm",
    "pyat": "pyat-fm", "pyorbit": "pyorbit-fm", "bmad": "bmad-fm",
}


def _supports_fastmath(cfg: Config) -> bool:
    """Whether a fast-math (overlay) variant exists for this base config."""
    if cfg.code == "scibmad":
        return False  # no fast-math knob (Julia 1.12 removed --math-mode; no @fastmath upstream)
    if cfg.code == "cheetah" and cfg.options.get("compile") != "inductor":
        return False  # eager Torch has no fast-math; only the compiled variant does
    return True


def _fm_companion(cfg: Config) -> Optional[Config]:
    """The fast-math sibling of an IEEE base config (or None if unsupported)."""
    if cfg.fast_math or not _supports_fastmath(cfg):
        return None
    env = _FM_BUILD_ENV.get(cfg.pixi_env, cfg.pixi_env)  # separate build env, or same (runtime)
    return Config(
        name=cfg.name + "-fm", code=cfg.code, precision=cfg.precision, device=cfg.device,
        ncores=cfg.ncores, sc_model=cfg.sc_model, options=cfg.options,
        env_override=env, fast_math=True,
    )


# NOTE: only CPU configs are defined here. The GitHub-hosted runners are CPU-only
# and serve as the single public source of truth. GPU (device="cuda") variants will
# be added later for local runs whose results are committed to the same harness; the
# `device` field already supports them and the runner filters by --device (default cpu).
# One config per code variant. Configs are scenario-agnostic: the (code, scenario)
# run template picks the right physics (e.g. 3D vs 2.5D space charge), so there is no
# need for separate per-SC-model configs.
# NOTE: only CPU configs are defined here. The GitHub-hosted runners are CPU-only
# and serve as the single public source of truth. GPU (device="cuda") variants will
# be added later for local runs whose results are committed to the same harness; the
# `device` field already supports them and the runner filters by --device (default cpu).
# BASE (IEEE / non-fast-math) configs -- the solid front bars. Each fast-math-capable base gets a
# "-fm" companion generated below (a lighter bar drawn behind it); see _fm_companion.
_BASE_CONFIGS = [
    # -- ImpactX: OpenMP + SIMD; 3D-PIC space charge. The DP-CPU IEEE build is also the verification
    #    baseline (is_reference) that all runs are validated against.
    _cfg("impactx-cpu-simd-dp", "impactx", DOUBLE,
         options={"simd": True, "compute": "OMP"}, is_reference=True),
    # -- Cheetah: PyTorch threads; eager (no fast-math) + torch.compile (Inductor; fast-math via -fm)
    _cfg("cheetah-cpu-dp", "cheetah", DOUBLE, options={"compile": "none"}),
    _cfg("cheetah-cpu-compiled-dp", "cheetah", DOUBLE, options={"compile": "inductor"}),
    # -- pyAT: OpenMP, native; tracking only -----------------------------------
    _cfg("pyat-cpu-dp", "pyat", DOUBLE, options={}),
    # -- PyORBIT3: MPI; does both 3D and 2.5D PIC (template per scenario) -------
    _cfg("pyorbit-cpu-dp", "pyorbit", DOUBLE, options={}),
    # -- Xsuite: OpenMP context; 2.5D PIC --------------------------------------
    _cfg("xsuite-cpu-dp", "xsuite", DOUBLE, options={}),
    # -- SciBmad.jl: Julia threads; tracking only (no fast-math knob) ----------
    _cfg("scibmad-cpu-dp", "scibmad", DOUBLE, options={}),
    # -- Bmad: native Fortran driver (libbmad, no Tao); exact-native + spin -----
    _cfg("bmad-cpu-dp", "bmad", DOUBLE, options={}),
    # -- Single-precision (FP32) variants: only ImpactX/Cheetah/SciBmad. ImpactX SP is a SEPARATE
    #    compiled build (impactx-sp env); Cheetah/SciBmad SP is a runtime dtype switch.
    _cfg("impactx-cpu-simd-sp", "impactx", SINGLE,
         options={"simd": True, "compute": "OMP"}, env_override="impactx-sp"),
    _cfg("cheetah-cpu-sp", "cheetah", SINGLE, options={"compile": "none"}),
    _cfg("cheetah-cpu-compiled-sp", "cheetah", SINGLE, options={"compile": "inductor"}),
    _cfg("scibmad-cpu-sp", "scibmad", SINGLE, options={}),
    # -- GPU (CUDA) variants, device="cuda" -> own pixi GPU env. 1 GPU by default (DP+SP). Only
    #    ImpactX/Cheetah/SciBmad/Xsuite have a CUDA path; pyAT/PyORBIT/Bmad stay CPU-only. ImpactX
    #    needs a separate compiled CUDA build per precision; Cheetah/Xsuite/SciBmad switch at runtime.
    _cfg("impactx-cuda-dp", "impactx", DOUBLE, device="cuda",
         options={"compute": "CUDA"}, env_override="impactx-cuda-dp"),
    _cfg("impactx-cuda-sp", "impactx", SINGLE, device="cuda",
         options={"compute": "CUDA"}, env_override="impactx-cuda-sp"),
    _cfg("cheetah-cuda-dp", "cheetah", DOUBLE, device="cuda",
         options={"compile": "none"}, env_override="cheetah-gpu"),
    _cfg("cheetah-cuda-sp", "cheetah", SINGLE, device="cuda",
         options={"compile": "none"}, env_override="cheetah-gpu"),
    _cfg("cheetah-cuda-compiled-dp", "cheetah", DOUBLE, device="cuda",
         options={"compile": "inductor"}, env_override="cheetah-gpu"),
    _cfg("cheetah-cuda-compiled-sp", "cheetah", SINGLE, device="cuda",
         options={"compile": "inductor"}, env_override="cheetah-gpu"),
    _cfg("xsuite-cuda-dp", "xsuite", DOUBLE, device="cuda", options={}, env_override="xsuite-gpu"),
    _cfg("scibmad-cuda-dp", "scibmad", DOUBLE, device="cuda", options={}, env_override="scibmad-gpu"),
    _cfg("scibmad-cuda-sp", "scibmad", SINGLE, device="cuda", options={}, env_override="scibmad-gpu"),
]

CONFIGS: dict[str, Config] = {c.name: c for c in _BASE_CONFIGS}
# Fast-math overlay companions (compile-time codes -> their -fm env; runtime codes -> same env).
for _b in _BASE_CONFIGS:
    _fm = _fm_companion(_b)
    if _fm is not None:
        CONFIGS[_fm.name] = _fm


# --------------------------------------------------------------------------- #
# Capability resolution
# --------------------------------------------------------------------------- #
def support_status(config: Config, scenario: Scenario) -> tuple[str, str]:
    """Return ``(status, reason)``.

    ``status`` is one of ``"supported"`` or ``"unsupported_physics"``. ``reason``
    is a short human label used for plot annotations when unsupported.
    """
    code = CODES[config.code]

    # explicit per-scenario exclusion (e.g. missing element types)
    if config.code in scenario.unsupported_codes:
        return "unsupported_physics", scenario.unsupported_codes[config.code]

    # required physics capabilities
    missing = scenario.required_capabilities - code.capabilities
    if missing:
        return "unsupported_physics", "no " + ", ".join(sorted(missing))

    # precision availability
    if config.precision not in code.precisions:
        return "unsupported_physics", f"no {config.precision} precision"

    return "supported", ""


def model_status(config: Config, scenario: Scenario) -> Optional[str]:
    """Return a model-mismatch label if the config's model differs from intended.

    ``None`` means the model matches (or the scenario has no intended model).
    """
    if scenario.intended_model is None or config.sc_model is None:
        return None
    if config.sc_model != scenario.intended_model:
        return config.sc_model
    return None


def configs_for(codes=None) -> list[Config]:
    """All configs, optionally filtered to a set of code names."""
    out = list(CONFIGS.values())
    if codes:
        codes = set(codes)
        out = [c for c in out if c.code in codes]
    return out
