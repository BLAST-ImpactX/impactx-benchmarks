"""Capture host / OS / CPU / compiler / version metadata.

Used both to annotate the results file and to compose the ``benchmarks``-branch
commit message. Everything degrades gracefully when a tool is missing.
"""

from __future__ import annotations

import datetime
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text else ""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def machine_slug() -> str:
    """Filesystem-/branch-safe identifier for this machine.

    Overridable via ``$BENCH_MACHINE_SLUG`` so CI uses a stable name (e.g.
    ``github-ubuntu-24.04``) instead of a random runner hostname.
    """
    import os

    override = os.environ.get("BENCH_MACHINE_SLUG")
    host = override or socket.gethostname().split(".")[0]
    return re.sub(r"[^A-Za-z0-9_.-]", "-", host) or "unknown"


def _os_pretty_name() -> str:
    osr = Path("/etc/os-release")
    if osr.exists():
        for line in osr.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    return platform.platform()


def _lscpu_map() -> dict:
    out: dict[str, str] = {}
    for line in _run(["lscpu"]).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _cpu_model() -> str:
    m = _lscpu_map()
    if "Model name" in m:
        return m["Model name"]
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def cpu_info() -> dict:
    """Detailed CPU identity -- crucial since hosted CI runners use a mixed pool.

    Captures vendor, model name, family/model/stepping, frequencies, cache sizes,
    SIMD flags and (via archspec) the microarchitecture name (e.g. 'skylake'),
    which together identify the exact CPU generation behind a timing.
    """
    m = _lscpu_map()
    keys = [
        "Vendor ID", "Model name", "CPU family", "Model", "Stepping",
        "CPU max MHz", "CPU min MHz", "BogoMIPS",
        "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)",
        "L1d cache", "L1i cache", "L2 cache", "L3 cache",
    ]
    info = {k: m[k] for k in keys if k in m}

    flags = set(m.get("Flags", "").split())
    simd = [f for f in (
        "sse4_2", "avx", "avx2", "fma",
        "avx512f", "avx512dq", "avx512bw", "avx512vl", "amx_tile",
    ) if f in flags]
    info["simd"] = ",".join(simd)

    try:
        import archspec.cpu

        info["microarch"] = archspec.cpu.host().name
    except Exception:
        pass
    return info


def gpu_info() -> list[dict]:
    """Per-GPU identity via ``nvidia-smi`` -- name, total memory, driver.

    Returns ``[]`` on a host with no NVIDIA GPU (e.g. a Perlmutter CPU node), so it is safe to
    call on every run and only populates on GPU nodes. Perlmutter's general ``gpu`` pool mixes
    40GB and 80GB A100s (the ``&hbm80g`` constraint restricts to the small 80GB subset -> long
    queue), so recording which card actually ran a timing keeps the numbers interpretable -- the
    GPU analogue of the CPU microarch. The card name already encodes the memory tier
    ('A100-SXM4-80GB' vs '-40GB'), which is exactly what makes a 10^9-particle OOM legible.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    # name,memory.total,driver_version are supported on every modern driver (avoid newer
    # query fields that would fail the whole CSV on an older nvidia-smi).
    raw = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    gpus: list[dict] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0]:
            gpus.append({
                "name": parts[0],
                "memory_total": parts[1] if len(parts) > 1 else "",
                "driver": parts[2] if len(parts) > 2 else "",
            })
    return gpus


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0


def _mem_total() -> str:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return f"{kb / 1024 / 1024:.1f} GiB"
    return ""


def _tool_version(tool: str, args: Optional[list] = None) -> str:
    if shutil.which(tool) is None:
        return ""
    return _first_line(_run([tool] + (args or ["--version"])))


def _cpu_governor() -> str:
    """CPU scaling governor (affects absolute throughput; recorded for reproducibility)."""
    from .affinity import cpu_governor
    return cpu_governor()


def host_metadata() -> dict:
    """OS / kernel / CPU / RAM / compiler versions of the executing host."""
    return {
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "machine_slug": machine_slug(),
        "os": _os_pretty_name(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": _cpu_count(),
        "cpu": cpu_info(),
        "cpu_governor": _cpu_governor(),
        "mem_total": _mem_total(),
        "python": platform.python_version(),
        "compilers": {
            "gcc": _tool_version("gcc", ["-dumpfullversion"]) or _tool_version("gcc"),
            "gxx": _tool_version("g++", ["-dumpfullversion"]) or _tool_version("g++"),
            "cmake": _tool_version("cmake"),
            "ninja": _tool_version("ninja"),
            "julia": _tool_version("julia"),
            "pixi": _tool_version("pixi"),
        },
    }


# Packages whose version we record per code environment (central dependencies +
# the code itself). Missing packages are simply skipped.
_CODE_PACKAGES = {
    "impactx": ["impactx", "fftw", "libblas", "liblapack"],
    "cheetah": ["cheetah-accelerator", "pytorch", "numpy"],
    "pyat": ["accelerator-toolbox", "numpy", "scipy"],
    "pyorbit": ["fftw", "mpich", "mpi4py", "numpy"],
    "xsuite": ["xsuite", "xtrack", "xfields", "numpy"],
    "scibmad": ["julia"],
    "elegant": ["mpich", "gsl", "liblapack", "fftw"],  # elegant itself is a source build (see _CODE_SRC_DIRS)
    "helix": ["pytorch", "numpy", "scipy"],  # linac_gen is a pip --no-deps source install (see _CODE_SRC_DIRS)
}


def env_package_versions(env: str) -> dict:
    """Resolved package versions for a pixi environment (best-effort, JSON)."""
    import json

    raw = _run(["pixi", "list", "--environment", env, "--json"])
    versions: dict[str, str] = {}
    if not raw:
        return versions
    try:
        pkgs = json.loads(raw)
    except json.JSONDecodeError:
        return versions
    wanted = set(_CODE_PACKAGES.get(env, []))
    for pkg in pkgs:
        name = pkg.get("name", "")
        if not wanted or name in wanted:
            versions[name] = pkg.get("version", "")
    return versions


def pip_package_versions(env: str, packages: list[str]) -> dict:
    """Versions for PyPI-installed packages (e.g. xsuite sub-packages, cheetah)."""
    code = (
        "import importlib.metadata as m, json, sys\n"
        "out={}\n"
        "for p in sys.argv[1:]:\n"
        "    try: out[p]=m.version(p)\n"
        "    except Exception: pass\n"
        "print(json.dumps(out))"
    )
    raw = _run(["pixi", "run", "-e", env, "python", "-c", code] + packages)
    import json

    try:
        return json.loads(raw.splitlines()[-1]) if raw else {}
    except (json.JSONDecodeError, IndexError):
        return {}


def code_versions(envs: list[str]) -> dict:
    """Collect resolved versions for the given code environments."""
    out: dict[str, dict] = {}
    for env in envs:
        versions = env_package_versions(env)
        out[env] = versions
    return out


# A single human-facing version string per code (shown on plots). For from-source codes
# we use ``git describe`` of the built checkout (captures the exact commit / PR branch);
# for pip codes the installed package version.
_CODE_SRC_DIRS = {
    # DP-CPU-IEEE reference tree first (the tag TAG=<device>-<prec><fmtag> naming build.sh uses now);
    # then legacy paths for older trees.
    "impactx": [".builds/src/impactx-cpu-dp-ieee", ".builds/src/impactx-cpu-dp", ".builds/src/impactx"],
    "pyorbit": [os.environ.get("PYORBIT_SRC", ""), "/home/axel/src/PyORBIT3", ".builds/src/PyORBIT3"],
    "scibmad": [os.environ.get("SCIBMAD_PATH", ""), "/home/axel/src/SciBmad.jl", ".builds/src/SciBmad.jl"],
    "bmad": [os.environ.get("BMAD_SRC", ""), "/home/axel/src/bmad-ecosystem", ".builds/src/bmad-ecosystem"],
    "elegant": [os.environ.get("ELEGANT_SRC", ""), "/home/axel/src/elegant", ".builds/src/elegant"],
    "helix": [os.environ.get("HELIX_SRC", ""), "/home/axel/src/HELIX", ".builds/src/HELIX"],
}
_CODE_MAIN_PKG = {
    "impactx": "impactx",
    "cheetah": "cheetah-accelerator",
    "pyat": "accelerator-toolbox",
    "xsuite": "xsuite",
    "helix": "linac_gen",
}


def _git_describe(path: str) -> str:
    if not path or not Path(path, ".git").exists():
        return ""
    out = _run(["git", "-C", path, "describe", "--tags", "--always", "--dirty"])
    return out.strip() if out else ""


def _built_ref(path: str) -> str:
    """The human ref the build stamped into the checkout (.bench_ref), e.g. "26.08". DRY: this is
    exactly what build.sh checked out, so build and plot label agree by construction. Preferred over
    git describe, which falls back to a bare SHA when a shallow tag fetch leaves no local tag."""
    try:
        return Path(path, ".bench_ref").read_text().strip()
    except Exception:
        return ""


def code_version_label(code: str) -> str:
    """Best single version string for a code: the build's stamped ref (.bench_ref, DRY) first, else
    git describe of the source checkout (captures the exact commit/PR), else the installed package
    version."""
    for d in _CODE_SRC_DIRS.get(code, []):
        v = _built_ref(d) or _git_describe(d)
        if v:
            return v
    pkg = _CODE_MAIN_PKG.get(code)
    if pkg:
        v = pip_package_versions(code, [pkg]).get(pkg, "")
        if v:
            return v
    return "?"


def code_version_labels(codes: list[str]) -> dict:
    """{code: version-string} for plot footers."""
    return {c: code_version_label(c) for c in codes}


def as_commit_message(meta: dict, summary: str = "") -> str:
    """Compose the metadata-rich commit message for the benchmarks branch."""
    host = meta.get("host", {})
    comps = host.get("compilers", {})
    cpu = host.get("cpu", {})
    cpu_detail = "  ".join(
        s for s in [
            f"vendor={cpu.get('Vendor ID', '')}" if cpu.get("Vendor ID") else "",
            f"family={cpu.get('CPU family', '')}/model={cpu.get('Model', '')}/step={cpu.get('Stepping', '')}"
            if cpu.get("CPU family") else "",
            f"uarch={cpu.get('microarch', '')}" if cpu.get("microarch") else "",
            f"maxMHz={cpu.get('CPU max MHz', '')}" if cpu.get("CPU max MHz") else "",
            f"simd={cpu.get('simd', '')}" if cpu.get("simd") else "",
        ] if s
    )
    gpus = meta.get("gpu") or []
    gpu_line = ""
    if gpus:
        g0 = gpus[0]
        gpu_line = (f"{len(gpus)}x {g0.get('name', '')} ({g0.get('memory_total', '')})"
                    f"  driver {g0.get('driver', '')}").strip()
    lines = [
        f"bench: {host.get('machine_slug', 'unknown')} @ {host.get('timestamp_utc', '')}",
        "",
        f"machine:  {host.get('hostname', '')}",
        f"os:       {host.get('os', '')} (kernel {host.get('kernel', '')}, {host.get('arch', '')})",
        f"cpu:      {host.get('cpu_model', '')} x{host.get('cpu_count', '')}",
        f"cpu-id:   {cpu_detail}",
        f"gpu:      {gpu_line}",
        f"mem:      {host.get('mem_total', '')}",
        f"python:   {host.get('python', '')}",
        f"gcc:      {comps.get('gcc', '')}   g++: {comps.get('gxx', '')}",
        f"cmake:    {comps.get('cmake', '')}   ninja: {comps.get('ninja', '')}",
        f"julia:    {comps.get('julia', '')}",
        f"pixi:     {comps.get('pixi', '')}",
        "",
        "code & dependency versions:",
    ]
    for env, versions in sorted(meta.get("versions", {}).items()):
        if versions:
            vs = ", ".join(f"{k}={v}" for k, v in sorted(versions.items()))
            lines.append(f"  {env}: {vs}")
    if summary:
        lines += ["", summary]
    return "\n".join(lines)


def main() -> None:
    import json

    meta = {"host": host_metadata()}
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
