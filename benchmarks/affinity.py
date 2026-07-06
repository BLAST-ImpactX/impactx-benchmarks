"""CPU affinity: make the ``MPI_ranks × threads ≤ ncores`` budget *physically real*.

Without pinning, ``mpirun -np N`` and a plain threaded ``python`` both float across all
logical CPUs, so on a multi-core workstation the "N-core budget" is only logical and the
OS placement (and P-core vs E-core, hyperthreading, turbo) dominates the measurement
rather than the code (see FIXME_local_running_harness_pinning_MPI-vs-OpenMP.md). We pin
every run to the SAME set of ``ncores`` *distinct physical cores* so OpenMP and MPI layouts
compete on identical hardware.
"""
from __future__ import annotations

import os
from pathlib import Path

_SYS_CPU = Path("/sys/devices/system/cpu")


def _parse_cpu_list(s: str) -> list[int]:
    """Parse a Linux CPU list like '0,2-4,7' into [0,2,3,4,7]."""
    out: list[int] = []
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _socket_of(cid: int) -> int | None:
    """Physical-package (socket) id of logical CPU ``cid``, or None if unreadable."""
    try:
        return int((_SYS_CPU / f"cpu{cid}/topology/physical_package_id").read_text())
    except Exception:
        return None


def physical_cpus(socket: int | None = None) -> list[int] | None:
    """One logical CPU per physical core (skip hyperthread siblings), in CPU-id order.

    If ``socket`` is given, restrict to that physical package -- e.g. pin to ONE socket of a
    dual-socket node (Perlmutter CPU = 2× EPYC 7763 → 64 physical cores per socket), so a run
    never spans both NUMA packages. Returns None if the topology cannot be read.
    """
    try:
        ids = sorted(
            int(p.name[3:]) for p in _SYS_CPU.glob("cpu[0-9]*") if p.name[3:].isdigit()
        )
    except Exception:
        return None
    seen: set[frozenset] = set()
    picked: list[int] = []
    for cid in ids:
        if socket is not None and _socket_of(cid) != socket:
            continue
        try:
            sib = (_SYS_CPU / f"cpu{cid}/topology/thread_siblings_list").read_text()
        except Exception:
            return None
        key = frozenset(_parse_cpu_list(sib))
        if key in seen:
            continue
        seen.add(key)
        picked.append(cid)
    return picked or None


def budget_cpus(ncores: int) -> list[int]:
    """The CPU ids forming the ``ncores``-core budget, shared by every layout.

    Priority: ``$BENCH_CPU_LIST`` override → distinct physical cores (of ``$BENCH_SOCKET`` only,
    if set) → logical 0..n-1. Using distinct physical cores emulates a real N-vCPU runner; pinning
    to one ``$BENCH_SOCKET`` keeps a Perlmutter run on a single EPYC socket (no cross-socket spill).
    """
    env = os.environ.get("BENCH_CPU_LIST")
    if env:
        cpus = _parse_cpu_list(env)
    else:
        sock = os.environ.get("BENCH_SOCKET")
        socket = int(sock) if sock and sock.lstrip("-").isdigit() else None
        phys = physical_cpus(socket=socket)
        cpus = phys if (phys and len(phys) >= ncores) else list(range(ncores))
    return cpus[:ncores]


def cpu_governor() -> str:
    """The scaling governor of cpu0 (e.g. 'performance'/'powersave'), or '' if unknown."""
    try:
        return (_SYS_CPU / "cpu0/cpufreq/scaling_governor").read_text().strip()
    except Exception:
        return ""
