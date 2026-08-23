"""Shared, dependency-light helpers for the rendered run scripts.

Importable from any Python code environment (only needs numpy). Provides a
wall-clock timer (throughput reflects real parallel speedup) and a uniform
beam-moment / emittance calculation so every code reports comparable observables.
"""

from __future__ import annotations

import time

import numpy as np


def _gpu_sync():
    """Block until all queued GPU work finishes, for any GPU framework already imported by the
    run script. GPU kernel launches are ASYNCHRONOUS: without this, a wall-clock timer around a
    track call measures only the Python-side launch (precision-independent, ~instant for a fused
    torch.compile kernel), not the execution -- inflating throughput ~100-450x and making FP32 and
    FP64 look identical. We only touch a framework if the run already imported it (sys.modules), so
    this stays a no-op for CPU runs and for codes that sync themselves (AMReX ImpactX, CUDA.@sync
    SciBmad -- an extra sync there is harmless)."""
    import sys
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            cupy.cuda.Stream.null.synchronize()
        except Exception:
            pass


class Timer:
    """Wall-clock timer in nanoseconds (``with Timer() as t: ...; t.ns``). Drains the GPU on both
    entry (flush any pending warm-up work) and exit (wait for the timed kernels) so the interval is
    the GPU EXECUTION time, not the async launch time. See _gpu_sync."""

    def __enter__(self):
        _gpu_sync()
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        _gpu_sync()
        self.ns = time.perf_counter_ns() - self._t0
        return False


def _host(a):
    """Coerce to a flat float64 host (numpy) array. GPU arrays (cupy from Xsuite's ContextCupy,
    torch tensors, etc.) forbid implicit ``np.asarray`` -- pull them to host via ``.get()`` /
    ``.cpu()`` first. Plain numpy/host arrays pass straight through."""
    for attr in ("get", "cpu"):  # cupy: .get(); torch: .cpu() then numpy
        fn = getattr(a, attr, None)
        if callable(fn):
            a = fn()
            break
    a = getattr(a, "numpy", lambda: a)() if hasattr(a, "numpy") else a  # torch tensor -> numpy
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _emit(u, up):
    """Geometric RMS emittance sqrt(<u^2><u'^2> - <u u'>^2)."""
    u = _host(u)
    up = _host(up)
    u = u - u.mean()
    up = up - up.mean()
    cuu = np.mean(u * u)
    cpp = np.mean(up * up)
    cup = np.mean(u * up)
    return float(np.sqrt(max(cuu * cpp - cup * cup, 0.0)))


def beam_observables(x, px, y, py, tau=None, p=None) -> dict:
    """RMS sizes, geometric emittances and centroids of a particle distribution."""
    x = _host(x)
    y = _host(y)
    obs = {
        "sigma_x": float(np.std(x)),
        "sigma_y": float(np.std(y)),
        "emit_x": _emit(x, px),
        "emit_y": _emit(y, py),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
    }
    if tau is not None:
        obs["sigma_t"] = float(np.std(_host(tau)))
    return obs


def gaussian_twiss_plane(emit, beta, alpha, n, rng):
    """Sample (u, u') from a matched Gaussian with the given Twiss parameters.

    Returns an array of shape ``(2, n)``.
    """
    gamma = (1.0 + alpha * alpha) / beta
    cov = emit * np.array([[beta, -alpha], [-alpha, gamma]])
    return rng.multivariate_normal([0.0, 0.0], cov, int(n)).T
