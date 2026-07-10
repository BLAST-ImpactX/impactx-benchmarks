"""Multithreaded CPU FFT for Xsuite space charge, via the supported ``plan_FFT`` extension point.

Why this exists:
  xfields' PIC Poisson solve is FFT-bound (~77% of a 3D SC step, ~64% of 2.5D). On a CPU context
  xobjects only offers a *threaded* FFT through pyfftw, and falls back to **single-threaded
  numpy/pocketfft** otherwise. pyfftw is (a) not installed and (b) currently broken with xfields:
  xobjects' pyfftw ``FFTCpu`` binds its plan to one buffer (``assert data is self.data``), while
  xfields reuses one plan across different arrays -> ``AssertionError``. So Xsuite SC runs its FFT
  on a single core, leaving it several-fold slower than ImpactX (``fftw3_omp``) / PyORBIT.

Fix (supported, not a monkeypatch):
  ``Context.plan_FFT()`` is the public FFT-plan factory and ``SpaceCharge3D(fftplan=...)`` is an
  exposed argument. We subclass ``ContextCpu`` and override ``plan_FFT()`` to return a plan backed
  by ``scipy.fft`` with ``workers = OpenMP threads``. scipy.fft is a drop-in for numpy.fft
  (bit-identical result, verified) but multithreaded, and -- unlike the pyfftw ``FFTCpu`` -- has no
  buffer binding, so it works with how xfields reuses the plan.

Use ``ContextCpuThreadedFFT(omp_num_threads=...)`` in place of ``xo.ContextCpu(...)``.
"""
from __future__ import annotations

import os

import scipy.fft as _spfft
import xobjects as xo


def _fastmath_on() -> bool:
    """Whether to enable fast-math (the harness default; toggled by BENCH_FASTMATH=0)."""
    return os.environ.get("BENCH_FASTMATH", "1").lower() not in ("0", "off", "false", "no")


def _make_cupy_context():
    """A ``ContextCupy`` that (by default) compiles every kernel with ``--use_fast_math`` — xobjects
    passes no fast-math on GPU, so this matches the CPU ``-ffast-math`` / ImpactX ``AMReX_CUDA_FASTMATH``
    baseline. Defined lazily so this module still imports in the CPU xsuite env (which has no cupy)."""
    import xobjects as xo

    if not _fastmath_on():
        return xo.ContextCupy()

    class _ContextCupyFastMath(xo.ContextCupy):
        def build_kernels(self, *args, **kwargs):  # inject --use_fast_math into the NVRTC options
            kwargs["extra_compile_args"] = ("--use_fast_math",
                                            *kwargs.get("extra_compile_args", ()))
            return super().build_kernels(*args, **kwargs)

    return _ContextCupyFastMath()


class _ScipyFFTPlan:
    """xobjects-compatible FFT plan (in-place ``transform``/``itransform``) using scipy.fft."""

    def __init__(self, axes, workers: int):
        self.axes = tuple(axes)
        self.workers = max(int(workers), 1)

    def transform(self, data):  # in-place forward FFT (matches xobjects FFTCpu)
        data[:] = _spfft.fftn(data, axes=self.axes, workers=self.workers)

    def itransform(self, data):  # in-place inverse FFT
        data[:] = _spfft.ifftn(data, axes=self.axes, workers=self.workers)


class ContextCpuThreadedFFT(xo.ContextCpu):
    """``ContextCpu`` whose ``plan_FFT`` returns a multithreaded scipy.fft plan.

    FFT threads = ``omp_get_max_threads()`` (i.e. the run's OpenMP thread count). Falls back to the
    stock (numpy) plan on any error, so it can never break a run.
    """

    def plan_FFT(self, data, axes):
        try:
            workers = self.omp_get_max_threads()
            return _ScipyFFTPlan(axes, workers)
        except Exception:
            return super().plan_FFT(data, axes)


def _make_cpu_context(threaded_fft: bool, omp_num_threads: int):
    """A CPU context (threaded-FFT for space charge, else plain) that, by default, compiles kernels
    with ``-ffast-math`` -- the runtime fast-math toggle (xsuite's activation CFLAGS no longer bakes
    it). The kernel-cache salt includes ``extra_compile_args``, so IEEE and fast-math kernels cache
    separately. ``BENCH_FASTMATH=0`` gives the plain IEEE context."""
    import xobjects as xo
    base = ContextCpuThreadedFFT if threaded_fft else xo.ContextCpu
    if not _fastmath_on():
        return base(omp_num_threads=omp_num_threads)

    class _FastMathCpu(base):
        def build_kernels(self, *args, **kwargs):
            # -fno-finite-math-only: -ffast-math implies -ffinite-math-only, which redirects some
            # libm calls to __*_finite symbols that modern glibc (>=2.31) removed -> undefined symbol
            # at load. Keep the fast-math opts, drop that redirection.
            kwargs["extra_compile_args"] = (
                "-ffast-math", "-fno-finite-math-only", *kwargs.get("extra_compile_args", ()))
            return super().build_kernels(*args, **kwargs)

    return _FastMathCpu(omp_num_threads=omp_num_threads)


def make_context(device: str, omp_num_threads: int = 0, threaded_fft: bool = False):
    """Return the xobjects context for the run.

    * ``device="cuda"`` -> ``ContextCupy`` (GPU; cupy's FFT is native/threaded on-device, so the
      CPU scipy-FFT/kernel-cache shims do not apply).
    * ``device="cpu"``  -> ``ContextCpuThreadedFFT`` for space charge (FFT-bound; threaded scipy.fft)
      or a plain ``ContextCpu`` for pure tracking.
    """
    if device == "cuda":
        return _make_cupy_context()  # ContextCupy (+ --use_fast_math unless BENCH_FASTMATH=0)
    return _make_cpu_context(threaded_fft, omp_num_threads)  # +-ffast-math unless BENCH_FASTMATH=0
