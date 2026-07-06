"""Persistent on-disk cache for Xsuite (xobjects) CPU tracking kernels.

xobjects compiles each tracker kernel to a *random-named* ``.so`` and deletes it after loading
(``context_cpu.build_kernels``: ``module_name = module_name or uuid4().hex`` and
``clean_up_so = not module_name``). So every process -- and every auto-tune (rank,thread) layout
-- recompiles the kernel from scratch (the slow ``cc1 -O3`` you see in the warm-up). The library's
"prebuilt kernels" path only matches the configs xsuite ships, so our exact/spin/OpenMP lines miss
it and always compile.

This monkeypatch intercepts the dynamic-compile path and keys the compiled module on a SHA1 of the
*specialized source* (+ cdefs + compile args). On a hit it loads the cached ``.so`` instead of
recompiling; on a miss it compiles once with a stable name into a persistent dir and keeps it.

Why it's correct and safe:
* The key is the actual specialized C source, which already encodes the element classes, the config
  flags (exact drift, spin) via headers, and the context kind (``cpu_openmp`` vs ``cpu_serial``).
  A different line/config/context => different source => different key.
* OpenMP thread count is a *runtime* argument, not compiled in -- so the same ``.so`` is valid for
  every thread layout (1t/2t/4t) and is shared across them. The thread count is deliberately NOT
  in the key.
* It only touches the untimed warm-up compile; the timed track reuses the in-process kernel either
  way, so measured numbers are unaffected.
* Any error falls back to the original ``build_kernels`` -- it can never produce a wrong kernel.
"""
from __future__ import annotations

import glob
import hashlib
import os
import platform
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path


def _default_cache_dir() -> str:
    return os.environ.get(
        "XSUITE_KERNEL_CACHE",
        str(Path(__file__).resolve().parents[1] / ".builds" / "cache" / "xsuite_kernels"),
    )


@lru_cache(maxsize=1)
def _env_salt() -> str:
    """Everything (besides the source) that changes the compiled artifact -- so an upgrade, a
    flag change, or a different CPU/toolchain yields a DIFFERENT cache key (never a wrong hit).

    Covers: Python ABI, arch/platform, the real compile/link flags (``$CFLAGS`` etc. -- where
    ``-march=native`` actually comes from), the CPU instruction set (so a ``-march=native`` .so
    built on one machine is never loaded on a different CPU), the xobjects/xtrack versions
    (codegen/header/ABI changes), and the C compiler version.
    """
    parts = [sys.implementation.cache_tag, platform.machine(), sysconfig.get_platform()]
    for var in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CC", "CXX"):
        parts.append(f"{var}={os.environ.get(var, '')}")
    for mod in ("xobjects", "xtrack"):
        try:
            import importlib
            parts.append(f"{mod}={importlib.import_module(mod).__version__}")
        except Exception:
            parts.append(f"{mod}=?")
    # CPU instruction set -> distinguishes machines for -march=native (avoids loading an
    # incompatible .so that would SIGILL or measure the wrong code path)
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("flags") or line.startswith("Features"):
                    parts.append("isa=" + hashlib.sha1(line.encode()).hexdigest())
                    break
    except Exception:
        pass
    # C compiler identity: realpath + size + mtime (deterministic, no subprocess; pixi swaps the
    # binary on a toolchain upgrade, so size/mtime change -> key changes).
    import shutil
    for cc in (os.environ.get("CC"), "cc", "gcc"):
        if not cc:
            continue
        path = shutil.which(cc)
        if not path:
            continue
        try:
            real = os.path.realpath(path)
            st = os.stat(real)
            parts.append(f"cc={real}:{st.st_size}:{int(st.st_mtime)}")
            break
        except Exception:
            continue
    return "\n".join(parts)


def enable() -> None:
    """Idempotently monkeypatch xobjects' CPU kernel build to use a persistent source-hash cache."""
    try:
        from xobjects.context_cpu import (
            ContextCpu,
            classes_from_kernels,
            sort_classes,
        )
    except Exception:
        return
    if getattr(ContextCpu, "_xsk_persistent_cache", False):
        return

    _orig = ContextCpu.build_kernels
    cache_dir = _default_cache_dir()

    def build_kernels(self, kernel_descriptions, module_name=None, containing_dir=".",
                      sources=None, specialize=True, apply_to_source=(), save_source_as=None,
                      extra_compile_args=(), extra_link_args=(), extra_cdef="", extra_classes=(),
                      extra_headers=(), compile=True):  # noqa: A002 (match xobjects signature)
        # Only intercept the default dynamic-compile path (xtrack passes module_name=None there).
        # Also: cache ONLY plain CPU contexts. GPU contexts are separate classes (ContextCupy /
        # ContextPyopencl) with their own build_kernels, so they never reach this patch -- but we
        # guard explicitly so a CPU and a (future) GPU kernel can never be confused, and we fold
        # the context class into the key as well. A GPU cache, when added, needs its own
        # device-arch salt (compute capability), analogous to the CPU instruction-set salt below.
        if module_name is not None or not compile or type(self).__name__ != "ContextCpu":
            return _orig(self, kernel_descriptions, module_name, containing_dir, sources,
                         specialize, apply_to_source, save_source_as, extra_compile_args,
                         extra_link_args, extra_cdef, extra_classes, extra_headers, compile)
        try:
            classes = list(classes_from_kernels(kernel_descriptions)) + list(extra_classes)
            classes = sort_classes(classes)
            _, specialized = self._build_sources(
                sources=sources or [], classes=classes, extra_headers=extra_headers,
                apply_to_source=apply_to_source, specialize=specialize,
            )
            cdefs = "\n".join(c._gen_c_decl({}) for c in classes) + "\n" + extra_cdef
            key = hashlib.sha1(
                (specialized + cdefs + repr(tuple(extra_compile_args))
                 + repr(tuple(extra_link_args)) + "\x00" + type(self).__name__
                 + "\x00" + _env_salt()).encode()
            ).hexdigest()
            mod = "xsk_" + key
            os.makedirs(cache_dir, exist_ok=True)
            if glob.glob(os.path.join(cache_dir, mod + ".*")):
                # cache hit: load the prebuilt module (no compile)
                return self.kernels_from_file(
                    module_name=mod, kernel_descriptions=kernel_descriptions,
                    containing_dir=cache_dir,
                )
            # cache miss: compile once into the cache with a stable name (kept, since module_name set)
            return _orig(self, kernel_descriptions, mod, cache_dir, sources, specialize,
                         apply_to_source, save_source_as, extra_compile_args, extra_link_args,
                         extra_cdef, extra_classes, extra_headers, compile)
        except Exception:
            # never let caching break the build -> fall back to the stock path
            return _orig(self, kernel_descriptions, None, containing_dir, sources, specialize,
                         apply_to_source, save_source_as, extra_compile_args, extra_link_args,
                         extra_cdef, extra_classes, extra_headers, compile)

    ContextCpu.build_kernels = build_kernels
    ContextCpu._xsk_persistent_cache = True
