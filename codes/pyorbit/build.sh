#!/usr/bin/env bash
# Build & install PyORBIT3 from the LOCAL source checkout into the active pixi env
# (Meson/C++/FFTW, MPI via MPICH). CXXFLAGS=-march=native -mtune=native come from the
# feature activation env. Override the source path with PYORBIT_SRC.
set -eu -o pipefail

SRC="${PYORBIT_SRC:-/home/axel/src/PyORBIT3}"
# Fall back to a fresh clone if the local checkout is absent (e.g. in CI).
if [ ! -f "$SRC/pyproject.toml" ]; then
    SRC=".builds/src/PyORBIT3"
    mkdir -p .builds/src
    if [ ! -d "$SRC/.git" ]; then
        git clone --depth 1 "${PYORBIT_REPO:-https://github.com/PyORBIT-Collaboration/PyORBIT3.git}" "$SRC"
    fi
fi

# PyORBIT3's meson.build derives the version via `python -m setuptools_scm`, but pip
# copies the source to a temp build dir without .git -> empty version -> build fails.
# Pin a valid PEP440 version (derive from the tag if available, else a fixed fallback).
PYORBIT_VER="${PYORBIT_VERSION:-$(git -C "$SRC" describe --tags --abbrev=0 2>/dev/null || echo 3.1.0)}"
export SETUPTOOLS_SCM_PRETEND_VERSION="$PYORBIT_VER"
echo "Using PyORBIT3 version $SETUPTOOLS_SCM_PRETEND_VERSION"

# Perlmutter/portable microarch + fast-math: meson reads CXXFLAGS/CFLAGS from the env; append after
# the activation's -march=native (last -march wins → BENCH_ARCH=znver3 on Perlmutter, default native
# is a no-op); add -ffast-math when BENCH_FASTMATH=1 (the default). See machines/PERLMUTTER.md.
MARCH="${BENCH_ARCH:-native}"
EXTRA_FLAGS="-march=${MARCH} -mtune=${MARCH}"
case "${BENCH_FASTMATH:-1}" in 0|off|OFF|false) : ;; *) EXTRA_FLAGS="${EXTRA_FLAGS} -ffast-math -fno-finite-math-only" ;; esac
export CXXFLAGS="${CXXFLAGS:-} ${EXTRA_FLAGS}"
export CFLAGS="${CFLAGS:-} ${EXTRA_FLAGS}"
echo "PyORBIT CXXFLAGS=${CXXFLAGS}"

python -m pip install --upgrade pip
# release buildtype -> -O3 -DNDEBUG; MPICH backend; build deps already in the pixi env.
python -m pip install --no-build-isolation --force-reinstall \
    --config-settings=setup-args="-Dbuildtype=release" \
    --config-settings=setup-args="-DUSE_MPI=mpich" \
    -v "$SRC"

python -c "from orbit.core.bunch import Bunch; from orbit.teapot import TEAPOT_Lattice; from orbit.core.spacecharge import SpaceChargeCalc3D, SpaceChargeCalc2p5D; print('PyORBIT3 import OK')"
