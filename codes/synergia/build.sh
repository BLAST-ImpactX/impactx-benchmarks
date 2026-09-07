#!/usr/bin/env bash
# Build & install Synergia3 (devel3) into the active pixi env.
#
# Synergia is a CMake project with a VENDORED Kokkos + Eigen (git submodules) and FetchContent
# cereal/Catch2. Backend: Kokkos OpenMP (CPU) or Kokkos CUDA + cuFFT (SYNERGIA_DEVICE=cuda).
#
# openPMD I/O is DISABLED (USE_OPENPMD_IO=OFF). The benchmark reads observables directly from the
# in-memory bunch (Core_diagnostics), never writing openPMD files, and Synergia's own recipe pins
# the FetchContent openPMD to v0.15.2, which does NOT compile with GCC 15 (a -Wtemplate-body /
# two-phase-lookup 'm_container' error in openPMD/backend/Container.hpp). Disabling it drops that
# dependency entirely and does not change any observable we measure. (Re-enable only with a
# GCC-15-compatible openPMD and USE_EXTERNAL_OPENPMD pointing at it.)
# It installs its python package to <prefix>/lib/python3.X/site-packages/synergia and its shared
# libs to <prefix>/lib with an install RPATH pointing there, so installing into $CONDA_PREFIX (the
# active pixi env) makes `import synergia` work and resolves the libs automatically -- the same
# install-into-the-env approach as codes/bmad and codes/elegant.
#
# Source resolution: the user's checkout at $SYNERGIA_SRC (default ~/src/synergia2) is used ONLY as
# a git source to clone from -- we NEVER modify it (its submodules are typically uninitialized and
# the brief forbids changing that checkout). We clone it (preserving its exact commit) into
# .builds/src/synergia2 and initialize the Kokkos/Eigen submodules THERE. With no local checkout we
# clone devel3 from GitHub. DOUBLE precision only; DP is the only variant.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_HINT="${SYNERGIA_SRC:-$HOME/src/synergia2}"
WORK="$REPO_ROOT/.builds/src/synergia2"          # our build source copy (submodules live here)
REPO_URL="${SYNERGIA_REPO:-https://github.com/fnalacceleratormodeling/synergia2.git}"
REF="${SYNERGIA_REF:-devel3}"

DEVICE="${SYNERGIA_DEVICE:-cpu}"                  # cpu (Kokkos OpenMP) | cuda (Kokkos CUDA + cuFFT)
JOBS="${SYNERGIA_BUILD_JOBS:-6}"                  # <= 6 parallel compile jobs (global build policy)

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: CONDA_PREFIX is not set -- run via 'pixi run --environment synergia[-cuda] build-synergia[-cuda]'." >&2
    exit 1
fi

# ------------------------------------------------------------------------------------------------ #
# 1) Materialize a build source copy with submodules (never touching the user's checkout).
# ------------------------------------------------------------------------------------------------ #
mkdir -p "$REPO_ROOT/.builds/src"
if [ ! -d "$WORK/.git" ]; then
    if [ -d "$SRC_HINT/.git" ]; then
        echo "Cloning local Synergia checkout $SRC_HINT -> $WORK (preserves its exact commit)"
        git clone "$SRC_HINT" "$WORK"
    else
        echo "No local Synergia checkout at $SRC_HINT; cloning $REPO_URL ($REF) -> $WORK"
        git clone --branch "$REF" "$REPO_URL" "$WORK"
    fi
fi
# Kokkos + Eigen are submodules and MUST be populated for USE_EXTERNAL_KOKKOS=OFF (add_subdirectory).
# (Catch2/openPMD/cereal are pulled by CMake FetchContent into the build dir at configure time.)
echo "Initializing Synergia submodules (Kokkos, Eigen) in $WORK"
git -C "$WORK" submodule update --init --recursive \
    src/synergia/utils/kokkos src/synergia/utils/eigen

# Human version label for plot footers (mirrors metadata._built_ref): the checked-out git describe.
git -C "$WORK" describe --tags --always --dirty > "$WORK/.bench_ref" 2>/dev/null || true

# ------------------------------------------------------------------------------------------------ #
# 2) Configure. Backend + GSV per device; portable microarch + (opt-in) fast-math via CXXFLAGS.
# ------------------------------------------------------------------------------------------------ #
# Perlmutter/portable microarch: CMake folds the CXXFLAGS/CFLAGS env into CMAKE_CXX_FLAGS at the
# FIRST configure. The activation env already sets -march=native; append BENCH_ARCH (last -march
# wins, so znver3 on Perlmutter; default native is a no-op). See machines/PERLMUTTER.md.
MARCH="${BENCH_ARCH:-native}"
EXTRA_FLAGS="-march=${MARCH} -mtune=${MARCH}"
# Fast-math is OFF by default. Synergia's own CMake attempt to add -ffast-math is a no-op bug
# (string(APPEND ${CMAKE_CXX_FLAGS} ...) passes the flag VALUE where a variable NAME is required),
# so the stock Release build is IEEE. BENCH_FASTMATH=1 forces it via -DEXTRA_CXX_FLAGS (honored at
# CMakeLists.txt:412). NOTE: unverified across Kokkos + the libFF SIMD maps -- the harness keeps
# Synergia IEEE-only (registry._supports_fastmath == False); this knob is for manual experiments.
EXTRA_CMAKE=()
case "${BENCH_FASTMATH:-0}" in
    1|on|ON|true) EXTRA_CMAKE+=("-DEXTRA_CXX_FLAGS=-ffast-math -fno-finite-math-only") ;;
    *) : ;;
esac
export CXXFLAGS="${CXXFLAGS:-} ${EXTRA_FLAGS}"
export CFLAGS="${CFLAGS:-} ${EXTRA_FLAGS}"

if [ "$DEVICE" = "cuda" ]; then
    KOKKOS_BACKEND="CUDA"
    GSV="${SYNERGIA_GSV:-DOUBLE}"                 # Synergia uses GSV=DOUBLE for the CUDA backend
    BUILD_DIR="$REPO_ROOT/.builds/synergia-cuda/build"
    # Kokkos CUDA needs the GPU compute capability. If SYNERGIA_CUDA_ARCH is set (e.g. AMPERE80 for
    # A100, AMPERE86 for a local A2000), pass it; otherwise Kokkos auto-detects from a visible GPU
    # at configure time (so this must configure on a node that has the target card).
    if [ -n "${SYNERGIA_CUDA_ARCH:-}" ]; then
        EXTRA_CMAKE+=("-DKokkos_ARCH_${SYNERGIA_CUDA_ARCH}=ON")
    fi
    echo "Synergia CUDA build (Kokkos CUDA + cuFFT), GSV=${GSV}, arch=${SYNERGIA_CUDA_ARCH:-auto}"
else
    KOKKOS_BACKEND="OpenMP"
    GSV="${SYNERGIA_GSV:-AVX}"                    # AVX SIMD path for the double libFF maps (x86)
    BUILD_DIR="$REPO_ROOT/.builds/synergia-cpu/build"
    echo "Synergia CPU build (Kokkos OpenMP), GSV=${GSV}, CXXFLAGS=${CXXFLAGS}"
fi

echo "Configuring Synergia -> install prefix $CONDA_PREFIX (build dir $BUILD_DIR)"
cmake -S "$WORK" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DENABLE_KOKKOS_BACKEND="$KOKKOS_BACKEND" \
    -DUSE_EXTERNAL_KOKKOS=OFF \
    -DUSE_OPENPMD_IO=OFF \
    -DBUILD_FD_SPACE_CHARGE_SOLVER=OFF \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DGSV="$GSV" \
    -DSIMPLE_TIMER=OFF \
    "${EXTRA_CMAKE[@]}"

# ------------------------------------------------------------------------------------------------ #
# 3) Build + install into the env.
# ------------------------------------------------------------------------------------------------ #
echo "Building Synergia with ${JOBS} jobs"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
echo "Installing Synergia into $CONDA_PREFIX"
cmake --install "$BUILD_DIR"

# ------------------------------------------------------------------------------------------------ #
# 4) Import smoke test (the exact API the run templates use).
# ------------------------------------------------------------------------------------------------ #
python -c "import synergia; \
from synergia.lattice import MadX_reader, Lattice, Lattice_element; \
from synergia.bunch import Bunch, Core_diagnostics, populate_6d; \
from synergia.foundation import Reference_particle, PCG_random_distribution; \
from synergia.simulation import Bunch_simulator, Propagator, Split_operator_stepper, Independent_stepper_elements; \
from synergia.collective import Space_charge_3d_open_hockney_options, Space_charge_2d_open_hockney_options; \
print('Synergia import OK')"
