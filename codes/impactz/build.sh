#!/bin/bash
# Source-build IMPACT-Z (LBNL Fortran/MPI PIC space-charge code; ImpactX's predecessor).
# Mirrors codes/elegant/build.sh: build, then stage the binary at a stable path the driver
# launches (codes/impactz/bin/ImpactZexe-mpi).
#
# IMPACT-Z is CMake + a Fortran compiler + MPI + FFTW3. Double precision only, MPI-only (no OpenMP),
# CPU-only (no GPU). Run from the repo root inside the `impactz` pixi env (`pixi run build-impactz`),
# which supplies gfortran, mpich and fftw.
#
# FFT backend: FFTW, not upstream's default bundled FFTPACK 5.1. FFTPACK allocates work arrays and
# recomputes its twiddle factors for every 1D transform, while FFTW caches a plan per size. Both run in
# IMPACT-Z's own 3D pipeline of 1D transforms and MPI transposes, so this is not a distributed FFTW.
set -eu -o pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: CONDA_PREFIX is not set -- run via 'pixi run build-impactz'." >&2
  exit 1
fi

ROOT="$PWD"
SRCDIR="$ROOT/.builds/src"
N="${BUILD_NPROC:-4}"                       # repo policy: <= 6 cores
REPO="${IMPACTZ_REPO:-https://github.com/impact-lbl/IMPACT-Z.git}"
DEST="$ROOT/codes/impactz/bin"

# Source tree: prefer an explicit override, then the user's canonical checkout, else clone.
if [ -n "${IMPACTZ_SRC:-}" ]; then
  SRC="$IMPACTZ_SRC"
elif [ -d "$HOME/src/IMPACT-Z/src" ]; then
  SRC="$HOME/src/IMPACT-Z"
else
  SRC="$SRCDIR/IMPACT-Z"
  if [ ! -d "$SRC/src" ]; then
    mkdir -p "$SRCDIR"
    git clone "$REPO" "$SRC"
  fi
fi
echo "== IMPACT-Z source: $SRC"

# Configure + build the MPI executable. find_package(MPI) locates the conda mpich; the compiler is the
# conda gfortran on PATH. CMake adds -fallow-argument-mismatch for gcc>=10.
#   * --fresh: an existing build dir's cache otherwise keeps the compiler, MPI and FFTW it first found.
#   * PKG_CONFIG_PATH: IMPACT-Z's CMake asks pkg-config first, so a system fftw3.pc would otherwise win.
#   * microarch: same as the other source builds, native or BENCH_ARCH (znver3 on Perlmutter; see
#     machines/PERLMUTTER.md).
MARCH="${BENCH_ARCH:-native}"
BUILD="$SRC/build-mpi"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cmake --fresh -S "$SRC/src" -B "$BUILD" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
      -DCMAKE_Fortran_FLAGS="${FFLAGS:-} -march=${MARCH} -mtune=${MARCH}" \
      -DUSE_MPI=ON \
      -DUSE_FFTW=ON
cmake --build "$BUILD" -j "$N"

# Stage the binary at the stable path the driver hard-codes (relative to the repo root).
mkdir -p "$DEST"
cp -f "$BUILD/ImpactZexe-mpi" "$DEST/ImpactZexe-mpi"

# Stamp the human-facing ref for plot labels / metadata (DRY: what the build actually used).
( cd "$SRC" && git describe --tags --always --dirty 2>/dev/null || echo "unknown" ) > "$SRC/.bench_ref"
echo "== built ref: $(cat "$SRC/.bench_ref")"

# Smoke test: binary present, all shared libs resolve.
if ldd "$DEST/ImpactZexe-mpi" 2>/dev/null | grep -q "not found"; then
  echo "ERROR: unresolved shared libraries in $DEST/ImpactZexe-mpi" >&2
  ldd "$DEST/ImpactZexe-mpi" | grep "not found" >&2
  exit 1
fi

# The backend must be the env's FFTW. A checkout older than upstream's FFTW support (IMPACT-Z #19)
# ignores USE_FFTW and would silently build FFTPACK, so check the binary rather than the configure.
linked="$(ldd "$DEST/ImpactZexe-mpi")"
if ! grep -qF "$CONDA_PREFIX/lib/libfftw3" <<< "$linked"; then
  echo "ERROR: $DEST/ImpactZexe-mpi does not link $CONDA_PREFIX/lib/libfftw3*." >&2
  echo "       Is $SRC older than IMPACT-Z #19?" >&2
  exit 1
fi
echo "== staged $DEST/ImpactZexe-mpi"
