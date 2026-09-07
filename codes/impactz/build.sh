#!/bin/bash
# Source-build IMPACT-Z (LBNL Fortran/MPI PIC space-charge code; ImpactX's predecessor).
# Mirrors codes/elegant/build.sh: build, then stage the binary at a stable path the driver
# launches (codes/impactz/bin/ImpactZexe-mpi).
#
# IMPACT-Z is CMake + a Fortran compiler + MPI; the FFT is the BUNDLED FFTPACK 5.1 (no external
# FFTW needed). Double precision only, MPI-only (no OpenMP), CPU-only (no GPU). Run from the repo
# root inside the `impactz` pixi env (`pixi run build-impactz`), which supplies gfortran + mpich.
set -eu -o pipefail

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

# Configure + build the MPI executable (bundled FFTPACK). find_package(MPI) locates the conda
# mpich; the compiler is the conda gfortran on PATH. CMake adds -fallow-argument-mismatch for gcc>=10.
BUILD="$SRC/build-mpi"
cmake -S "$SRC/src" -B "$BUILD" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_MPI=ON \
      -DUSE_FFTW=OFF
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
echo "== staged $DEST/ImpactZexe-mpi"
