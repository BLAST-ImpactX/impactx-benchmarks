#!/usr/bin/env bash
# Source-build Elegant (+ SDDS toolkit sibling) into the active pixi env.
#
# Elegant's GNU-Make build assumes the APS/EPICS system layout: it discovers libraries by scanning a
# hardcoded LIB_DIRS list (NOT $CONDA_PREFIX) and links most of them by ABSOLUTE PATH, while LAPACK
# links with -l. We drive it onto the conda toolchain purely via make-variable overrides (no Makefile
# edits): point LIB_DIRS at $CONDA_PREFIX/lib (so the include dirs, derived as lib->include, resolve
# to $CONDA_PREFIX/include too), give LAPACK explicit conda paths, and pass conda's compilers + MPI +
# nvcc. LD_RUN_PATH embeds the rpath so the abs-path-linked .so are found at runtime, without fighting
# the Makefile's own `LDFLAGS += ...`.
#
# The elegant top-level `make all` also builds the required SDDS subdirs in place (they are in its
# DIRS list), so a single invocation builds SDDS + physics + src (elegant, Pelegant) + the GPU
# variants. Pelegant is built when MPI_CC/MPI_CCC are set; gpu-elegant/gpu-Pelegant when HAVE_CUDA=1.
#
# Env knobs:  ELEGANT_DEVICE=cpu|cuda (default cpu), ELEGANT_CUDA_ARCH (default sm_86),
#             BUILD_NPROC (default 4). Elegant is DP-only and has no fast-math axis.
set -eu -o pipefail

ROOT="$PWD"
SRCDIR="$ROOT/.builds/src"
DEVICE="${ELEGANT_DEVICE:-cpu}"
N="${BUILD_NPROC:-4}"
mkdir -p "$SRCDIR"

# Source trees, consistent with the other source codes (pyorbit/bmad/scibmad): an env hint
# (ELEGANT_SRC) points at a local checkout, else we clone fresh under .builds/src. SDDS is required
# by elegant's Makefile as the SIBLING ../SDDS, so it always lives next to the chosen elegant tree
# (cloned there if absent) -- this stays self-consistent whether elegant is local or freshly cloned.
ELE="${ELEGANT_SRC:-$SRCDIR/elegant}"
SDDS="$(dirname "$ELE")/SDDS"
[ -d "$ELE/.git" ]  || git clone --depth 1 "${ELEGANT_REPO_URL:-https://github.com/rtsoliday/elegant.git}" "$ELE"
[ -d "$SDDS/.git" ] || git clone --depth 1 "${SDDS_REPO_URL:-https://github.com/rtsoliday/SDDS.git}" "$SDDS"
echo "Elegant $(git -C "$ELE" rev-parse --short HEAD) / SDDS $(git -C "$SDDS" rev-parse --short HEAD)  device=$DEVICE"

# Toolchain: do NOT set CC/CCC/F77 on the make command line -- a command-line make variable is
# immutable and would override Makefile.mpi's per-target `CC = $(MPI_CC)`, so Pelegant would link
# with the plain compiler and miss -lmpi. Instead we rely on the conda env providing bare
# `gcc`/`g++`/`gfortran` (the serial build uses those) and pass only MPI_CC/MPI_CCC (the wrappers,
# used for the MPI targets). The conda compiler + MPI wrappers already inject -I/-L/-rpath for
# $CONDA_PREFIX; belt-and-suspenders LD_RUN_PATH covers the abs-path-linked .so (gsl/lapack/fftw).
export LD_RUN_PATH="$CONDA_PREFIX/lib${LD_RUN_PATH:+:$LD_RUN_PATH}"
export MPICH_CC="${CC:-gcc}" MPICH_CXX="${CXX:-g++}" MPICH_FC="${FC:-gfortran}"

MK=(
  "LIB_DIRS=$CONDA_PREFIX/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /lib64 /usr/lib /lib"
  "LAPACK_INCLUDE=-I$CONDA_PREFIX/include"
  "LAPACK_LIB=$CONDA_PREFIX/lib/liblapack.so $CONDA_PREFIX/lib/libblas.so"
  "MPI_CC=$CONDA_PREFIX/bin/mpicc" "MPI_CCC=$CONDA_PREFIX/bin/mpicxx"
)
if [ "$DEVICE" = "cuda" ]; then
    NVCC_BIN="${NVCC:-$CONDA_PREFIX/bin/nvcc}"
    MK+=( "HAVE_CUDA=1" "NVCC=$NVCC_BIN" "CUDA_ARCH=${ELEGANT_CUDA_ARCH:-sm_86}" )
    echo "Elegant CUDA build: NVCC=$NVCC_BIN arch=${ELEGANT_CUDA_ARCH:-sm_86}"
else
    MK+=( "CUDA_AUTO=0" )   # force CPU-only even if a system nvcc is on PATH
fi

make -C "$ELE" -j"$N" all "${MK[@]}"

# Stage the freshly built binaries at stable, per-device paths the runner/driver launch.
BIN_CPU="$ELE/bin/Linux-x86_64"
BIN_GPU="$ELE/bin/Linux-x86_64-gpu"
DEST="$ROOT/codes/elegant/bin"
mkdir -p "$DEST"
# Elegant needs the RPN definitions file at runtime (RPN_DEFNS env var). Stage it to a stable path
# so the driver does not depend on the disposable .builds/ tree.
cp -f "$SDDS/defns.rpn" "$DEST/defns.rpn" && echo "  staged defns.rpn"
stage() {  # stage <srcfile> <destname>  (only if built)
    if [ -x "$1" ]; then cp -f "$1" "$DEST/$2"; echo "  staged $2"; fi
}
if [ "$DEVICE" = "cuda" ]; then
    stage "$BIN_GPU/gpu-elegant"  gpu-elegant
    stage "$BIN_GPU/gpu-Pelegant" gpu-Pelegant
else
    stage "$BIN_CPU/elegant"  elegant
    stage "$BIN_CPU/Pelegant" Pelegant
fi

# Smoke-test: the staged serial binary must at least print its version banner.
SMOKE="$DEST/elegant"; [ "$DEVICE" = "cuda" ] && SMOKE="$DEST/gpu-elegant"
if [ -x "$SMOKE" ]; then
    echo "=== version check: $(basename "$SMOKE") ==="
    "$SMOKE" -v 2>&1 | head -3 || true
else
    echo "ERROR: expected binary $SMOKE was not produced" >&2
    exit 1
fi
echo "elegant build OK (device=$DEVICE)"
