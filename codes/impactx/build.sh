#!/usr/bin/env bash
# Build & install ImpactX into the active pixi environment.
#
# Each (device, precision) combination gets its OWN source tree + build dir AND its own pixi
# env, so builds never share a CMake/AMReX tree and a pip install of one never overwrites
# another (same package name). Tag = <device>-<precision>:
#   cpu-dp  -> .builds/src/impactx-cpu-dp   (env: impactx)
#   cpu-sp  -> .builds/src/impactx-cpu-sp   (env: impactx-sp)
#   cuda-*  -> .builds/src/impactx-cuda-*    (added later)
# In a dedicated tree, -DImpactX_PRECISION alone controls fields AND particles.
# CXXFLAGS=-march=native -mtune=native come from the feature activation env; we additionally
# append -march=$BENCH_ARCH (default native; znver3 on Perlmutter -- last -march wins) and route
# fast-math via BENCH_FASTMATH (default 1/ON). Override the git ref with IMPACTX_REF, the device
# with IMPACTX_DEVICE.
set -eu -o pipefail

# TEMPORARY: pin to PR #1521 (perf-quad-branch-free, ed88d69) for the perf-quad optimization.
# Revert to "development" once the PR merges. DP and SP MUST use the same ref (same physics+perf).
REF="${IMPACTX_REF:-pull/1521/head}"
DEVICE="${IMPACTX_DEVICE:-cpu}"                 # cpu | cuda (later)
PRECISION="${IMPACTX_PRECISION:-DOUBLE}"
case "$PRECISION" in SINGLE) ptag=sp ;; *) ptag=dp ;; esac
case "$DEVICE"    in cuda)   COMPUTE=CUDA ;; *) COMPUTE=OMP ;; esac
# Fast-math is baked at compile time, so an IEEE (BENCH_FASTMATH=0) build needs its OWN tree+env
# (tag suffix "-ieee") -- this is the DP verification baseline (impactx-ref). Default (ON) keeps
# the plain tag, so existing fast-math builds are unaffected.
case "${BENCH_FASTMATH:-1}" in 0|off|OFF|false) FMBOOL=OFF; fmtag="-ieee" ;; *) FMBOOL=ON; fmtag="" ;; esac
TAG="${DEVICE}-${ptag}${fmtag}"                 # cpu-dp, cpu-sp, cuda-dp, cuda-sp (+ -ieee if OFF)
SRC=".builds/src/impactx-${TAG}"               # dedicated tree => dedicated build dir + AMReX
mkdir -p .builds/src

# Fetch+checkout works for branches, tags, commits AND PR heads (pull/N/head); -b clone does not.
if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 https://github.com/BLAST-ImpactX/impactx.git "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$REF"
git -C "$SRC" checkout -q FETCH_HEAD
echo "ImpactX ref=$REF -> $(git -C "$SRC" rev-parse --short HEAD)"

export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_NPROC:-6}"
# ccache speeds up repeat CI builds; CCACHE_BASEDIR normalizes the per-tree absolute paths so
# the (identical) AMReX objects are shared across the separate cpu-dp/cpu-sp/... build trees.
export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-ccache}"
export CCACHE_BASEDIR="$PWD"

python -m pip install --upgrade pip
echo "Building ImpactX device=$DEVICE precision=$PRECISION (tag=$TAG, src=$SRC)"
# SIMD is a CPU-only AMReX backend; disable it for CUDA. For CUDA, build for the GPU's
# compute capability. CUDA_ARCH is auto-detected from the GPU (so the same script is correct
# on a local card AND on Perlmutter); an explicit CUDA_ARCH env always wins.
detect_cuda_arch() {
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d '.[:space:]')"
    # "8.6" -> "86" (A2000/RTX30xx), "8.0" -> "80" (A100/Perlmutter). Fallback 80.
    [ -n "$cc" ] && echo "$cc" || echo "80"
}
if [ "$DEVICE" = "cuda" ]; then
    CUDA_ARCH="${CUDA_ARCH:-$(detect_cuda_arch)}"
    echo "ImpactX CUDA arch (AMReX_CUDA_ARCH) = $CUDA_ARCH"
    EXTRA_CMAKE=( "IMPACTX_CMAKE_ImpactX_SIMD=OFF" "IMPACTX_CMAKE_AMReX_CUDA_ARCH=${CUDA_ARCH}" )
else
    EXTRA_CMAKE=( "IMPACTX_CMAKE_ImpactX_SIMD=ON" )
fi
# Fast-math (FMBOOL from BENCH_FASTMATH, computed above with the tag).
# HOST/CPU: we do NOT use AMReX_FASTMATH -- it adds -ffast-math *without* -fno-finite-math-only, and
# AMReX quad/bend maps call sinh/cosh whose __*_finite symbols modern glibc (>=2.31) removed -> the
# .so fails to load (undefined symbol). Instead drive host fast-math through CXXFLAGS/CFLAGS with the
# -fno-finite-math-only guard (last -march also wins over the activation's -march=native).
# DEVICE/CUDA: AMReX_CUDA_FASTMATH -> nvcc --use_fast_math (no glibc _finite issue on the GPU).
[ "$DEVICE" = "cuda" ] && EXTRA_CMAKE+=( "IMPACTX_CMAKE_AMReX_CUDA_FASTMATH=${FMBOOL}" )
FMFLAGS=""
[ "$FMBOOL" = "ON" ] && FMFLAGS="-ffast-math -fno-finite-math-only"
echo "ImpactX fast-math = ${FMBOOL} (tag=${TAG}); host FMFLAGS='${FMFLAGS}'"
MARCH="${BENCH_ARCH:-native}"
export CXXFLAGS="${CXXFLAGS:-} -march=${MARCH} -mtune=${MARCH} ${FMFLAGS}"
export CFLAGS="${CFLAGS:-} -march=${MARCH} -mtune=${MARCH} ${FMFLAGS}"
# ImpactX pip build reads these env vars (see impactx.readthedocs.io install/cmake)
env IMPACTX_COMPUTE="$COMPUTE" \
    IMPACTX_PRECISION="$PRECISION" \
    IMPACTX_MPI=ON \
    IMPACTX_FFT=ON \
    IMPACTX_CMAKE_ImpactX_OPENPMD=OFF \
    "${EXTRA_CMAKE[@]}" \
    python -m pip install -v --force-reinstall --no-deps "$SRC"

python -c "import impactx; print('impactx', impactx.__version__)"
