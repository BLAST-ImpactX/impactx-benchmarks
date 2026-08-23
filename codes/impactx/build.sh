#!/usr/bin/env bash
# Build & install ImpactX into the active pixi environment.
#
# Each (device, precision, fast-math) combination gets its OWN source tree + build dir AND its own
# pixi env, so builds never share a CMake/AMReX tree and a pip install of one never overwrites
# another (same package name). Tag = <device>-<precision>-<fastmath>:
#   cpu-dp-ieee -> .builds/src/impactx-cpu-dp-ieee (env: impactx    -- the verification baseline)
#   cpu-dp-fm   -> .builds/src/impactx-cpu-dp-fm   (env: impactx-fm)
#   cpu-sp-*    -> .builds/src/impactx-cpu-sp-*    (envs: impactx-sp / impactx-sp-fm)
#   cuda-*      -> .builds/src/impactx-cuda-*      (envs: impactx-cuda-{dp,sp}[-fm])
# In a dedicated tree, -DImpactX_PRECISION alone controls fields AND particles.
# CXXFLAGS=-march=native -mtune=native come from the feature activation env; we additionally
# append -march=$BENCH_ARCH (default native; znver3 on Perlmutter -- last -march wins) and route
# fast-math via BENCH_FASTMATH (default 1/ON). Override the git ref with IMPACTX_REF, the device
# with IMPACTX_DEVICE.
set -eu -o pipefail

# Pin to the ImpactX 26.08 release tag. It already contains the (Chr)Quad perf optimization
# (PR #1521, merged 2026-06-25) that we previously pinned the PR head for. DP and SP MUST use the
# same ref (same physics + perf). Override with IMPACTX_REF (branch/tag/commit/pull-N-head all work).
REF="${IMPACTX_REF:-26.08}"
DEVICE="${IMPACTX_DEVICE:-cpu}"                 # cpu | cuda (later)
PRECISION="${IMPACTX_PRECISION:-DOUBLE}"
case "$PRECISION" in SINGLE) ptag=sp ;; *) ptag=dp ;; esac
case "$DEVICE"    in cuda)   COMPUTE=CUDA ;; *) COMPUTE=OMP ;; esac
# Fast-math is baked at compile time, so IEEE and fast-math each need their OWN tree+env. Both tags
# are EXPLICIT (-ieee / -fm) so neither can collide with a legacy bare "<device>-<prec>" tree and
# silently reuse its CMake cache (see the stale-cache guard below).
case "${BENCH_FASTMATH:-1}" in 0|off|OFF|false) FMBOOL=OFF; fmtag="-ieee" ;; *) FMBOOL=ON; fmtag="-fm" ;; esac
TAG="${DEVICE}-${ptag}${fmtag}"                 # e.g. cpu-dp-ieee, cpu-dp-fm, cuda-sp-fm
SRC=".builds/src/impactx-${TAG}"               # dedicated tree => dedicated build dir + AMReX
mkdir -p .builds/src

# Fetch+checkout works for branches, tags, commits AND PR heads (pull/N/head); -b clone does not.
if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 https://github.com/BLAST-ImpactX/impactx.git "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$REF"
git -C "$SRC" checkout -q FETCH_HEAD
HEAD_NOW="$(git -C "$SRC" rev-parse HEAD)"
echo "ImpactX ref=$REF -> $(git -C "$SRC" rev-parse --short HEAD)"
# DRY version label: record the exact human ref we built (e.g. "26.08"), so metadata/plots show
# THAT single source of truth instead of re-deriving it -- a shallow tag fetch never creates the
# local tag, so `git describe` would otherwise fall back to a bare SHA. (benchmarks/metadata.py
# reads .bench_ref first.) Works for tags, branches, and PR heads alike.
echo "$REF" > "$SRC/.bench_ref"
# Ref-change guard: a pin bump (e.g. 26.06 -> 26.08) can require a NEWER fetched AMReX, but a stale
# build/ reuses the OLD AMReX -> missing-header failures (e.g. AMReX_GpuParallelReduce.H). The
# flag-based guard below only catches CXXFLAGS changes, not a source-ref change -- so wipe build/ here
# whenever the checked-out commit differs from the last SUCCESSFUL build (stamp written at the end).
REFSTAMP="$SRC/.bench_built_ref"
if [ ! -f "$REFSTAMP" ] || [ "$(cat "$REFSTAMP" 2>/dev/null)" != "$HEAD_NOW" ]; then
    [ -d "$SRC/build" ] && { echo "  source ref changed -> wiping ${SRC}/build for a fresh AMReX fetch"; rm -rf "${SRC}/build"; }
fi

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
    # Pin CMAKE_CUDA_ARCHITECTURES up front, NOT only AMReX_CUDA_ARCH: CMake's initial "check for
    # working CUDA compiler" runs at enable_language(CUDA) -- BEFORE AMReX applies AMReX_CUDA_ARCH --
    # and otherwise probes the full default arch list including compute_50, which nvcc>=13 REJECTS
    # ("nvcc fatal: Unsupported gpu architecture 'compute_50'"; CUDA 13 dropped Maxwell/Pascal). That
    # breaks the whole configure on a CUDA-13 toolkit (local AND Perlmutter). Pinning it fixes the check.
    EXTRA_CMAKE=( "IMPACTX_CMAKE_ImpactX_SIMD=OFF" "IMPACTX_CMAKE_AMReX_CUDA_ARCH=${CUDA_ARCH}"
                  "IMPACTX_CMAKE_CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" )
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
# CMake seeds CMAKE_CXX_FLAGS from $CXXFLAGS only on the FIRST configure; afterwards the cache is
# sticky and the env is IGNORED. So a changed fast-math/arch would SILENTLY produce a stale build.
# Wipe the build dir whenever the cached flags disagree with what we want.
# NOTE: keep this -e/-o pipefail safe -- on a fresh tree $SRC/build does not exist and `find` would
# fail the pipeline and abort the script.
CACHE=""
if [ -d "$SRC/build" ]; then
    CACHE="$(find "$SRC/build" -name CMakeCache.txt 2>/dev/null | head -1 || true)"
fi
if [ -n "$CACHE" ]; then
    cached="$(grep -m1 '^CMAKE_CXX_FLAGS:STRING=' "$CACHE" | cut -d= -f2-)"
    case "$cached" in *-ffast-math*) had_fm=ON ;; *) had_fm=OFF ;; esac
    case "$cached" in *"-march=${MARCH}"*) had_arch=yes ;; *) had_arch=no ;; esac
    if [ "$had_fm" != "$FMBOOL" ] || [ "$had_arch" != "yes" ]; then
        echo "build flags changed (fast-math ${had_fm}->${FMBOOL}, arch ${MARCH}) -> wiping ${SRC}/build"
        rm -rf "${SRC}/build"
    fi
fi
# ImpactX pip build reads these env vars (see impactx.readthedocs.io install/cmake)
env IMPACTX_COMPUTE="$COMPUTE" \
    IMPACTX_PRECISION="$PRECISION" \
    IMPACTX_MPI=ON \
    IMPACTX_FFT=ON \
    IMPACTX_CMAKE_ImpactX_OPENPMD=OFF \
    "${EXTRA_CMAKE[@]}" \
    python -m pip install -v --force-reinstall --no-deps "$SRC"

python -c "import impactx; print('impactx', impactx.__version__)"
echo "$HEAD_NOW" > "$SRC/.bench_built_ref"   # record the built commit for the ref-change guard above
