#!/usr/bin/env bash
# Build pyAT (accelerator-toolbox) from source so the C integrators are compiled
# with -march=native -mtune=native (CFLAGS) and OpenMP (OPENMP=1) -- both come from
# the feature activation env. Override the version with PYAT_VERSION.
set -eu -o pipefail

SPEC="accelerator-toolbox${PYAT_VERSION:+==${PYAT_VERSION}}"

# Perlmutter/portable microarch + fast-math for the C integrators: append to CFLAGS after the
# activation's -march=native (last -march wins → BENCH_ARCH=znver3 on Perlmutter, default native
# is a no-op); add -ffast-math when BENCH_FASTMATH=1 (the default). See machines/PERLMUTTER.md.
MARCH="${BENCH_ARCH:-native}"
EXTRA_CFLAGS="-march=${MARCH} -mtune=${MARCH}"
case "${BENCH_FASTMATH:-1}" in 0|off|OFF|false) : ;; *) EXTRA_CFLAGS="${EXTRA_CFLAGS} -ffast-math" ;; esac
export CFLAGS="${CFLAGS:-} ${EXTRA_CFLAGS}"
echo "pyAT CFLAGS=${CFLAGS}"

# --no-binary ONLY for accelerator-toolbox so its C integrators compile locally
# with CFLAGS / OPENMP; numpy/scipy/h5py come from conda (--no-deps keeps them).
python -m pip install --upgrade pip
python -m pip install \
    --no-binary accelerator-toolbox \
    --no-deps --force-reinstall \
    "${SPEC}"

python -c "import at; print('accelerator-toolbox', at.__version__)"
