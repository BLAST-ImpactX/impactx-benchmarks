#!/usr/bin/env bash
# Native source-build of Bmad (forest/PTC + sim_utils + bmad libs only -- NO Tao/pytao),
# then compile our standalone Fortran driver (codes/bmad/driver) against libbmad.
# Uses the local ~/src/bmad-ecosystem checkout if present, else a shallow clone (CI).
# NOTE: Bmad's dist scripts are NOT `set -e`-safe (they reference unset vars and return
# non-zero), so we run without -e and gate success on the produced driver binary.
set -u -o pipefail

REPO_ROOT="$PWD"
DRIVER_SRC="$REPO_ROOT/codes/bmad/driver"
SRC="${BMAD_SRC:-/home/axel/src/bmad-ecosystem}"
if [ ! -d "$SRC/util" ]; then
    SRC="$REPO_ROOT/.builds/src/bmad-ecosystem"
    mkdir -p "$REPO_ROOT/.builds/src"
    if [ ! -d "$SRC/.git" ]; then
        git clone --depth 1 "${BMAD_REPO:-https://github.com/bmad-sim/bmad-ecosystem.git}" "$SRC" \
            || { echo "ERROR: bmad clone failed"; exit 1; }
    fi
fi
echo "Bmad source: $SRC"

# Build preferences: user_prefs is sourced AFTER dist_prefs by dist_source_me, so it wins.
cat > "$SRC/util/user_prefs" <<EOF
export DIST_F90_REQUEST="gfortran"
export ACC_PLOT_PACKAGE="pgplot"        # conda-provided; no plplot source build
export ACC_ENABLE_OPENMP="Y"
export ACC_ENABLE_MPI="N"
export ACC_ENABLE_GFORTRAN_OPTIMIZATION="Y"
export ACC_ENABLE_SHARED="Y"
export ACC_ENABLE_SHARED_ONLY="Y"
export ACC_ENABLE_FPIC="Y"
export ACC_SET_GMAKE_JOBS="${BUILD_NPROC:-6}"
export ACC_CONDA_BUILD="Y"
export ACC_CONDA_PATH="${CONDA_PREFIX}"
EOF

cd "$SRC" || { echo "ERROR: cannot cd $SRC"; exit 1; }
set +u  # bmad scripts reference unset vars; dist_source_me is not -e/-u safe
source util/dist_source_me

# Perlmutter/portable microarch + fast-math for gfortran. Set AFTER dist_source_me (which may reset
# FFLAGS) so the ACC cmake configure picks them up; -O2 still comes from ACC_ENABLE_GFORTRAN_OPTIMIZATION.
# BENCH_ARCH default native (no-op locally; znver3 on Perlmutter); -ffast-math when BENCH_FASTMATH=1.
# NOTE: verify the ACC build honors FFLAGS/FCFLAGS (Master.cmake) -- if not, these are inert.
MARCH="${BENCH_ARCH:-native}"
EXTRA_F="-march=${MARCH} -mtune=${MARCH}"
case "${BENCH_FASTMATH:-1}" in 0|off|OFF|false) : ;; *) EXTRA_F="${EXTRA_F} -ffast-math" ;; esac
export FFLAGS="${FFLAGS:-} ${EXTRA_F}"
export FCFLAGS="${FCFLAGS:-} ${EXTRA_F}"
echo "Bmad FFLAGS=${FFLAGS}"

# Build only the libraries the driver needs, in dependency order (skip tao/bsim/util_programs).
for pkg in forest sim_utils bmad; do
    echo "=== building $pkg ==="
    ( cd "$pkg" && mk ) || echo "WARNING: mk for $pkg returned non-zero"
done

# Build our standalone driver against the freshly built libbmad.
echo "=== building bmad_driver ==="
( cd "$DRIVER_SRC" && mk ) || echo "WARNING: mk for driver returned non-zero"
set -u

# Locate + stage the driver binary at a stable path the harness launches.
# `mk` builds to <driver>/../production/bin (i.e. codes/bmad/production/bin/bmad_driver).
BIN="$(find "$REPO_ROOT/codes/bmad" -name bmad_driver -type f -perm -u+x 2>/dev/null \
       | grep -v '/bmad_driver$' | head -1)"
[ -n "$BIN" ] || BIN="$(find "$REPO_ROOT/codes/bmad/production" "$DRIVER_SRC" -name bmad_driver -type f 2>/dev/null | head -1)"
[ -n "$BIN" ] || { echo "ERROR: bmad_driver binary not found after build"; exit 1; }
cp -f "$BIN" "$REPO_ROOT/codes/bmad/bmad_driver"
echo "Installed driver: codes/bmad/bmad_driver"
"$REPO_ROOT/codes/bmad/bmad_driver" --version 2>/dev/null || true
echo "bmad build OK"
