#!/usr/bin/env bash
# Build/install HELIX (the `linac_gen` package) into the active pixi env.
#
# We only use HELIX's pure-PyTorch space-charge path (torch.fft IGF), so the optional C++ pybind
# kernels are NOT needed -- they degrade gracefully without a compiler (LINAC_GEN_REQUIRE_CPP unset).
# torch / numpy / scipy / h5py / cma come from the pixi env (conda); pip only registers linac_gen.
#
# Source resolution (consistent with the other from-source codes: env hint, then git-clone fallback):
#   $HELIX_SRC (default ~/src/HELIX)  ->  else a shallow clone into .builds/src/HELIX.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${HELIX_SRC:-$HOME/src/HELIX}"
if [ ! -f "$SRC/pyproject.toml" ] || [ ! -d "$SRC/linac_gen" ]; then
    SRC="$REPO_ROOT/.builds/src/HELIX"
    if [ ! -d "$SRC/linac_gen" ]; then
        echo "HELIX source not found via HELIX_SRC; cloning into $SRC"
        mkdir -p "$(dirname "$SRC")"
        git clone --depth 1 https://github.com/Accel-Toolkit/HELIX "$SRC"
    fi
fi

echo "Installing HELIX (linac_gen) from $SRC"
python -m pip install --upgrade pip
# --no-deps: torch/numpy/scipy/h5py/cma are provided by the pixi env (conda). This avoids pip pulling
# a second (possibly CPU-only or mismatched-CUDA) torch wheel over the conda one.
python -m pip install --no-deps "$SRC"
python -c "import linac_gen, torch; from linac_gen.pic.torch.sc_kick import torch_pic_sc_kick; \
print('HELIX import OK; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
