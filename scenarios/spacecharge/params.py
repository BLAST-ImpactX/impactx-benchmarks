"""3D space-charge scenario: one kick on a cold cigar-shaped bunch, checked against the exact answer.

A Gaussian electron bunch, 1 mm RMS in x, y and z in the lab frame, at 250 MeV (gamma = 490), so it is
490:1 in the rest frame where the field is solved. A rest-frame mesh with that aspect ratio is what
separates the kernels: a point-sampled 1/r kernel is only accurate when cells are close to cubic.

Three choices make the result depend on the space-charge solver and little else:

* **Cold:** the initial momentum spread is 1% of the RMS kick, so the kick is the signal.
* **Thin:** over the drift the kick moves a particle by 1e-3 of the beam size. Codes place their one
  kick differently (entrance, centre or exit), which this makes irrelevant, while on a beam that grows
  along the drift they would kick different beams.
* **Exact reference:** a Gaussian bunch has a closed-form field (``analytic.py``), so no code is the
  reference, and the check also catches an error shared by all codes.

The validated momentum-space observables cannot be fooled by the beam size, which barely changes here.
``<x px>`` and ``<y py>`` are the linear kick and are robust to shot noise; the RMS spreads and the
emittances add the nonlinear part; ``sigma_pe`` is the longitudinal kick.

The mesh half-width applies to the codes with a configurable fixed box (Cheetah, Xsuite, HELIX), which
drop or clamp particles outside it. At +/-3 sigma the dropped tail charge moved the emittance of
Cheetah and Xsuite by about 4%; at +/-5 sigma the linear kick of the full 3D IGF codes agrees to about
0.5% at N=1M.

The sweep starts at 10k particles: at 1k on this 64^3 mesh, shot noise alone moved the emittance of
ImpactX by +21% and of Cheetah by +15%.
"""

from __future__ import annotations

from functools import lru_cache

from scenarios.spacecharge import analytic

MASS_MEV = 0.51099895069
KIN_ENERGY_MEV = 250.0
BUNCH_CHARGE_C = 1.0e-9

SIGMA_X = 1.0e-3  # m
SIGMA_Y = 1.0e-3  # m
SIGMA_T = 1.0e-3  # m

DRIFT_LENGTH = 6.4  # m; the kick moves a particle by 1e-3 of SIGMA_X over this length
COLD_FRACTION = 0.01  # initial RMS momentum spread / RMS kick
N_CELL = 64  # space-charge mesh per dimension (64^3)
GRID_EXTENT_SIGMA = 5.0  # mesh half-width in beam sigmas (codes with a configurable fixed box)

_KICK = analytic.kick(q=BUNCH_CHARGE_C, ekin_mev=KIN_ENERGY_MEV, mc2_mev=MASS_MEV,
                      sx=SIGMA_X, sy=SIGMA_Y, sz_lab=SIGMA_T, length=DRIFT_LENGTH)
_SIGMA_PX = COLD_FRACTION * _KICK["fx"]
_SIGMA_PY = COLD_FRACTION * _KICK["fy"]
_SIGMA_P = COLD_FRACTION * _KICK["fe"]

PARAMS = {
    "mass_MeV": MASS_MEV,
    "kin_energy_MeV": KIN_ENERGY_MEV,
    "bunch_charge_C": BUNCH_CHARGE_C,
    # Twiss form of an upright beam: emit = sigma * sigma_p and beta = sigma / sigma_p. The large beta
    # (about 6e5 m) is what "cold" means here, not a focusing lattice.
    "emit_x": SIGMA_X * _SIGMA_PX,
    "emit_y": SIGMA_Y * _SIGMA_PY,
    "beta_x": SIGMA_X / _SIGMA_PX,
    "beta_y": SIGMA_Y / _SIGMA_PY,
    "alpha_x": 0.0,
    "alpha_y": 0.0,
    "sigma_t": SIGMA_T,
    "sigma_p": _SIGMA_P,
    "drift_length": DRIFT_LENGTH,
    "n_cell": N_CELL,
    "grid_extent_sigma": GRID_EXTENT_SIGMA,
}


@lru_cache(maxsize=1)
def _reference() -> dict:
    sigma_px = (_SIGMA_PX**2 + _KICK["fx"] ** 2) ** 0.5
    sigma_py = (_SIGMA_PY**2 + _KICK["fy"] ** 2) ** 0.5
    return {
        "sigma_x": SIGMA_X,
        "sigma_y": SIGMA_Y,
        "sigma_t": SIGMA_T,
        "xpx": _KICK["xpx"],
        "ypy": _KICK["ypy"],
        "sigma_px": sigma_px,
        "sigma_py": sigma_py,
        "sigma_pe": (_SIGMA_P**2 + _KICK["fe"] ** 2) ** 0.5,
        "emit_x": (SIGMA_X**2 * sigma_px**2 - _KICK["xpx"] ** 2) ** 0.5,
        "emit_y": (SIGMA_Y**2 * sigma_py**2 - _KICK["ypy"] ** 2) ** 0.5,
    }


def analytic_observables(npart: int) -> dict:
    """Exact post-kick observables. They do not depend on ``npart``: the reference has no sampling."""
    return dict(_reference())
