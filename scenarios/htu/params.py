"""HTU scenario: the BELLA HTU laser-plasma-accelerator beamline (tracking only).

Ported from the NAPAC25 benchmark. The lattice (PMQ triplet, chicane, EMQ triplet,
undulator FODO cells) is built per code by ``scenarios/htu/htu_lattice.py``. This is a
linear-ish tracking benchmark (no space charge); there is no analytic reference, so
validation uses the reference code (ImpactX).
"""

from __future__ import annotations

import math

from scipy.constants import c, e, m_e

TOTAL_ENERGY_EV = 100.0e6
MASS_EV = (m_e * c**2) / e
KIN_ENERGY_EV = TOTAL_ENERGY_EV - MASS_EV

_GAMMA = TOTAL_ENERGY_EV / MASS_EV
_BG = math.sqrt(_GAMMA**2 - 1.0)  # beta*gamma

# Twiss + normalized emittance (1.5 mm.mrad) -> geometric emittance
NORM_EMIT = 1.5e-6  # m.rad (normalized)
EMIT_X = NORM_EMIT / _BG
EMIT_Y = NORM_EMIT / _BG
BETA_X = 0.002  # m
BETA_Y = 0.002  # m
ALPHA_X = 0.0
ALPHA_Y = 0.0
SIGMA_T = 1.0e-6  # m
SIGMA_P = 2.5e-2  # dimensionless
MU_P = 1.0e-2  # relative momentum offset

BUNCH_CHARGE_C = 25.0e-12

PARAMS = {
    "mass_MeV": MASS_EV * 1e-6,
    "mass_eV": MASS_EV,
    "kin_energy_MeV": KIN_ENERGY_EV * 1e-6,
    "kin_energy_eV": KIN_ENERGY_EV,
    "total_energy_eV": TOTAL_ENERGY_EV,
    "emit_x": EMIT_X,
    "emit_y": EMIT_Y,
    "beta_x": BETA_X,
    "beta_y": BETA_Y,
    "alpha_x": ALPHA_X,
    "alpha_y": ALPHA_Y,
    "sigma_t": SIGMA_T,
    "sigma_p": SIGMA_P,
    "mu_p": MU_P,
    "bunch_charge_C": BUNCH_CHARGE_C,
}

# No analytic reference for the full HTU line; ImpactX is the reference code.
