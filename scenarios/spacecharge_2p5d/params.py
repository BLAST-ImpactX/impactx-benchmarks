"""2.5D space-charge scenario: a charged Gaussian bunch drifting under its own self-field.

A single drift with one space-charge kick, validated against the 2.5D ImpactX result. This beam is
the one the 3D benchmark used before it moved to a cold thin-kick test against an exact reference.

Known limitation: this beam has the weakness that test was built to fix. The predicted space-charge
gain in sigma_x is below the sampling noise, so this check mostly confirms the beam setup rather than
the 2.5D solver. See BLAST-ImpactX/impactx-benchmarks#1.
"""

from __future__ import annotations

# electron beam
MASS_MEV = 0.51099895069
KIN_ENERGY_MEV = 250.0
BUNCH_CHARGE_C = 1.0e-9

# round Gaussian bunch (geometric emittance), with finite bunch length
EMIT_X = 1.0e-7  # m.rad
EMIT_Y = 1.0e-7  # m.rad
BETA_X = 1.0  # m
BETA_Y = 1.0  # m
ALPHA_X = 0.0
ALPHA_Y = 0.0
SIGMA_T = 1.0e-3  # m
SIGMA_P = 1.0e-4  # dimensionless

DRIFT_LENGTH = 6.0  # m
N_CELL = 64  # space-charge mesh per dimension (64^3)
GRID_EXTENT_SIGMA = 3.0  # mesh half-width in beam sigmas

PARAMS = {
    "mass_MeV": MASS_MEV,
    "kin_energy_MeV": KIN_ENERGY_MEV,
    "bunch_charge_C": BUNCH_CHARGE_C,
    "emit_x": EMIT_X,
    "emit_y": EMIT_Y,
    "beta_x": BETA_X,
    "beta_y": BETA_Y,
    "alpha_x": ALPHA_X,
    "alpha_y": ALPHA_Y,
    "sigma_t": SIGMA_T,
    "sigma_p": SIGMA_P,
    "drift_length": DRIFT_LENGTH,
    "n_cell": N_CELL,
    "grid_extent_sigma": GRID_EXTENT_SIGMA,
}

# No analytic reference for this beam; ImpactX 2.5D is the reference code.
