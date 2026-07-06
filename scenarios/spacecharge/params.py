"""Space-charge scenario: a charged Gaussian bunch drifting under its own 3D self-field.

A single drift of length ``ds`` with one space-charge kick. Unlike drift/FODO this is
*not* a linear map -- the self-consistent field grows the beam, so there is no analytic
reference; validation uses the 3D-PIC reference code (ImpactX) with a loose tolerance.
Codes using a different space-charge model (Xsuite 2.5D, PyORBIT 2.5D) are flagged as
``model_mismatch`` and plotted dashed.

The grid is intentionally modest (64^3) so the FFT Poisson solve fits the 4-vCPU /
16 GB hosted runners; bump ``N_CELL`` for local high-resolution runs.
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

# No analytic reference for self-consistent space charge.
