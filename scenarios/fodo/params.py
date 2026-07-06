"""FODO scenario: one focusing-defocusing cell as a CHROMATIC tracking benchmark.

Unlike ``drift`` (a linear sanity check with an analytic reference), FODO exercises
each code's **chromatic** quadrupole model. To make this a fair comparison, all four
codes are pinned to the **same** physical model:

* the SAME exact thick chromatic quadrupole map (analytic sin/cos focusing,
  sinh/cosh defocusing, focusing strength scaled by ``k1 -> k1/(1+delta)``):
  ImpactX ``ChrQuad``, Cheetah ``drift_kick_drift`` (num_steps=1), Xsuite
  ``model="mat-kick-mat"``, pyAT ``PassMethod="QuadLinearPass"`` (NOT its default
  10-step Yoshida thin-kick integrator);
* the full non-paraxial **exact drift** in every code;
* a meaningful momentum spread (``sigma_p``) so chromatic focusing matters;
* no analytic reference -- results are validated cross-code against ImpactX.
"""

from __future__ import annotations

# electron beam
MASS_MEV = 0.51099895069
KIN_ENERGY_MEV = 100.0

EMIT_X = 1.0e-9
EMIT_Y = 1.0e-9
BETA_X = 1.0
BETA_Y = 1.0
ALPHA_X = 0.0
ALPHA_Y = 0.0
SIGMA_T = 1.0e-3  # m
# meaningful momentum spread so the chromatic quad model is exercised
SIGMA_P = 1.0e-2  # dimensionless (1%)

# FODO cell: QF(+k1) -- drift -- QD(-k1) -- drift
QUAD_LENGTH = 0.1  # m
DRIFT_LENGTH = 0.5  # m
K1 = 2.0  # 1/m^2 (focusing strength of QF; QD uses -K1)

PARAMS = {
    "mass_MeV": MASS_MEV,
    "kin_energy_MeV": KIN_ENERGY_MEV,
    "emit_x": EMIT_X,
    "emit_y": EMIT_Y,
    "beta_x": BETA_X,
    "beta_y": BETA_Y,
    "alpha_x": ALPHA_X,
    "alpha_y": ALPHA_Y,
    "sigma_t": SIGMA_T,
    "sigma_p": SIGMA_P,
    "quad_length": QUAD_LENGTH,
    "drift_length": DRIFT_LENGTH,
    "k1": K1,
}

# No analytic reference: chromatic dynamics are validated cross-code vs ImpactX.
