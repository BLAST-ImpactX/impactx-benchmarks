"""fodo_exact scenario parameters — a HOT, QUAD-DOMINATED FODO that isolates the exact
NON-PARAXIAL quad map.

Purpose: separate codes whose quad is the full exact (non-paraxial) Hamiltonian --
Ps = sqrt((1+dp)^2 - px^2 - py^2), i.e. amplitude-dependent focusing / "spherical aberration"
-- from codes whose quad is only chromatic-paraxial (a linear sin/cos matrix with k1/(1+dp)).
Both classes use an EXACT drift, so the drift model must NOT be what the test measures.

Two design requirements, both learned the hard way:

1) HOT beam (large transverse angles). The non-paraxial quad correction scales as ~sigma_x'^2,
   so at a cold beam (sigma_x' ~ 32 urad) exact and paraxial agree to ~1e-9. We use beta = 0.01 m,
   geometric emittance 1e-4 m.rad -> sigma_x' = sqrt(emit/beta) = 0.10 rad (100 mrad),
   sigma_x = sqrt(emit*beta) = 1 mm.

2) QUAD-DOMINATED geometry (long quads, short drifts). The FIRST version of this scenario used
   L_quad = 0.1 m with L_drift = 0.5 m -- but every code already does the drift EXACTLY, so a
   drift-dominated cell mostly measures the exact *drift* (which all codes get right). The quad
   model then moved sigma_x by only ~0.3% (below the 0.5% tolerance), so a paraxial-quad code
   like Cheetah passed and the test was meaningless. Inverting the ratio to L_quad = 0.5 m,
   L_drift = 0.1 m makes the quad map dominate: the quad-only sigma_x separation (ImpactX
   ExactQuad vs ChrQuad, identical exact drift in both) rises to ~1.5% -- well above tolerance.
   K1 = 2 m^-2 is the ceiling: K1 = 4 over-focuses the hot beam and loses particles.

100 MeV electron beam. No analytic reference: the exact non-paraxial dynamics are validated
cross-code against ImpactX ExactQuad + ExactDrift (the non-paraxial pole); ImpactX ChrQuad marks
the chromatic-paraxial pole. Codes landing at the paraxial pole are flagged model_mismatch. This
is a deliberately extreme stress test of the exact quad integrators, not a realistic operating beam.
"""

from __future__ import annotations

# electron beam (same species/energy as the shared fodo cell)
MASS_MEV = 0.51099895069
KIN_ENERGY_MEV = 100.0

# HOT, highly divergent beam: sigma_x' = sqrt(EMIT/BETA) = 0.10 rad (100 mrad)
EMIT_X = 1.0e-4  # m.rad (geometric) -- deliberately large to reach 100 mrad divergence
EMIT_Y = 1.0e-4
BETA_X = 0.01  # m
BETA_Y = 0.01
ALPHA_X = 0.0
ALPHA_Y = 0.0
SIGMA_T = 1.0e-3  # m
SIGMA_P = 1.0e-2  # dimensionless (1%)

# QUAD-DOMINATED FODO cell: QF(+k1) -- drift -- QD(-k1) -- drift. Long quads / short drifts so the
# exact non-paraxial QUAD map (not the exact drift, which every code has) dominates the observable.
QUAD_LENGTH = 0.5  # m  (was 0.1 -- inverted with the drift to make the cell quad-dominated)
DRIFT_LENGTH = 0.1  # m  (was 0.5)
K1 = 2.0  # 1/m^2  (ceiling for the hot beam: K1=4 over-focuses and loses particles)

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

# No analytic reference: exact non-paraxial dynamics validated cross-code vs ImpactX.
