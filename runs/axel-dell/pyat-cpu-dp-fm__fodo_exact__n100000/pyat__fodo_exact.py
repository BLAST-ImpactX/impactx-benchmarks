#!/usr/bin/env python3
# Auto-generated benchmark run script: pyAT / fodo_exact (ExactMultipolePass quad + ExactDriftPass; exact non-paraxial).
import json

import numpy as np
import at

from scenarios._obs import Timer, beam_observables, gaussian_twiss_plane

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 100.0, 'emit_x': 0.0001, 'emit_y': 0.0001, 'beta_x': 0.01, 'beta_y': 0.01, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.01, 'quad_length': 0.5, 'drift_length': 0.1, 'k1': 2.0}
npart = 100000
rng = np.random.default_rng(12345)

xx = gaussian_twiss_plane(p["emit_x"], p["beta_x"], p["alpha_x"], npart, rng)
yy = gaussian_twiss_plane(p["emit_y"], p["beta_y"], p["alpha_y"], npart, rng)
rin = np.asfortranarray(np.zeros((6, npart)))
rin[0], rin[1] = xx[0], xx[1]
rin[2], rin[3] = yy[0], yy[1]
rin[4] = rng.normal(0.0, p["sigma_p"], npart)
rin[5] = rng.normal(0.0, p["sigma_t"], npart)

energy_eV = (p["kin_energy_MeV"] + p["mass_MeV"]) * 1e6
# Use the SAME model as the other codes:
#  - QuadLinearPass: exact thick chromatic quad (analytic sin/cos with k1/(1+delta)),
#    matching Cheetah drift_kick_drift / ImpactX ChrQuad / Xsuite mat-kick-mat.
#    (NOT the default StrMPoleSymplectic4Pass 10-step Yoshida thin-kick integrator.)
#  - ExactDriftPass: full non-paraxial exact drift, matching the others.
ring = at.Lattice(
    [
        # NumIntSteps=4: converged on the hot beam through the long 0.5 m quad (default is 10).
        at.Quadrupole("QF", p["quad_length"], p["k1"], PassMethod="ExactMultipolePass", NumIntSteps=4),
        at.Drift("D1", p["drift_length"], PassMethod="ExactDriftPass"),
        at.Quadrupole("QD", p["quad_length"], -p["k1"], PassMethod="ExactMultipolePass", NumIntSteps=4),
        at.Drift("D2", p["drift_length"], PassMethod="ExactDriftPass"),
    ],
    energy=energy_eV,
    periodicity=1,
)

at.lattice_track(ring, rin.copy(), nturns=1, refpts=at.End)  # warm-up (NOT timed)
with Timer() as t:
    out = at.lattice_track(ring, rin, nturns=1, refpts=at.End)

rout = out[0] if isinstance(out, tuple) else out
final = np.asarray(rout)[:, :, -1, -1]
obs = beam_observables(final[0], final[1], final[2], final[3])

print(f"Track: {t.ns}ns")
print("Validate: " + json.dumps(obs))
