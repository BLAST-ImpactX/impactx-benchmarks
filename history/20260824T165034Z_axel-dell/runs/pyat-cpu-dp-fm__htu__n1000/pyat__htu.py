#!/usr/bin/env python3
# Auto-generated benchmark run script: pyAT / htu (BELLA HTU beamline tracking).
# Uses the shared htu_lattice with fair models (QuadLinearPass exact thick chromatic
# quad, ExactDriftPass, symplectic bend; zero-current correctors -> markers).
import json

import numpy as np
import at

from scenarios._obs import Timer, beam_observables, gaussian_twiss_plane
from scenarios.htu.htu_lattice import get_lattice

p = {'mass_MeV': 0.510998950691753, 'mass_eV': 510998.9506917531, 'kin_energy_MeV': 99.48900104930824, 'kin_energy_eV': 99489001.04930824, 'total_energy_eV': 100000000.0, 'emit_x': 7.665084336342948e-09, 'emit_y': 7.665084336342948e-09, 'beta_x': 0.002, 'beta_y': 0.002, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 1e-06, 'sigma_p': 0.025, 'mu_p': 0.01, 'bunch_charge_C': 2.5e-11}
npart = 1000
rng = np.random.default_rng(12345)

xx = gaussian_twiss_plane(p["emit_x"], p["beta_x"], p["alpha_x"], npart, rng)
yy = gaussian_twiss_plane(p["emit_y"], p["beta_y"], p["alpha_y"], npart, rng)
rin = np.asfortranarray(np.zeros((6, npart)))
rin[0], rin[1] = xx[0], xx[1]
rin[2], rin[3] = yy[0], yy[1]
rin[4] = rng.normal(0.0, p["sigma_p"], npart) + p["mu_p"]  # delta with mean offset
rin[5] = rng.normal(0.0, p["sigma_t"], npart)

ring = at.Lattice(get_lattice("pyat", screens_as_markers=True),
                  energy=p["total_energy_eV"], periodicity=1)

at.lattice_track(ring, rin.copy(), nturns=1, refpts=at.End)  # warm-up (NOT timed)
with Timer() as t:
    out = at.lattice_track(ring, rin, nturns=1, refpts=at.End)

rout = out[0] if isinstance(out, tuple) else out
final = np.asarray(rout)[:, :, -1, -1]
obs = beam_observables(final[0], final[1], final[2], final[3])
print(f"Track: {t.ns}ns")
print("Validate: " + json.dumps(obs))
