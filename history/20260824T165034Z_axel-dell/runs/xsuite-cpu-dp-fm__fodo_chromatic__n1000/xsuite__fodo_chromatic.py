#!/usr/bin/env python3
# Auto-generated benchmark run script: Xsuite / fodo_chromatic (mat-kick-mat + Drift expanded; chromatic-paraxial).
import json
import math

import numpy as np
import xobjects as xo
import xtrack as xt
from scenarios._xsuite_kernel_cache import enable as _enable_xsk_cache
from scenarios._xsuite_threaded_fft import make_context
_enable_xsk_cache()  # persistent compiled-kernel cache: compile once, reuse across layouts/runs

from scenarios._obs import Timer, beam_observables, gaussian_twiss_plane

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 100.0, 'emit_x': 1e-09, 'emit_y': 1e-09, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.01, 'quad_length': 0.1, 'drift_length': 0.5, 'k1': 2.0}
npart = 1000
rng = np.random.default_rng(12345)
ctx = make_context("cpu", omp_num_threads=2)

xx = gaussian_twiss_plane(p["emit_x"], p["beta_x"], p["alpha_x"], npart, rng)
yy = gaussian_twiss_plane(p["emit_y"], p["beta_y"], p["alpha_y"], npart, rng)
delta = rng.normal(0.0, p["sigma_p"], npart)
zeta = rng.normal(0.0, p["sigma_t"], npart)

mass0 = p["mass_MeV"] * 1e6
e_tot = (p["kin_energy_MeV"] + p["mass_MeV"]) * 1e6
p0c = math.sqrt(e_tot * e_tot - mass0 * mass0)

particles = xt.Particles(
    _context=ctx, mass0=mass0, q0=-1.0, p0c=p0c,
    x=xx[0], px=xx[1], y=yy[0], py=yy[1], zeta=zeta, delta=delta,
)

# Same model as the other codes:
#  - mat-kick-mat: exact thick chromatic quad (analytic sin/cos, k1/(1+delta)) -- this
#    is also Xsuite's default, pinned here for robustness; matches Cheetah/ImpactX/pyAT.
#  - Drift model="expanded": chromatic-paraxial drift x += L*px/(1+delta).
line = xt.Line(
    elements=[
        xt.Quadrupole(length=p["quad_length"], k1=p["k1"], model="mat-kick-mat"),
        xt.Drift(length=p["drift_length"], model="expanded"),
        xt.Quadrupole(length=p["quad_length"], k1=-p["k1"], model="mat-kick-mat"),
        xt.Drift(length=p["drift_length"], model="expanded"),
    ]
)
line.build_tracker(_context=ctx)

line.track(particles.copy())  # warm-up: cffi kernel JIT (NOT timed)
with Timer() as t:
    line.track(particles)

obs = beam_observables(particles.x, particles.px, particles.y, particles.py)

print(f"Track: {t.ns}ns")
print("Validate: " + json.dumps(obs))
