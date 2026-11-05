#!/usr/bin/env python3
# Auto-generated benchmark run script: Xsuite / htu (BELLA HTU beamline tracking).
# Shared htu_lattice with fair models (mat-kick-mat exact thick chromatic quad, exact
# drift, thick bend; zero-current correctors -> markers).
import json
import math

import numpy as np
import xobjects as xo
import xtrack as xt
from scenarios._xsuite_kernel_cache import enable as _enable_xsk_cache
from scenarios._xsuite_threaded_fft import make_context
_enable_xsk_cache()  # persistent compiled-kernel cache: compile once, reuse across layouts/runs

from scenarios._obs import Timer, beam_observables, gaussian_twiss_plane
from scenarios.htu.htu_lattice import get_lattice

p = {'mass_MeV': 0.510998950691753, 'mass_eV': 510998.9506917531, 'kin_energy_MeV': 99.48900104930824, 'kin_energy_eV': 99489001.04930824, 'total_energy_eV': 100000000.0, 'emit_x': 7.665084336342948e-09, 'emit_y': 7.665084336342948e-09, 'beta_x': 0.002, 'beta_y': 0.002, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 1e-06, 'sigma_p': 0.025, 'mu_p': 0.01, 'bunch_charge_C': 2.5e-11}
npart = 1000
ctx = make_context("cpu", omp_num_threads=16)
rng = np.random.default_rng(12345)

xx = gaussian_twiss_plane(p["emit_x"], p["beta_x"], p["alpha_x"], npart, rng)
yy = gaussian_twiss_plane(p["emit_y"], p["beta_y"], p["alpha_y"], npart, rng)
delta = rng.normal(0.0, p["sigma_p"], npart) + p["mu_p"]
zeta = rng.normal(0.0, p["sigma_t"], npart)

mass0 = p["mass_MeV"] * 1e6
e_tot = p["total_energy_eV"]
p0c = math.sqrt(e_tot * e_tot - mass0 * mass0)

particles = xt.Particles(
    _context=ctx, mass0=mass0, q0=-1.0, p0c=p0c,
    x=xx[0], px=xx[1], y=yy[0], py=yy[1], zeta=zeta, delta=delta,
)
line = xt.Line(elements=get_lattice("xsuite", screens_as_markers=True))
line.build_tracker(_context=ctx)

line.track(particles.copy())  # warm-up: cffi kernel JIT (NOT timed)
with Timer() as t:
    line.track(particles)

obs = beam_observables(particles.x, particles.px, particles.y, particles.py)
print(f"Track: {t.ns}ns")
print("Validate: " + json.dumps(obs))
