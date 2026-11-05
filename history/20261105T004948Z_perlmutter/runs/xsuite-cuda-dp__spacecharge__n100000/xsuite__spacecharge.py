#!/usr/bin/env python3
# Auto-generated benchmark run script: Xsuite / spacecharge (3D PIC space charge).
# xfields SpaceCharge3D with solver="FFTSolver3D": the genuine 3D integrated-Green-function
# open-boundary FFT Poisson solver (same family as ImpactX/Cheetah). n_cell^3 grid, exact drift. One
# SC kick, including its longitudinal part (apply_z_kick), which the check validates via sigma_pe.
# The box is FIXED at construction, at +/- grid_extent_sigma * sigma of this beam, and particles
# outside it get no kick. That is sound here because the cold beam barely grows over the drift; on a
# beam that grows, the mesh would no longer hold the charge when the kick comes after the drift.
import json
import math

import numpy as np
import xobjects as xo
import xtrack as xt
from scenarios._xsuite_kernel_cache import enable as _enable_xsk_cache
_enable_xsk_cache()  # persistent compiled-kernel cache: compile once, reuse across layouts/runs
import xfields as xf

from scenarios._obs import Timer, beam_observables, gaussian_twiss_plane, momentum_observables
from scenarios._xsuite_threaded_fft import make_context

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 250.0, 'bunch_charge_C': 1e-09, 'emit_x': 1.5533051715234088e-12, 'emit_y': 1.5533051715234088e-12, 'beta_x': 643788.4958686174, 'beta_y': 643788.4958686174, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 1.9241656871487193e-08, 'drift_length': 6.4, 'n_cell': 64, 'grid_extent_sigma': 5.0}
npart = 100000
n = int(p["n_cell"])
N = float(p["grid_extent_sigma"])
ds = p["drift_length"]
ELEM_CHARGE = 1.602176634e-19
# threaded scipy.fft plan (workers = OMP threads); xobjects' default CPU FFT is single-threaded
# numpy (pyfftw, its threaded path, is broken with xfields). See scenarios/_xsuite_threaded_fft.py
ctx = make_context("cuda", omp_num_threads=1, threaded_fft=True)

rng = np.random.default_rng(12345)
xx = gaussian_twiss_plane(p["emit_x"], p["beta_x"], p["alpha_x"], npart, rng)
yy = gaussian_twiss_plane(p["emit_y"], p["beta_y"], p["alpha_y"], npart, rng)
zeta = rng.normal(0.0, p["sigma_t"], npart)
delta = rng.normal(0.0, p["sigma_p"], npart)

mass0 = p["mass_MeV"] * 1e6
e_tot = (p["kin_energy_MeV"] + p["mass_MeV"]) * 1e6
p0c = math.sqrt(e_tot * e_tot - mass0 * mass0)
gamma0 = e_tot / mass0

particles = xt.Particles(
    _context=ctx, mass0=mass0, q0=-1.0, p0c=p0c,
    x=xx[0], px=xx[1], y=yy[0], py=yy[1], zeta=zeta, delta=delta,
)
# weight so the total bunch charge equals bunch_charge_C
particles.weight[:] = (p["bunch_charge_C"] / ELEM_CHARGE) / npart

sigx = math.sqrt(p["emit_x"] * p["beta_x"])
sigy = math.sqrt(p["emit_y"] * p["beta_y"])
sigz = p["sigma_t"]

sc = xf.SpaceCharge3D(
    _context=ctx, update_on_track=True, length=ds, apply_z_kick=True,
    x_range=(-N * sigx, N * sigx), y_range=(-N * sigy, N * sigy),
    z_range=(-N * sigz, N * sigz), nx=n, ny=n, nz=n,
    solver="FFTSolver3D", gamma0=gamma0,
)
line = xt.Line(elements=[xt.Drift(length=ds, model="exact"), sc])
line.build_tracker(_context=ctx)

line.track(particles.copy())  # warm-up: cffi kernel JIT + FFT plan (NOT timed)
with Timer() as t:
    line.track(particles)

obs = beam_observables(particles.x, particles.px, particles.y, particles.py,
                       tau=particles.zeta)
obs.update(momentum_observables(particles.x, particles.px, particles.y, particles.py, particles.ptau))
print(f"Track: {t.ns}ns")
print("Validate: " + json.dumps(obs))
