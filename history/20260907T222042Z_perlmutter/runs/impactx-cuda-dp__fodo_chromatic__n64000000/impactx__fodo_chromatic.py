#!/usr/bin/env python3
# Auto-generated benchmark run script: ImpactX / fodo_chromatic (ChrQuad + ChrDrift; consistent chromatic-paraxial).
# One untimed warm-up track, then reset beam + reference particle (track advances
# both), then time the steady-state track. See drift template for rationale.
import json

from impactx import ImpactX, distribution, elements, twiss

from scenarios._obs import Timer

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 100.0, 'emit_x': 1e-09, 'emit_y': 1e-09, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.01, 'quad_length': 0.1, 'drift_length': 0.5, 'k1': 2.0}
npart = 64000000


def make_distr():
    return distribution.Gaussian(**twiss(
        beta_x=p["beta_x"], beta_y=p["beta_y"], beta_t=p["sigma_t"] / p["sigma_p"],
        emitt_x=p["emit_x"], emitt_y=p["emit_y"], emitt_t=p["sigma_t"] * p["sigma_p"],
        alpha_x=p["alpha_x"], alpha_y=p["alpha_y"], alpha_t=0.0,
    ))


def set_ref(sim):
    ref = sim.beam.ref
    ref.set_charge_qe(-1.0).set_mass_MeV(p["mass_MeV"]).set_kin_energy_MeV(p["kin_energy_MeV"])
    ref.s = 0.0
    ref.t = 0.0


def load_beam(sim):
    sim.beam.clear_particles()
    set_ref(sim)
    sim.add_particles(1.0e-12, make_distr(), npart)


sim = ImpactX()
sim.particle_shape = 1
sim.space_charge = False
sim.slice_step_diagnostics = False
sim.diagnostics = False
sim.verbose = 0
sim.tiny_profiler = False
sim.init_grids()
set_ref(sim)
sim.add_particles(1.0e-12, make_distr(), npart)

# FODO: ChrQuad (chromatic-paraxial thick quad, k1/(1+delta)) + ChrDrift
sim.lattice.extend([
    elements.ChrQuad(name="qf", ds=p["quad_length"], k=p["k1"], unit=0, nslice=1),
    elements.ChrDrift(name="d1", ds=p["drift_length"], nslice=1),
    elements.ChrQuad(name="qd", ds=p["quad_length"], k=-p["k1"], unit=0, nslice=1),
    elements.ChrDrift(name="d2", ds=p["drift_length"], nslice=1),
])

# warm-up track (NOT timed), then reset beam + reference particle
sim.track_particles()
load_beam(sim)

with Timer() as t:
    sim.track_particles()

rbc = sim.beam.reduced_beam_characteristics()
obs = {
    "sigma_x": float(rbc["sig_x"]),
    "sigma_y": float(rbc["sig_y"]),
    "emit_x": float(rbc["emittance_x"]),
    "emit_y": float(rbc["emittance_y"]),
}
# under MPI only rank 0 prints (reduced_beam_characteristics is already global)
try:
    import amrex.space3d as amr
    _rank = amr.ParallelDescriptor.MyProc()
except Exception:
    _rank = 0
if _rank == 0:
    print(f"Track: {t.ns}ns")
    print("Validate: " + json.dumps(obs))

sim.finalize()
