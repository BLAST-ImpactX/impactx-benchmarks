#!/usr/bin/env python3
# Auto-generated benchmark run script: ImpactX / spacecharge_2p5d (2.5D PIC space charge).
#
# 2.5D PIC space charge (space_charge="2p5D"): transverse 2D FFT Poisson per longitudinal
# slice. This is the 2.5D reference (vs PyORBIT/Xsuite 2.5D). Open-boundary FFT, n_cell^3
# grid, linear (CIC) deposition (particle_shape=1), exact (chromatic) drift. Domain padded
# 10% over the beam max extent (prob_relative=1.2); open BC -> size-insensitive.
# One untimed warm-up track (also warms the FFT plan), then reset beam + reference particle.
import json

from impactx import ImpactX, distribution, elements, twiss

from scenarios._obs import Timer

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 250.0, 'bunch_charge_C': 1e-09, 'emit_x': 1e-07, 'emit_y': 1e-07, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.0001, 'drift_length': 6.0, 'n_cell': 64, 'grid_extent_sigma': 3.0}
npart = 1000
n = int(p["n_cell"])


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
    sim.add_particles(p["bunch_charge_C"], make_distr(), npart)


sim = ImpactX()
# Grid decomposition keyed on the MPI rank count of this layout (8 rank(s), baked in at
# render time). ONE rank (every GPU run + 1-rank pure-OpenMP CPU layouts) -> EXACTLY ONE grid, so
# there is no inter-box particle Redistribute (which otherwise dominates and scales super-linearly).
# >1 rank -> split so each rank can own a grid. The open-BC FFT solver gathers the full domain, so
# the result is unchanged. amr.max_grid_size = n_cell (before init_grids) stops AMReX auto-splitting.
bf = max(n // 2, 1)
sim.max_level = 0
sim.n_cell = [n, n, n]
sim.blocking_factor_x = [bf]
sim.blocking_factor_y = [bf]
sim.blocking_factor_z = [bf]
sim.particle_shape = 1
sim.space_charge = "2p5D"
sim.poisson_solver = "fft"   # IGF / open boundary (matches Cheetah)
sim.dynamic_size = True
sim.prob_relative = [1.2]     # domain = beam max-extent + 10% (open BC -> size-insensitive)
sim.slice_step_diagnostics = False
sim.diagnostics = False
sim.verbose = 0
sim.mlmg_verbosity = 0
sim.tiny_profiler = False
sim.init_grids()
set_ref(sim)
sim.add_particles(p["bunch_charge_C"], make_distr(), npart)

sim.lattice.extend([elements.ExactDrift(name="d1", ds=p["drift_length"], nslice=1)])

# warm-up track (NOT timed), then reset beam + reference particle
sim.track_particles()
load_beam(sim)

with Timer() as t:
    sim.track_particles()

rbc = sim.beam.reduced_beam_characteristics()
obs = {
    "sigma_x": float(rbc["sig_x"]),
    "sigma_y": float(rbc["sig_y"]),
    "sigma_t": float(rbc["sig_t"]),
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
