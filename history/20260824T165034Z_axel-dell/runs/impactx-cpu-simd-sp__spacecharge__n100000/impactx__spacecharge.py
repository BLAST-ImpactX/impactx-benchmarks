#!/usr/bin/env python3
# Auto-generated benchmark run script: ImpactX / spacecharge (3D FFT space charge).
#
# Model matched to Cheetah's SpaceChargeKick for a fair comparison:
#   * solver: integrated Green function (IGF), OPEN boundaries via FFT (poisson_solver="fft")
#             -- the SAME method Cheetah uses (Qiang/Ryne IGF + Hockney zero-padding).
#   * grid:   n_cell^3 cells (= Cheetah grid_shape).
#   * deposition/gather: linear / cloud-in-cell (particle_shape=1) = Cheetah CIC.
#   * exact (chromatic) drift.
# Domain: because the solver is OPEN-boundary, the result is insensitive to the box size
# as long as it contains the beam, so we just pad the beam's max extent by 10%
# (prob_relative=1.2). Cheetah uses a +/-3 sigma box; for a Gaussian both contain the beam
# and the two agree to <0.1% (verified) -- the padding is not a meaningful model difference.
# One untimed warm-up track (also warms the FFT plan), then reset beam + reference particle.
import json

from impactx import ImpactX, distribution, elements, twiss

from scenarios._obs import Timer

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 250.0, 'bunch_charge_C': 1e-09, 'emit_x': 1e-07, 'emit_y': 1e-07, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.0001, 'drift_length': 6.0, 'n_cell': 64, 'grid_extent_sigma': 3.0}
npart = 100000
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
# Grid decomposition keyed on the MPI rank count of this layout (1 rank(s), baked in at
# render time by the harness). With ONE rank -- every GPU run AND the 1-rank pure-OpenMP CPU layouts
# -- use EXACTLY ONE grid (a single box covering the whole domain). Otherwise AMReX's multi-box
# particle Redistribute dominates (profiled at 68-76% of the GPU time) and scales super-linearly.
# With >1 rank we split so each rank can own a grid (a single n_cell-sized box leaves extra ranks
# with none). The open-BC FFT solver gathers the full domain either way, so the result is unchanged.
# amr.max_grid_size = n_cell stops AMReX auto-splitting n_cell^3 into 8 boxes; MUST precede init_grids.
import amrex.space3d as _amr
_amr.ParmParse("amr").add("max_grid_size", n)
bf = n                    # one block per axis -> a single grid (no inter-box Redistribute)
sim.max_level = 0
sim.n_cell = [n, n, n]
sim.blocking_factor_x = [bf]
sim.blocking_factor_y = [bf]
sim.blocking_factor_z = [bf]
sim.particle_shape = 1
sim.space_charge = "3D"
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
