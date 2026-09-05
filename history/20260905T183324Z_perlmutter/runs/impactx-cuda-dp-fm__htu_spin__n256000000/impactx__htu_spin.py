#!/usr/bin/env python3
# Auto-generated benchmark run script: ImpactX / htu_spin.
# HTU beamline with Thomas-BMT spin tracking. The beam starts fully spin-aligned (+z);
# the dispersive chicane differentially precesses the spins -> depolarization, measured
# as the RMS spin spread sigma_sx/sy/sz. The zero-current Kicker steering elements have no
# spin support, so (being zero) they are swapped for spin-safe Markers.
# One untimed warm-up track, then reset beam + reference particle, then time.
import json

from impactx import ImpactX, distribution, elements, twiss  # noqa: F401

from scenarios._obs import Timer
from scenarios.htu.htu_lattice import get_lattice

p = {'mass_MeV': 0.510998950691753, 'mass_eV': 510998.9506917531, 'kin_energy_MeV': 99.48900104930824, 'kin_energy_eV': 99489001.04930824, 'total_energy_eV': 100000000.0, 'emit_x': 7.665084336342948e-09, 'emit_y': 7.665084336342948e-09, 'beta_x': 0.002, 'beta_y': 0.002, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 1e-06, 'sigma_p': 0.025, 'mu_p': 0.01, 'bunch_charge_C': 2.5e-11}
npart = 256000000


def make_distr():
    tw = twiss(
        beta_x=p["beta_x"], beta_y=p["beta_y"], beta_t=p["sigma_t"] / p["sigma_p"],
        emitt_x=p["emit_x"], emitt_y=p["emit_y"], emitt_t=p["sigma_t"] * p["sigma_p"],
        alpha_x=p["alpha_x"], alpha_y=p["alpha_y"], alpha_t=0.0,
    )
    tw["meanPt"] = -p["mu_p"]
    return distribution.Gaussian(**tw)


def set_ref(sim):
    ref = sim.beam.ref
    # set_species sets mass, charge AND the gyromagnetic anomaly required for spin tracking
    ref.set_species("electron").set_kin_energy_MeV(p["kin_energy_MeV"])
    ref.s = 0.0
    ref.t = 0.0


def load_beam(sim):
    sim.beam.clear_particles()
    set_ref(sim)
    # all spins aligned along +z (zero initial spread); depolarization measured after track
    sim.add_particles(p["bunch_charge_C"], make_distr(), npart,
                      distribution.SpinvMF(0.0, 0.0, 1.0))


sim = ImpactX()
sim.spin = True
sim.particle_shape = 1
sim.space_charge = False
sim.slice_step_diagnostics = False
sim.diagnostics = False
sim.verbose = 0
sim.tiny_profiler = False
sim.init_grids()
load_beam(sim)

# spin-safe lattice: zero-current Kickers -> Markers (Kicker has no spin map; kick is 0)
_lat = [elements.Marker(name="kick") if type(e).__name__ == "Kicker" else e
        for e in get_lattice("impactx", screens_as_markers=True)]
sim.lattice.extend(_lat)

# warm-up track (NOT timed), then reset beam + reference particle
sim.track_particles()
load_beam(sim)

with Timer() as t:
    sim.track_particles()

rbc = sim.beam.reduced_beam_characteristics()
obs = {
    "sigma_sx": float(rbc["sigma_sx"]),
    "sigma_sy": float(rbc["sigma_sy"]),
    "sigma_sz": float(rbc["sigma_sz"]),
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
