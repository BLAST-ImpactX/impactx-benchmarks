#!/usr/bin/env python3
# Auto-generated benchmark run script: Synergia3 / spacecharge_2p5d (2.5D PIC).
#
# Space_charge_2d_open_hockney: transverse 2D open-boundary Hockney FFT Poisson with longitudinal
# binning (per-slice line-density weighting + Lorentz longitudinal correction) -- the 2.5D solver,
# the fair counterpart to ImpactX space_charge="2p5D" (both open transverse BC, n_cell^3 grid /
# gridz longitudinal bins, CIC deposit). ONE SC solve over the drift (fairness invariant), applied
# by the split-operator as a CENTERED kick: half-drift -> 1 SC kick -> half-drift. Exact non-paraxial
# libFF drift. One untimed warm-up propagate (warms the FFT plan), then a fresh timed propagate.
import json
import math

import numpy as np
import synergia

from scenarios._obs import Timer

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 250.0, 'bunch_charge_C': 1e-09, 'emit_x': 1e-07, 'emit_y': 1e-07, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.0001, 'drift_length': 6.0, 'n_cell': 64, 'grid_extent_sigma': 3.0}
npart = 1000000
n = int(p["n_cell"])
RANKS = 16
ELEM_CHARGE = 1.602176634e-19

rank = synergia.utils.Commxx.World.rank()
quiet = synergia.utils.Logger(0, synergia.utils.LoggerV.ERROR)

mass_GeV = p["mass_MeV"] * 1e-3
kin_GeV = p["kin_energy_MeV"] * 1e-3
etot_GeV = kin_GeV + mass_GeV
real_num = p["bunch_charge_C"] / ELEM_CHARGE

ref = synergia.foundation.Reference_particle(-1, mass_GeV, etot_GeV)


def twiss_cov():
    cov = np.zeros((6, 6))
    gx = (1.0 + p["alpha_x"] ** 2) / p["beta_x"]
    gy = (1.0 + p["alpha_y"] ** 2) / p["beta_y"]
    cov[0, 0] = p["emit_x"] * p["beta_x"]
    cov[0, 1] = cov[1, 0] = -p["emit_x"] * p["alpha_x"]
    cov[1, 1] = p["emit_x"] * gx
    cov[2, 2] = p["emit_y"] * p["beta_y"]
    cov[2, 3] = cov[3, 2] = -p["emit_y"] * p["alpha_y"]
    cov[3, 3] = p["emit_y"] * gy
    cov[4, 4] = p["sigma_t"] ** 2
    cov[5, 5] = p["sigma_p"] ** 2
    return cov


MEANS = np.zeros(6)
COV = twiss_cov()


def make_lattice():
    lattice = synergia.lattice.Lattice("sc2p5d")
    d = synergia.lattice.Lattice_element("drift", "d1")
    d.set_double_attribute("l", p["drift_length"])
    lattice.append(d)
    lattice.set_reference_particle(ref)
    lattice.set_all_string_attribute("extractor_type", "libff")
    return lattice


def make_propagator(lattice):
    sc = synergia.collective.Space_charge_2d_open_hockney_options(n, n, n)  # gridz = long. bins
    sc.comm_group_size = 1
    stepper = synergia.simulation.Split_operator_stepper(sc, 1)  # 1 step -> 1 solve
    return synergia.simulation.Propagator(lattice, stepper)


def make_sim(nn):
    sim = synergia.simulation.Bunch_simulator.create_single_bunch_simulator(ref, nn, real_num)
    dist = synergia.foundation.PCG_random_distribution(12345, rank)
    synergia.bunch.populate_6d(dist, sim.get_bunch(), MEANS, COV)
    return sim


# See spacecharge.py.jinja for the rationale: warm up + time the SAME sim (SC workspaces/cuFFT plan
# cache on its id, else they rebuild inside the timed region), and sync via an on-device reduction
# rather than an O(N) device->host checkout inside the timer -- both were unfair overheads vs ImpactX.
lattice = make_lattice()
propagator = make_propagator(lattice)

sim = make_sim(npart)
bunch = sim.get_bunch()
propagator.propagate(sim, quiet, 1)  # warm-up (NOT timed): builds plan + workspaces, JITs kernels
synergia.bunch.populate_6d(
    synergia.foundation.PCG_random_distribution(12345, rank), bunch, MEANS, COV)  # reset the beam

with Timer() as t:
    propagator.propagate(sim, quiet, 1)
    synergia.bunch.Core_diagnostics.calculate_mean(bunch)  # on-device reduction = cheap sync, no O(N) copy

bunch.checkout_particles()
mean = synergia.bunch.Core_diagnostics.calculate_mean(bunch)
std = synergia.bunch.Core_diagnostics.calculate_std(bunch, mean)
mom2 = synergia.bunch.Core_diagnostics.calculate_mom2(bunch, mean)

obs = {
    "sigma_x": float(std[0]),
    "sigma_y": float(std[2]),
    "sigma_t": float(std[4]),
    "emit_x": float(math.sqrt(max(mom2[0, 0] * mom2[1, 1] - mom2[0, 1] ** 2, 0.0))),
    "emit_y": float(math.sqrt(max(mom2[2, 2] * mom2[3, 3] - mom2[2, 3] ** 2, 0.0))),
}

if rank == 0:
    print(f"Track: {t.ns}ns")
    print("Validate: " + json.dumps(obs))

# Tear down the Kokkos-backed objects (Bunch_simulator + the collective SC operator's FFT/field
# Views) BEFORE the interpreter exits. `import synergia` registers an atexit hook that calls
# Kokkos::finalize; any such object still alive at Py_Finalize is deallocated AFTER
# Kokkos::finalize -> a host_abort ("... deallocated after Kokkos::finalize was called", exit 134)
# that the harness would misread as a FAILED run (runner keys on the process return code). Freeing
# them here runs the destructors while Kokkos is still live, so the process exits 0.
del sim, bunch, propagator, lattice
import gc
gc.collect()
