#!/usr/bin/env python3
# Auto-generated benchmark run script: Synergia3 / fodo_exact (EXACT non-paraxial FODO tracking).
#
# Synergia's libFF quadrupole is a Yoshida-6 SYMPLECTIC integrator (default 4 sub-steps) of thin
# quad kicks interleaved with the EXACT non-paraxial drift (Ps = sqrt((1+dp)^2 - px^2 - py^2)) --
# i.e. the exact (non-paraxial, nonlinear/amplitude-dependent) quad Hamiltonian flow, the SAME
# exact model as ImpactX ExactQuad + ExactDrift and SciBmad MatrixKick. So Synergia is a fair
# EXACT participant in this quad-dominated hot-beam test (not the paraxial pole). No space charge.
# Coordinates (x, xp=px/p0, y, yp=py/p0, cdt, dpop); xp/yp are canonical (like ImpactX px/py/p0).
# One untimed warm-up propagate, then a fresh beam + timed propagate.
import json
import math

import numpy as np
import synergia

from scenarios._obs import Timer

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 100.0, 'emit_x': 0.0001, 'emit_y': 0.0001, 'beta_x': 0.01, 'beta_y': 0.01, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.01, 'quad_length': 0.5, 'drift_length': 0.1, 'k1': 2.0}
npart = 1000000000

rank = synergia.utils.Commxx.World.rank()
quiet = synergia.utils.Logger(0, synergia.utils.LoggerV.ERROR)

mass_GeV = p["mass_MeV"] * 1e-3
kin_GeV = p["kin_energy_MeV"] * 1e-3
etot_GeV = kin_GeV + mass_GeV
REAL_NUM = 1.0e10  # arbitrary (no space charge here -> unused by the physics)

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


def quad(name, k1):
    q = synergia.lattice.Lattice_element("quadrupole", name)
    q.set_double_attribute("l", p["quad_length"])
    q.set_double_attribute("k1", k1)  # MadX convention (1/m^2), same sign as ImpactX ChrQuad unit=0
    return q


def drift(name):
    d = synergia.lattice.Lattice_element("drift", name)
    d.set_double_attribute("l", p["drift_length"])
    return d


def make_lattice():
    lattice = synergia.lattice.Lattice("fodo")
    for elem in (quad("qf", p["k1"]), drift("d1"), quad("qd", -p["k1"]), drift("d2")):
        lattice.append(elem)
    lattice.set_reference_particle(ref)
    lattice.set_all_string_attribute("extractor_type", "libff")  # exact non-paraxial maps
    return lattice


def make_propagator(lattice):
    # tracking only (no collective operator); 1 slice/element (each quad still integrates with its
    # default 4 Yoshida-6 sub-steps -> converged exact map, like ImpactX ExactQuad nslice=1).
    stepper = synergia.simulation.Independent_stepper_elements(1)
    return synergia.simulation.Propagator(lattice, stepper)


def make_sim(nn):
    sim = synergia.simulation.Bunch_simulator.create_single_bunch_simulator(ref, nn, REAL_NUM)
    dist = synergia.foundation.PCG_random_distribution(12345, rank)
    synergia.bunch.populate_6d(dist, sim.get_bunch(), MEANS, COV)
    return sim


lattice = make_lattice()
propagator = make_propagator(lattice)

# warm-up propagate (NOT timed)
propagator.propagate(make_sim(min(2000, npart)), quiet, 1)

sim = make_sim(npart)
bunch = sim.get_bunch()
with Timer() as t:
    propagator.propagate(sim, quiet, 1)
    bunch.checkout_particles()  # device sync on GPU / cheap host copy on CPU

mean = synergia.bunch.Core_diagnostics.calculate_mean(bunch)
std = synergia.bunch.Core_diagnostics.calculate_std(bunch, mean)
mom2 = synergia.bunch.Core_diagnostics.calculate_mom2(bunch, mean)

obs = {
    "sigma_x": float(std[0]),
    "sigma_y": float(std[2]),
    "emit_x": float(math.sqrt(max(mom2[0, 0] * mom2[1, 1] - mom2[0, 1] ** 2, 0.0))),
    "emit_y": float(math.sqrt(max(mom2[2, 2] * mom2[3, 3] - mom2[2, 3] ** 2, 0.0))),
}

if rank == 0:
    print(f"Track: {t.ns}ns")
    print("Validate: " + json.dumps(obs))

# Tear down the Kokkos-backed objects (Bunch_simulator etc.) BEFORE the interpreter exits.
# `import synergia` registers an atexit hook that calls Kokkos::finalize; any such object still
# alive at Py_Finalize is deallocated AFTER Kokkos::finalize -> a host_abort ("... deallocated
# after Kokkos::finalize was called", exit 134) that the harness would misread as a FAILED run
# (runner keys on the process return code). Freeing them here runs the destructors while Kokkos
# is still live, so the process exits 0.
del sim, bunch, propagator, lattice
import gc
gc.collect()
