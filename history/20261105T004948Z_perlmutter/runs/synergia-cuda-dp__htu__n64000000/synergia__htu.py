#!/usr/bin/env python3
# Auto-generated benchmark run script: Synergia3 / htu (BELLA HTU beamline tracking).
#
# The lattice is baked from the shared, code-agnostic htu_lattice "spec" (same physics: k1,
# bend angles, lengths as every other code), mapped to Synergia libFF elements:
#   quad -> "quadrupole" (k1 = MAD-X geometric strength, same sign as the spec used by
#           SciBmad/pyAT/Xsuite/PyORBIT), drift -> "drift", bend -> "sbend" (sector),
#   screens/zero-current correctors -> "marker".
#
# MODEL NOTE (untuned, like Cheetah/SciBmad/Bmad): htu is a chromatic-paraxial scenario, but
# Synergia has ONLY the exact (non-paraxial) libFF maps -- a Yoshida-6 symplectic quad + exact
# drift. So it runs the costlier EXACT map here. On this weakly divergent line the exact and
# chromatic-paraxial models agree within the 3% htu tolerance (as they do for the other exact
# codes), so it validates against the ImpactX ChrQuad+ChrDrift reference.
#
# Coordinates (x, xp=px/p0, y, yp=py/p0, cdt, dpop); dpop mean is set to +mu_p (the same mean
# momentum offset convention as SciBmad/pyAT/PyORBIT). One untimed warm-up propagate, then a
# fresh beam + timed propagate.
import json
import math

import numpy as np
import synergia

from scenarios._obs import Timer

p = {'mass_MeV': 0.510998950691753, 'mass_eV': 510998.9506917531, 'kin_energy_MeV': 99.48900104930824, 'kin_energy_eV': 99489001.04930824, 'total_energy_eV': 100000000.0, 'emit_x': 7.665084336342948e-09, 'emit_y': 7.665084336342948e-09, 'beta_x': 0.002, 'beta_y': 0.002, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 1e-06, 'sigma_p': 0.025, 'mu_p': 0.01, 'bunch_charge_C': 2.5e-11}
npart = 64000000
htu_spec = [{'kind': 'drift', 'name': 'SrcToPMQ1', 'L': 0.052}, {'kind': 'quad', 'name': 'PMQ1V', 'L': 0.02903, 'k1': -620.5784903834164}, {'kind': 'drift', 'name': 'L1', 'L': 0.029035}, {'kind': 'quad', 'name': 'PMQ2H', 'L': 0.0289, 'k1': 620.5784903834164}, {'kind': 'drift', 'name': 'L2', 'L': 0.0473895}, {'kind': 'quad', 'name': 'PMQ3V', 'L': 0.016321, 'k1': -553.1243066460885}, {'kind': 'drift', 'name': 'PMQTripToTCPhos', 'L': 0.2158}, {'kind': 'marker', 'name': 'TCPhosphor'}, {'kind': 'drift', 'name': 'TCPhosToChicane', 'L': 0.42}, {'kind': 'marker', 'name': 'S1'}, {'kind': 'bend', 'name': 'BEND1', 'L': 0.175, 'angle': -0.020341905108622157}, {'kind': 'drift', 'name': 'L12', 'L': 0.125}, {'kind': 'bend', 'name': 'BEND2', 'L': 0.175, 'angle': 0.020341905108622157}, {'kind': 'drift', 'name': 'L23', 'L': 0.15}, {'kind': 'marker', 'name': 'ChicaneSlit'}, {'kind': 'drift', 'name': 'L23', 'L': 0.15}, {'kind': 'bend', 'name': 'BEND3', 'L': 0.175, 'angle': 0.020341905108622157}, {'kind': 'drift', 'name': 'L12', 'L': 0.125}, {'kind': 'bend', 'name': 'BEND4', 'L': 0.175, 'angle': -0.020341905108622157}, {'kind': 'marker', 'name': 'S2'}, {'kind': 'drift', 'name': 'DriftToDCPhos', 'L': 0.27}, {'kind': 'marker', 'name': 'DCPhosphor'}, {'kind': 'drift', 'name': 'DriftToEMQTrip', 'L': 0.405}, {'kind': 'quad', 'name': 'EMQ1H', 'L': 0.1408, 'k1': 6.279005094089149}, {'kind': 'drift', 'name': 'EMQL1', 'L': 0.112735}, {'kind': 'quad', 'name': 'EMQ2V', 'L': 0.28141, 'k1': -7.732737058559862}, {'kind': 'drift', 'name': 'EMQL2', 'L': 0.112735}, {'kind': 'quad', 'name': 'EMQ3H', 'L': 0.1409, 'k1': 9.794000540820837}, {'kind': 'marker', 'name': 'S3'}, {'kind': 'drift', 'name': 'DriftToPhos1', 'L': 0.084325}, {'kind': 'marker', 'name': 'Phosphor1'}, {'kind': 'drift', 'name': 'DriftToSpec', 'L': 0.28}, {'kind': 'bend', 'name': 'MagSpec', 'L': 0.4826, 'angle': 0.0}, {'kind': 'marker', 'name': 'S4'}, {'kind': 'drift', 'name': 'DriftToAline1', 'L': 0.4009}, {'kind': 'marker', 'name': 'UC_ALineEbeam1'}, {'kind': 'drift', 'name': 'DriftToAline2', 'L': 0.3825}, {'kind': 'marker', 'name': 'UC_ALineEBeam2'}, {'kind': 'drift', 'name': 'DriftToAline3', 'L': 0.4191}, {'kind': 'marker', 'name': 'UC_ALineEBeam3'}, {'kind': 'drift', 'name': 'DriftToUndulator', 'L': 0.2945}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS1'}, {'kind': 'marker', 'name': 'UC_VisaEBeam1'}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS2'}, {'kind': 'marker', 'name': 'UC_VisaEBeam2'}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ1', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ2', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS3'}, {'kind': 'marker', 'name': 'UC_VisaEBeam3'}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS4'}, {'kind': 'marker', 'name': 'UC_VisaEBeam4'}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ3', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ4', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS5'}, {'kind': 'marker', 'name': 'UC_VisaEBeam5'}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS6'}, {'kind': 'marker', 'name': 'UC_VisaEBeam6'}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ5', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ6', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS7'}, {'kind': 'marker', 'name': 'UC_VisaEBeam7'}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'marker', 'name': 'VS8'}, {'kind': 'marker', 'name': 'UC_VisaEBeam8'}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'quad', 'name': 'VQ7', 'L': 0.0504, 'k1': 98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'quad', 'name': 'VQ8', 'L': 0.0504, 'k1': -98.93280281474755}, {'kind': 'drift', 'name': 'VD', 'L': 0.018}]

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
MEANS[5] = p["mu_p"]  # mean dpop offset (same sign convention as SciBmad/pyAT/PyORBIT)
COV = twiss_cov()


def make_element(el):
    kind = el["kind"]
    name = el.get("name", "e")
    if kind == "drift":
        e = synergia.lattice.Lattice_element("drift", name)
        e.set_double_attribute("l", el["L"])
        return e
    if kind == "quad":
        e = synergia.lattice.Lattice_element("quadrupole", name)
        e.set_double_attribute("l", el["L"])
        e.set_double_attribute("k1", el["k1"])  # MAD-X 1/m^2, same sign as ImpactX ChrQuad unit=0
        return e
    if kind == "bend":
        if abs(el["angle"]) < 1e-12:  # zero-angle "bend" (spectrometer off) == drift
            e = synergia.lattice.Lattice_element("drift", name)
            e.set_double_attribute("l", el["L"])
            return e
        e = synergia.lattice.Lattice_element("sbend", name)  # sector bend (e1=e2=0), like ImpactX ExactSbend
        e.set_double_attribute("l", el["L"])
        e.set_double_attribute("angle", el["angle"])
        return e
    return synergia.lattice.Lattice_element("marker", name)  # screens / zero-current correctors


def make_lattice():
    lattice = synergia.lattice.Lattice("htu")
    for el in htu_spec:
        lattice.append(make_element(el))
    lattice.set_reference_particle(ref)
    lattice.set_all_string_attribute("extractor_type", "libff")  # exact non-paraxial maps
    return lattice


def make_propagator(lattice):
    stepper = synergia.simulation.Independent_stepper_elements(1)  # tracking only, 1 slice/element
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

# Tear down the Kokkos-backed objects BEFORE the interpreter exits (see fodo templates): any
# Kokkos::View still alive at Py_Finalize is freed AFTER Kokkos::finalize -> a host_abort the
# harness would misread as a FAILED run. Freeing here runs the destructors while Kokkos is live.
del sim, bunch, propagator, lattice
import gc
gc.collect()
