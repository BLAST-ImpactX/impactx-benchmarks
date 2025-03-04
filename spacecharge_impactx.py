#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import time
from impactx import ImpactX, distribution, elements

sim = ImpactX()

# set numerical parameters and IO control
sim.max_level = 0
sim.n_cell = [256, 256, 256]
sim.blocking_factor_x = [256]
sim.blocking_factor_y = [256]
sim.blocking_factor_z = [256]

# B-spline order for charge deposition
# note: gather in ImpactX is always linear polynomial right now
sim.particle_shape = 1
sim.space_charge = "3D"
sim.poisson_solver = "fft"
sim.dynamic_size = True
sim.prob_relative = [1.2]
sim.verbose = 0
sim.mlmg_verbosity = 0

# beam diagnostics
sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = False

# domain decomposition & space charge mesh
sim.init_grids()

# load a 2 GeV electron beam with an initial
# unnormalized rms emittance of 2 nm
kin_energy_MeV = 250.0   # reference energy
bunch_charge_C = 1.0e-8  # used with space charge
npart = 30_000_000  # number of macro particles

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(-1.0).set_mass_MeV(0.510998950).set_kin_energy_MeV(kin_energy_MeV)

#   particle bunch
distr = distribution.Kurth6D(
    lambdaX=4.472135955e-4,
    lambdaY=4.472135955e-4,
    lambdaT=9.12241869e-7,
    lambdaPx=0.0,
    lambdaPy=0.0,
    lambdaPt=0.0,
)
sim.add_particles(bunch_charge_C, distr, npart)

# design the accelerator lattice
sim.lattice.extend([elements.Drift(name="d1", ds=6.0, nslice=1)])

# run simulation
start_time = time.perf_counter()
sim.track_particles()
end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution time ImpactX: {execution_time:.4f} seconds ({execution_time/npart*1e6}us / particle)")

# clean shutdown
sim.finalize()
