#!/usr/bin/env python3
# Auto-generated benchmark run script: PyORBIT3 / htu (BELLA HTU beamline tracking).
# Shared htu_lattice with TEAPOT nodes (QuadTEAPOT nParts=8 -> exact thick chromatic
# quad, paraxial drift, BendTEAPOT; zero-current correctors -> zero-length drifts).
import copy
import json
import math
import time

import numpy as np

from orbit.core.bunch import Bunch, BunchTwissAnalysis
from orbit.teapot import TEAPOT_Lattice
from orbit.core.orbit_mpi import mpi_comm, MPI_Comm_rank, MPI_Comm_size

from scenarios._obs import gaussian_twiss_plane
from scenarios.htu.htu_lattice import get_lattice

p = {'mass_MeV': 0.510998950691753, 'mass_eV': 510998.9506917531, 'kin_energy_MeV': 99.48900104930824, 'kin_energy_eV': 99489001.04930824, 'total_energy_eV': 100000000.0, 'emit_x': 7.665084336342948e-09, 'emit_y': 7.665084336342948e-09, 'beta_x': 0.002, 'beta_y': 0.002, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 1e-06, 'sigma_p': 0.025, 'mu_p': 0.01, 'bunch_charge_C': 2.5e-11}
npart = 1000

comm = mpi_comm.MPI_COMM_WORLD
rank = MPI_Comm_rank(comm)
size = MPI_Comm_size(comm)

mass_GeV = p["mass_MeV"] * 1e-3
kin_GeV = p["kin_energy_MeV"] * 1e-3
e_tot = kin_GeV + mass_GeV
p0c = math.sqrt(e_tot * e_tot - mass_GeV * mass_GeV)
beta = p0c / e_tot

rng = np.random.default_rng(12345)
xx = gaussian_twiss_plane(p["emit_x"], p["beta_x"], p["alpha_x"], npart, rng)
yy = gaussian_twiss_plane(p["emit_y"], p["beta_y"], p["alpha_y"], npart, rng)
zc = rng.normal(0.0, p["sigma_t"], npart)
dE = (rng.normal(0.0, p["sigma_p"], npart) + p["mu_p"]) * beta * p0c


def make_bunch(nn):
    b = Bunch()
    b.mass(mass_GeV)
    b.charge(-1.0)
    b.getSyncParticle().kinEnergy(kin_GeV)
    for i in range(nn):
        if i % size == rank:
            b.addParticle(float(xx[0, i]), float(xx[1, i]), float(yy[0, i]),
                          float(yy[1, i]), float(zc[i]), float(dE[i]))
    b.macroSize(1.0)
    return b


def make_lattice():
    # PyORBIT requires unique node objects/names; the HTU lattice reuses elements
    # (L12, L23, FODO cells), so clone + uniquely rename each occurrence.
    lat = TEAPOT_Lattice("htu")
    for idx, node in enumerate(get_lattice("pyorbit", screens_as_markers=True)):
        nd = copy.deepcopy(node)
        nd.setName(f"{nd.getName()}_{idx}")
        lat.addNode(nd)
    lat.initialize()
    return lat


lattice = make_lattice()
lattice.trackBunch(make_bunch(min(2000, npart)))  # warm-up (NOT timed)

b = make_bunch(npart)
t0 = time.perf_counter_ns()
lattice.trackBunch(b)
dt = time.perf_counter_ns() - t0

ta = BunchTwissAnalysis()
ta.analyzeBunch(b)


def cc(i, j):
    return ta.getCorrelation(i, j)


obs = {
    "sigma_x": math.sqrt(max(cc(0, 0), 0.0)),
    "sigma_y": math.sqrt(max(cc(2, 2), 0.0)),
    "emit_x": math.sqrt(max(cc(0, 0) * cc(1, 1) - cc(0, 1) ** 2, 0.0)),
    "emit_y": math.sqrt(max(cc(2, 2) * cc(3, 3) - cc(2, 3) ** 2, 0.0)),
}
if rank == 0:
    print(f"Track: {dt}ns")
    print("Validate: " + json.dumps(obs))
