#!/usr/bin/env python3
# Auto-generated benchmark run script: PyORBIT3 / spacecharge (3D PIC).
# SpaceChargeCalc3D: open-boundary (Hockney zero-padded) FFT Poisson on an n_cell^3 grid.
# NOTE vs ImpactX/Cheetah: PyORBIT's 3D solver uses a plain (point-sampled 1/r) Green
# function and quadratic (TSC) deposition, whereas ImpactX/Cheetah use the integrated
# Green function (IGF) with linear (CIC) deposition -- same open-BC FFT-PIC family, but
# the GF/deposition differ. One SC kick over the drift (as in the other codes).
import json
import math
import time

import numpy as np

from orbit.core.bunch import Bunch, BunchTwissAnalysis
from orbit.teapot import TEAPOT_Lattice, DriftTEAPOT
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.core.orbit_mpi import mpi_comm, MPI_Comm_rank, MPI_Comm_size

from scenarios._obs import gaussian_twiss_plane

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 250.0, 'bunch_charge_C': 1e-09, 'emit_x': 1e-07, 'emit_y': 1e-07, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.0001, 'drift_length': 6.0, 'n_cell': 64, 'grid_extent_sigma': 3.0}
npart = 10000
n = int(p["n_cell"])
ELEM_CHARGE = 1.602176634e-19

comm = mpi_comm.MPI_COMM_WORLD
rank = MPI_Comm_rank(comm)
size = MPI_Comm_size(comm)

mass_GeV = p["mass_MeV"] * 1e-3
kin_GeV = p["kin_energy_MeV"] * 1e-3
e_tot = kin_GeV + mass_GeV
p0c = math.sqrt(e_tot * e_tot - mass_GeV * mass_GeV)
beta = p0c / e_tot
macro_size = (p["bunch_charge_C"] / ELEM_CHARGE) / npart

rng = np.random.default_rng(12345)
xx = gaussian_twiss_plane(p["emit_x"], p["beta_x"], p["alpha_x"], npart, rng)
yy = gaussian_twiss_plane(p["emit_y"], p["beta_y"], p["alpha_y"], npart, rng)
zc = rng.normal(0.0, p["sigma_t"], npart)
dE = rng.normal(0.0, p["sigma_p"], npart) * beta * p0c


def make_bunch(nn):
    b = Bunch()
    b.mass(mass_GeV)
    b.charge(-1.0)
    b.getSyncParticle().kinEnergy(kin_GeV)
    for i in range(nn):
        if i % size == rank:
            b.addParticle(float(xx[0, i]), float(xx[1, i]), float(yy[0, i]),
                          float(yy[1, i]), float(zc[i]), float(dE[i]))
    b.macroSize(macro_size)
    return b


def make_lattice():
    lat = TEAPOT_Lattice("sc")
    d = DriftTEAPOT("d1")
    d.setLength(p["drift_length"])
    d.setnParts(1)  # one SC kick over the drift
    lat.addNode(d)
    lat.initialize()
    calc = SpaceChargeCalc3D(n, n, n)
    setSC3DAccNodes(lat, p["drift_length"], calc)  # min spacing >= ds -> single SC node
    return lat


lattice = make_lattice()
lattice.trackBunch(make_bunch(min(2000, npart)))  # warm-up (FFT plan etc.), NOT timed

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
    "sigma_t": math.sqrt(max(cc(4, 4), 0.0)),
    "emit_x": math.sqrt(max(cc(0, 0) * cc(1, 1) - cc(0, 1) ** 2, 0.0)),
    "emit_y": math.sqrt(max(cc(2, 2) * cc(3, 3) - cc(2, 3) ** 2, 0.0)),
}

if rank == 0:
    print(f"Track: {dt}ns")
    print("Validate: " + json.dumps(obs))
