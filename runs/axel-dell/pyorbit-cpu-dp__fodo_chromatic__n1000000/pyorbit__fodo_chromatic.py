#!/usr/bin/env python3
# Auto-generated benchmark run script: PyORBIT3 / fodo_chromatic (QuadTEAPOT + DriftTEAPOT; native chromatic-paraxial).
# PyORBIT's quad is the TEAPOT drift-kick integrator (no closed-form exact thick quad);
# it converges to the exact thick CHROMATIC quad as nParts grows -- we use a moderate
# nParts so it matches the other codes' exact-thick-quad map within tolerance. Drift is
# the TEAPOT paraxial drift (negligible at these angles). Coordinates (x, xp, y, yp, z, dE).
import json
import math
import time

import numpy as np

from orbit.core.bunch import Bunch, BunchTwissAnalysis
from orbit.teapot import TEAPOT_Lattice, DriftTEAPOT, QuadTEAPOT
from orbit.core.orbit_mpi import mpi_comm, MPI_Comm_rank, MPI_Comm_size

from scenarios._obs import gaussian_twiss_plane

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 100.0, 'emit_x': 1e-09, 'emit_y': 1e-09, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.01, 'quad_length': 0.1, 'drift_length': 0.5, 'k1': 2.0}
npart = 1000000
NSTEP = 2  # TEAPOT integration steps per quad -- minimal converged (self-converges by 2; was 8)

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
dE = rng.normal(0.0, p["sigma_p"], npart) * beta * p0c


def make_bunch(n):
    b = Bunch()
    b.mass(mass_GeV)
    b.charge(-1.0)
    b.getSyncParticle().kinEnergy(kin_GeV)
    for i in range(n):
        if i % size == rank:
            b.addParticle(float(xx[0, i]), float(xx[1, i]), float(yy[0, i]),
                          float(yy[1, i]), float(zc[i]), float(dE[i]))
    b.macroSize(1.0)
    return b


def quad(name, k1):
    q = QuadTEAPOT(name)
    q.setLength(p["quad_length"])
    q.setParam("kq", k1)
    q.setnParts(NSTEP)
    return q


def drift(name):
    d = DriftTEAPOT(name)
    d.setLength(p["drift_length"])
    d.setnParts(1)
    return d


lattice = TEAPOT_Lattice("fodo")
for node in (quad("QF", p["k1"]), drift("d1"), quad("QD", -p["k1"]), drift("d2")):
    lattice.addNode(node)
lattice.initialize()

lattice.trackBunch(make_bunch(min(2000, npart)))  # warm-up (NOT timed)

b = make_bunch(npart)
t0 = time.perf_counter_ns()
lattice.trackBunch(b)
dt = time.perf_counter_ns() - t0

ta = BunchTwissAnalysis()
ta.analyzeBunch(b)


def cc(i, j):
    return ta.getCorrelation(i, j)


sx = math.sqrt(max(cc(0, 0), 0.0))
sy = math.sqrt(max(cc(2, 2), 0.0))
ex = math.sqrt(max(cc(0, 0) * cc(1, 1) - cc(0, 1) ** 2, 0.0))
ey = math.sqrt(max(cc(2, 2) * cc(3, 3) - cc(2, 3) ** 2, 0.0))

if rank == 0:
    print(f"Track: {dt}ns")
    print("Validate: " + json.dumps(
        {"sigma_x": sx, "sigma_y": sy, "emit_x": ex, "emit_y": ey}))
