#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# config benchmark
nturns = 200

# imports
import at
import numpy as np

# logic
Dr = at.Drift("Dr", 0.5)
HalfDr = at.Drift("Dr2", 0.25)
QF = at.Quadrupole("QF", 0.5, 1.2)
QD = at.Quadrupole("QD", 0.5, -1.2)
Bend = at.Dipole("Bend", 1, 2 * np.pi / 40)

FODOcell = at.Lattice(
    [HalfDr, Bend, Dr, QF, Dr, Bend, Dr, QD, HalfDr],
    name="Simple FODO cell",
    energy=1e9,
)
print(FODOcell)
FODO = FODOcell * 20
print(FODO)

#Z01 = np.array([0.001, 0, 0, 0, 0, 0])
#Z1, *_ = FODO.track(Z01, nturns)

[_, beamdata, _] = at.get_optics(FODO, get_chrom=True)

print(beamdata.tune)
print(beamdata.chromaticity)
