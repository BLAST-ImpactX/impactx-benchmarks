#!/usr/bin/env python3
#
# see https://github.com/desy-ml/cheetah/blob/master/tests/test_space_charge_kick.py
#
# -*- coding: utf-8 -*-

import time
import torch
from scipy import constants
from scipy.constants import physical_constants
from torch import nn

import cheetah

"""
Tests that that a cold uniform beam doubles in size in both dimensions when
travelling through a drift section with space_charge. (cf ImpactX test:
https://impactx.readthedocs.io/en/latest/usage/examples/cfchannel/README.html#constant-focusing-channel-with-space-charge)
See Free Expansion of a Cold Uniform Bunch in
https://accelconf.web.cern.ch/hb2023/papers/thbp44.pdf.
"""

# Simulation parameters
R0 = torch.tensor(0.001)
energy = torch.tensor(2.5e8)
rest_energy = torch.tensor(
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge
)
elementary_charge = torch.tensor(constants.elementary_charge)
electron_radius = torch.tensor(physical_constants["classical electron radius"][0])
gamma = energy / rest_energy
beta = torch.sqrt(1 - 1 / gamma**2)
npart = 100_000

incoming = cheetah.ParticleBeam.uniform_3d_ellipsoid(
    num_particles=torch.tensor(npart),
    total_charge=torch.tensor(1e-8),
    energy=energy,
    radius_x=R0,
    radius_y=R0,
    radius_tau=R0 / gamma,  # Radius of the beam in s direction in the lab frame
    sigma_px=torch.tensor(1e-15),
    sigma_py=torch.tensor(1e-15),
    sigma_p=torch.tensor(1e-15),
)

# Compute section length
nslice = torch.tensor(40)
ds = torch.tensor(6.0)
section_length = ds / nslice

segment = cheetah.Segment(
    elements=[
        cheetah.Drift(section_length),
        cheetah.SpaceChargeKick(section_length),
    ] * nslice
)

start_time = time.perf_counter()
outgoing = segment.track(incoming)
end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution time Cheetah: {execution_time:.4f} seconds ({execution_time/npart*1e6}us / particle)")

#assert torch.isclose(outgoing.sigma_x, 2 * incoming.sigma_x, rtol=2e-2)
#assert torch.isclose(outgoing.sigma_y, 2 * incoming.sigma_y, rtol=2e-2)
#assert torch.isclose(outgoing.sigma_tau, 2 * incoming.sigma_tau, rtol=2e-2)
