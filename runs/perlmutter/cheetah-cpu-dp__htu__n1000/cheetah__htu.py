#!/usr/bin/env python3
# Auto-generated benchmark run script: cheetah / htu (BELLA HTU beamline tracking).
import json

import torch

from cheetah import ParticleBeam, Segment

from scenarios._obs import Timer, beam_observables
from scenarios.htu.htu_lattice import get_lattice

try:
    torch.set_num_interop_threads(1)  # match compute thread count
except RuntimeError:
    pass
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.set_default_device("cpu")
torch.set_float32_matmul_precision("highest")

p = {'mass_MeV': 0.510998950691753, 'mass_eV': 510998.9506917531, 'kin_energy_MeV': 99.48900104930824, 'kin_energy_eV': 99489001.04930824, 'total_energy_eV': 100000000.0, 'emit_x': 7.665084336342948e-09, 'emit_y': 7.665084336342948e-09, 'beta_x': 0.002, 'beta_y': 0.002, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 1e-06, 'sigma_p': 0.025, 'mu_p': 0.01, 'bunch_charge_C': 2.5e-11}
npart = 1000

segment = Segment(get_lattice("cheetah", screens_as_markers=True))
incoming = ParticleBeam.from_twiss(
    num_particles=1000,
    beta_x=torch.tensor(p["beta_x"]),
    alpha_x=torch.tensor(p["alpha_x"]),
    emittance_x=torch.tensor(p["emit_x"]),
    beta_y=torch.tensor(p["beta_y"]),
    alpha_y=torch.tensor(p["alpha_y"]),
    emittance_y=torch.tensor(p["emit_y"]),
    sigma_tau=torch.tensor(p["sigma_t"]),
    sigma_p=torch.tensor(p["sigma_p"]),
    energy=torch.tensor(p["total_energy_eV"]),
    total_charge=torch.tensor(p["bunch_charge_C"]),
)
incoming.p = incoming.p + p["mu_p"]
segment.to("cpu")
incoming.to("cpu")

segment.track(incoming=incoming)  # warm-up (NOT timed)
with Timer() as t:
    outgoing = segment.track(incoming=incoming)

obs = beam_observables(
    outgoing.x.detach().cpu().numpy(),
    outgoing.px.detach().cpu().numpy(),
    outgoing.y.detach().cpu().numpy(),
    outgoing.py.detach().cpu().numpy(),
)
print(f"Track: {t.ns}ns")
print("Validate: " + json.dumps(obs))
