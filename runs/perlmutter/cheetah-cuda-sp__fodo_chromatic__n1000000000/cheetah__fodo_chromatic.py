#!/usr/bin/env python3
# Auto-generated benchmark run script: cheetah / fodo_chromatic.
# Cheetah has no chromatic-paraxial quad/drift, so it runs the EXACT (drift_kick_drift)
# map as a stand-in here -> marked with an asterisk (untuned model for this problem).
import json

import torch

import cheetah

from scenarios._obs import Timer, beam_observables

try:
    torch.set_num_interop_threads(1)  # match compute thread count
except RuntimeError:
    pass
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")
torch.set_float32_matmul_precision("highest")

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 100.0, 'emit_x': 1e-09, 'emit_y': 1e-09, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.01, 'quad_length': 0.1, 'drift_length': 0.5, 'k1': 2.0}
npart = 1000000000
total_energy_eV = (p["kin_energy_MeV"] + p["mass_MeV"]) * 1e6

incoming = cheetah.ParticleBeam.from_twiss(
    num_particles=1000000000,
    beta_x=torch.tensor(p["beta_x"]),
    alpha_x=torch.tensor(p["alpha_x"]),
    emittance_x=torch.tensor(p["emit_x"]),
    beta_y=torch.tensor(p["beta_y"]),
    alpha_y=torch.tensor(p["alpha_y"]),
    emittance_y=torch.tensor(p["emit_y"]),
    sigma_tau=torch.tensor(p["sigma_t"]),
    sigma_p=torch.tensor(p["sigma_p"]),
    energy=torch.tensor(total_energy_eV),
)

Lq = torch.tensor(p["quad_length"])
Ld = torch.tensor(p["drift_length"])
k1 = torch.tensor(p["k1"])
# chromatic symplectic tracking (renamed from "bmadx" in newer Cheetah), NOT the
# default purely-linear method, to match ImpactX/pyAT/Xsuite chromatic quad models
TM = "drift_kick_drift"
segment = cheetah.Segment(
    elements=[
        cheetah.Quadrupole(length=Lq, k1=k1, tracking_method=TM),
        cheetah.Drift(length=Ld, tracking_method=TM),
        cheetah.Quadrupole(length=Lq, k1=-k1, tracking_method=TM),
        cheetah.Drift(length=Ld, tracking_method=TM),
    ]
)
segment.to("cuda")
incoming.to("cuda")

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
