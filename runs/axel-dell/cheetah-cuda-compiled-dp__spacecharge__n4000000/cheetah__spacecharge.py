#!/usr/bin/env python3
# Auto-generated benchmark run script: cheetah / spacecharge (3D FFT space charge).
import json

import torch

import cheetah

from scenarios._obs import Timer, beam_observables

try:
    torch.set_num_interop_threads(1)  # match compute thread count
except RuntimeError:
    pass
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")
torch.set_float32_matmul_precision("highest")

p = {'mass_MeV': 0.51099895069, 'kin_energy_MeV': 250.0, 'bunch_charge_C': 1e-09, 'emit_x': 1e-07, 'emit_y': 1e-07, 'beta_x': 1.0, 'beta_y': 1.0, 'alpha_x': 0.0, 'alpha_y': 0.0, 'sigma_t': 0.001, 'sigma_p': 0.0001, 'drift_length': 6.0, 'n_cell': 64, 'grid_extent_sigma': 3.0}
npart = 4000000
n = int(p["n_cell"])
ext = float(p["grid_extent_sigma"])
ds = p["drift_length"]
total_energy_eV = (p["kin_energy_MeV"] + p["mass_MeV"]) * 1e6

incoming = cheetah.ParticleBeam.from_twiss(
    num_particles=4000000,
    beta_x=torch.tensor(p["beta_x"]),
    alpha_x=torch.tensor(p["alpha_x"]),
    emittance_x=torch.tensor(p["emit_x"]),
    beta_y=torch.tensor(p["beta_y"]),
    alpha_y=torch.tensor(p["alpha_y"]),
    emittance_y=torch.tensor(p["emit_y"]),
    sigma_tau=torch.tensor(p["sigma_t"]),
    sigma_p=torch.tensor(p["sigma_p"]),
    energy=torch.tensor(total_energy_eV),
    total_charge=torch.tensor(p["bunch_charge_C"]),
)

segment = cheetah.Segment(
    elements=[
        cheetah.Drift(length=torch.tensor(ds), tracking_method="drift_kick_drift"),
        cheetah.SpaceChargeKick(
            torch.tensor(ds),
            grid_shape=(n, n, n),
            grid_extent_x=torch.tensor(ext),
            grid_extent_y=torch.tensor(ext),
            grid_extent_tau=torch.tensor(ext),
        ),
    ]
)
segment.to("cuda")
incoming.to("cuda")

import torch._inductor.config as _ind
_fast = False
_ind.cpp.enable_unsafe_math_opt_flag = _fast
_ind.cpp.enable_floating_point_contract_flag = "fast"  # FMA fusion ON, matching gcc/nvcc default in every other code (apples-to-apples)
_ind.cuda.use_fast_math = _fast
_track = torch.compile(segment.track, backend="inductor")
# warm-up on the SAME beam that is timed (compile + stabilize dynamo guards);
# track is functional, so incoming is not mutated.
for _ in range(3):
    _track(incoming=incoming)
with Timer() as t:
    outgoing = _track(incoming=incoming)

obs = beam_observables(
    outgoing.x.detach().cpu().numpy(),
    outgoing.px.detach().cpu().numpy(),
    outgoing.y.detach().cpu().numpy(),
    outgoing.py.detach().cpu().numpy(),
    tau=outgoing.tau.detach().cpu().numpy(),
)

print(f"Track: {t.ns}ns")
print("Validate: " + json.dumps(obs))
