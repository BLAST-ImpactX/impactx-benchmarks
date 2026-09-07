"""Parse IMPACT-Z ``fort.*`` statistics files into harness observables.

IMPACT-Z (z as the independent variable) writes per-step beam moments to Fortran
unit files. We read the LAST data row of the transverse/longitudinal statistics
files, which is the beam state at the end of the lattice:

* ``fort.24`` -- X plane: ``z, <x>[m], sigma_x[m], <x'>[rad], sigma_x'[rad], alpha_x, eps_nx[m.rad]``
* ``fort.25`` -- Y plane: same columns for Y
* ``fort.26`` -- Z plane: ``z, <z>[deg], sigma_z[deg], <dE>[MeV], sigma_dE[MeV], alpha_z, eps_z[deg.MeV]``

(Column definitions verified in ``src/Contrl/Output.f90:341-352``; scaling constants
in ``src/DataStruct/PhysConst.f90``.)

Unit reconciliation with ImpactX (the reference), whose ``reduced_beam_characteristics``
reports GEOMETRIC emittance and physical RMS sizes in metres:

* ``sigma_x/sigma_y``  -- fort.24/25 col 3, already in metres. Direct.
* ``emit_x/emit_y``    -- fort.24/25 col 7 is the *normalised* RMS emittance
  (the momentum column is gamma*beta*x'); divide by (beta*gamma) to get the
  geometric emittance ImpactX reports. (Verified numerically against Example1.)
* ``sigma_t``          -- fort.26 col 3 is the RMS bunch length in DEGREES of RF
  phase. Convert back to metres: sigma_t = sigma_phi[rad] * beta * xl, where
  xl = c/(2*pi*f_ref) is IMPACT-Z's length scale and f_ref is the reference RF
  frequency from the input deck. (Internal phase = z/(beta*xl).)

The beam gamma/beta/xl needed for these conversions are recovered from the input
deck's beam header line (kinetic energy, mass, reference frequency), so this module
needs only the run directory and that one header line -- no external state.
"""

from __future__ import annotations

import math
from pathlib import Path

C_LIGHT = 299792458.0


def _data_lines(path: Path):
    """Yield non-empty, non-comment lines of a fort.* file."""
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("!"):
                yield s


def last_row(path: Path) -> list[float]:
    """Return the last data row of a fort.* file as a list of floats.

    Fortran ``g0`` output can emit tokens like ``1.0E-3`` or (rarely, on overflow)
    ``*****``; a row with an unparseable token is skipped so we return the last
    *valid* numeric row.
    """
    last = None
    for s in _data_lines(Path(path)):
        toks = s.replace("D", "E").replace("d", "e").split()
        try:
            row = [float(t) for t in toks]
        except ValueError:
            continue
        if row:
            last = row
    if last is None:
        raise ValueError(f"no numeric data rows in {path}")
    return last


def beam_header(impactz_in: Path) -> dict:
    """Read the 11th non-comment line of ImpactZ.in (the beam header):

    ``Bcurrent  Bkin_energy[eV]  Bmass[eV]  Bcharge  Bfreq[Hz]  Bphase``

    Returns ``{gamma, beta, betagamma, xl, kin_energy_eV, mass_eV, freq_hz}``.
    """
    rows = list(_data_lines(Path(impactz_in)))
    if len(rows) < 11:
        raise ValueError(f"{impactz_in}: fewer than 11 header lines")
    toks = rows[10].replace("D", "E").replace("d", "e").split()
    kin_eV = float(toks[1])
    mass_eV = float(toks[2])
    freq = float(toks[4])
    gamma = 1.0 + kin_eV / mass_eV
    betagamma = math.sqrt(max(gamma * gamma - 1.0, 0.0))
    beta = betagamma / gamma if gamma > 0 else 0.0
    xl = C_LIGHT / (2.0 * math.pi * freq)
    return {
        "gamma": gamma, "beta": beta, "betagamma": betagamma, "xl": xl,
        "kin_energy_eV": kin_eV, "mass_eV": mass_eV, "freq_hz": freq,
    }


def observables(workdir: Path, impactz_in: Path) -> dict:
    """Final-state observables from fort.24/25/26, converted to ImpactX conventions.

    Returns ``{sigma_x, sigma_y, sigma_t, emit_x, emit_y}`` (SI: metres, m.rad).
    """
    workdir = Path(workdir)
    hdr = beam_header(Path(impactz_in))
    bg = hdr["betagamma"]
    beta = hdr["beta"]
    xl = hdr["xl"]

    fx = last_row(workdir / "fort.24")
    fy = last_row(workdir / "fort.25")
    fz = last_row(workdir / "fort.26")

    deg2rad = math.pi / 180.0
    return {
        "sigma_x": fx[2],                 # col 3: RMS x [m]
        "sigma_y": fy[2],                 # col 3: RMS y [m]
        "sigma_t": fz[2] * deg2rad * beta * xl,   # deg of phase -> metres
        "emit_x": fx[6] / bg,             # col 7: normalised -> geometric
        "emit_y": fy[6] / bg,
    }


if __name__ == "__main__":  # pragma: no cover - manual debugging
    import json
    import sys

    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    deck = Path(sys.argv[2]) if len(sys.argv) > 2 else wd / "ImpactZ.in"
    print(json.dumps(observables(wd, deck), indent=2))
