"""Exact space-charge kick of a 3D Gaussian bunch: the reference for the ``spacecharge`` scenario.

No PIC and no sampling, so the reference has neither grid nor shot noise and is independent of every
code under test. The rest-frame potential of a Gaussian charge ``Q`` with RMS sizes ``(sx, sy, sz)``,
with ``sz = gamma * sz_lab``, is, with ``K = Q / (4 pi eps0)``,

    phi(x, y, z) = K / sqrt(pi) * Int_0^inf exp(-x^2/(2 sx^2 + u) - y^2/(2 sy^2 + u) - z^2/(2 sz^2 + u))
                                            / sqrt((2 sx^2 + u) (2 sy^2 + u) (2 sz^2 + u)) du

so ``E_x = -d(phi)/dx`` is a 1D integral. Averaging over the Gaussian beam analytically leaves

    <x E_x> = K * (2 sx^2 / sqrt(pi)) * Int_0^inf du / ((4 sx^2 + u)^(3/2) (4 sy^2 + u)^(1/2) (4 sz^2 + u)^(1/2))
    <E_x^2> = a double integral over (u, v) with a closed-form integrand (see ``ex2``)

Applied once over a length ``L``, the rest-frame field gives the lab-frame kicks

    d(px/p0)    = E_x * L / (gamma^2 beta^2 mc2)        [mc2 in eV]
    d(E)/(p0 c) = E_z * L / (gamma beta mc2)

Self-check, independent of the benchmarked codes: ``python -m scenarios.spacecharge.analytic``.
"""

from __future__ import annotations

import math

import numpy as np

EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m] (CODATA 2018)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(600)


def _log_nodes(sizes, decades_below=14.0, decades_above=10.0):
    """Nodes and weights for ``Int_0^inf f(u) du`` on a log grid in ``u`` (substitution u = exp(v))."""
    s2 = np.array(sizes, dtype=float) ** 2
    lo = math.log(s2.min()) - decades_below * math.log(10)
    hi = math.log(s2.max()) + decades_above * math.log(10)
    v = 0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo)
    u = np.exp(v)
    return u, 0.5 * (hi - lo) * _GL_W * u


def x_ex(sx, sy, sz, q):
    """``<x E_x>`` [V] for a Gaussian of charge ``q`` [C] with rest-frame sizes ``(sx, sy, sz)`` [m]."""
    k = q / (4.0 * math.pi * EPS0)
    u, w = _log_nodes((sx, sy, sz))
    f = (4 * sx**2 + u) ** -1.5 * (4 * sy**2 + u) ** -0.5 * (4 * sz**2 + u) ** -0.5
    return k * 2.0 * sx**2 / math.sqrt(math.pi) * float(np.sum(w * f))


def ex2(sx, sy, sz, q):
    """``<E_x^2>`` [(V/m)^2]: the Gaussian average is exact, then a 2D quadrature over ``(u, v)``."""
    k = q / (4.0 * math.pi * EPS0)
    u, wu = _log_nodes((sx, sy, sz))
    uu, vv = np.meshgrid(u, u, indexing="ij")
    weights = np.outer(wu, wu)

    def alpha(s):  # 1/(2 s^2 + u) + 1/(2 s^2 + v)
        return 1.0 / (2 * s**2 + uu) + 1.0 / (2 * s**2 + vv)

    def norm(s):  # 1 / sqrt((2 s^2 + u) (2 s^2 + v))
        return ((2 * s**2 + uu) * (2 * s**2 + vv)) ** -0.5

    # <x^2 exp(-alpha x^2)> = s^2 (1 + 2 alpha s^2)^(-3/2); <exp(-alpha y^2)> = (1 + 2 alpha s^2)^(-1/2)
    avg = (sx**2 * (1 + 2 * alpha(sx) * sx**2) ** -1.5
           * (1 + 2 * alpha(sy) * sy**2) ** -0.5
           * (1 + 2 * alpha(sz) * sz**2) ** -0.5)
    integrand = 4.0 / (2 * sx**2 + uu) / (2 * sx**2 + vv) * norm(sx) * norm(sy) * norm(sz) * avg
    return k**2 / math.pi * float(np.sum(weights * integrand))


def kick(*, q, ekin_mev, mc2_mev, sx, sy, sz_lab, length):
    """RMS kicks and correlations of one kick over ``length`` on a beam with no initial momentum spread.

    Returns ``xpx``, ``ypy`` [m rad] and the RMS kicks ``fx``, ``fy`` [rad] and ``fe`` (dE / (p0 c)).
    """
    gamma = 1.0 + ekin_mev / mc2_mev
    betagamma = math.sqrt(gamma * gamma - 1.0)
    beta = betagamma / gamma
    mc2 = mc2_mev * 1e6
    sz = gamma * sz_lab
    transverse = length / (gamma**2 * beta**2 * mc2)
    longitudinal = length / (gamma * beta * mc2)
    return {
        "xpx": transverse * x_ex(sx, sy, sz, q),
        "ypy": transverse * x_ex(sy, sx, sz, q),
        "fx": transverse * math.sqrt(ex2(sx, sy, sz, q)),
        "fy": transverse * math.sqrt(ex2(sy, sx, sz, q)),
        "fe": longitudinal * math.sqrt(ex2(sz, sx, sy, q)),
    }


def _ex_pointwise(x, y, z, sx, sy, sz, q):
    """``E_x`` at sample points by 1D quadrature; only used to cross-check the averaged forms."""
    k = q / (4.0 * math.pi * EPS0)
    u, w = _log_nodes((sx, sy, sz))
    u, w = u[::3], w[::3] * 3
    ax, ay, az = 2 * sx**2 + u, 2 * sy**2 + u, 2 * sz**2 + u
    weights = w / (ax**1.5 * ay**0.5 * az**0.5)
    x2, y2, z2 = x[:, None] ** 2, y[:, None] ** 2, z[:, None] ** 2
    return k * 2.0 * x / math.sqrt(math.pi) * np.sum(weights * np.exp(-x2 / ax - y2 / ay - z2 / az), axis=1)


def _self_check() -> int:
    """Compare the reference against independent results; returns the number of failed checks."""
    failed = 0

    def check(name, value, expected, tol):
        nonlocal failed
        error = value / expected - 1.0
        ok = abs(error) <= tol
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {value:.8g} vs {expected:.8g} ({error:+.2e}, tol {tol:.0e})")

    s, q = 1e-3, 1e-9
    k = q / (4 * math.pi * EPS0)
    # Sphere: <r.E> equals the Gaussian self-energy result K / (2 sqrt(pi) s), so <x E_x> is a third of it.
    check("sphere <x E_x> vs self-energy", x_ex(s, s, s, q), k / (6 * math.sqrt(math.pi) * s), 1e-10)
    # Long round bunch: the field tends to that of a line charge, <x E_x> -> K / (4 sqrt(pi) sz).
    sz = 1e4 * s
    check("long round bunch <x E_x> vs line charge", x_ex(s, s, sz, q), k / (4 * math.sqrt(math.pi) * sz), 1e-3)
    # Triaxial: brute-force Monte Carlo over the pointwise field. The tolerance covers its sampling noise.
    rng = np.random.default_rng(1)
    sx, sy, sz = 1e-3, 1e-4, 2e-2
    n = 400_000
    x, y, z = rng.normal(0, sx, n), rng.normal(0, sy, n), rng.normal(0, sz, n)
    field = np.concatenate([_ex_pointwise(x[i:i + 20_000], y[i:i + 20_000], z[i:i + 20_000], sx, sy, sz, q)
                            for i in range(0, n, 20_000)])
    check("triaxial <x E_x> vs Monte Carlo", x_ex(sx, sy, sz, q), float(np.mean(x * field)), 5e-3)
    check("triaxial rms E_x vs Monte Carlo", math.sqrt(ex2(sx, sy, sz, q)), float(np.sqrt(np.mean(field**2))), 5e-3)
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _self_check() else 0)
