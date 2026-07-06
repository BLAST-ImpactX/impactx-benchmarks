"""htu_spin scenario: identical beam + lattice to htu, tracked with spin.

The beam starts fully spin-aligned (+z); the dispersive chicane differentially precesses
the spins, producing a measurable depolarization (RMS spin spread). Reuse the htu params.
"""
from scenarios.htu.params import PARAMS  # noqa: F401
