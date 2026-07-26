"""Minimal reader for Elegant SDDS output files (parameters and/or columns).

Why bespoke: Elegant's SDDS dialect isn't parseable by the PyPI/conda ``sdds`` package (pylhc,
MAD-X oriented), and Elegant's own ``sdds2stream`` needs the full SDDSaps link chain we don't build.
We only need a handful of scalar parameters (Sx, Sy, ex, ey from ``final``) and a few per-particle
columns (spx, spy, spz from the spin ``output``), so we parse the ASCII SDDS header and the binary
data stream directly.

SDDS binary page layout: int32 row_count, then each NON-fixed_value parameter in header order, then
column data. Column data is ROW-MAJOR by default (each row = all columns in definition order); the
column-major variant (``&data ... column_major=1``) is not emitted by Elegant tracking and is rejected.
Endianness comes from the ``!# little-endian``/``big-endian`` comment or the ``&data endian=...`` field.
"""
from __future__ import annotations

import re
import struct

# SDDS binary numeric type -> (struct code, byte size). SDDS `long` is 32-bit; `long64`/`ulong64` 64-bit.
_NUM = {
    "double": ("d", 8), "float": ("f", 4),
    "long64": ("q", 8), "ulong64": ("Q", 8),
    "long": ("i", 4), "ulong": ("I", 4),
    "short": ("h", 2), "ushort": ("H", 2),
    "char": ("b", 1), "character": ("b", 1),
}
_PARAM_RE = re.compile(rb"^&parameter\b(.*?)&end", re.S | re.M)
_COLUMN_RE = re.compile(rb"^&column\b(.*?)&end", re.S | re.M)
_FIELD_RE = re.compile(rb"(\w+)\s*=\s*(\"[^\"]*\"|[^,]+)")


def _fields(body: bytes) -> dict:
    return {m.group(1).decode(): m.group(2).decode().strip().strip('"')
            for m in _FIELD_RE.finditer(body)}


def _defs(regex, header: bytes) -> list:
    """Ordered [(name, type, is_fixed)] for &parameter/&column definitions."""
    out = []
    for m in regex.finditer(header):
        f = _fields(m.group(1))
        if "name" in f:
            out.append((f["name"], f.get("type", "double"), "fixed_value" in f))
    return out


def _open(path: str):
    """Return (raw_bytes, header, body_offset, endian, params, columns)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    data_m = re.search(rb"^&data\b.*?&end\s*?\n", raw, re.M | re.S)
    if not data_m:
        raise ValueError(f"{path}: no &data section")
    if b"column_major=1" in data_m.group(0):
        raise ValueError(f"{path}: column-major binary not supported")
    if b"mode=ascii" in data_m.group(0):
        raise ValueError(f"{path}: ASCII SDDS not supported (expected binary)")
    header = raw[: data_m.start()]
    big = b"big-endian" in header or b"endian=big" in data_m.group(0)
    endian = ">" if big else "<"
    return raw, header, data_m.end(), endian, _defs(_PARAM_RE, header), _defs(_COLUMN_RE, header)


def _read_value(raw, off, typ, endian):
    if typ == "string":
        (slen,) = struct.unpack_from(endian + "i", raw, off)
        return raw[off + 4: off + 4 + slen].decode("latin-1"), off + 4 + slen
    if typ not in _NUM:
        raise ValueError(f"unsupported SDDS type {typ!r}")
    code, size = _NUM[typ]
    (val,) = struct.unpack_from(endian + code, raw, off)
    return val, off + size


def read_parameters(path: str) -> dict:
    """Return ``{parameter_name: value}`` for the first page's scalar parameters."""
    raw, _, off, endian, params, _cols = _open(path)
    off += 4  # skip row_count
    values = {}
    for name, typ, is_fixed in params:
        if is_fixed:
            continue  # stored in the header, not the binary stream
        values[name], off = _read_value(raw, off, typ, endian)
    return values


def read_columns(path: str, want=None) -> dict:
    """Return ``{column_name: [values]}`` for the first page's columns (row-major binary).

    ``want`` optionally restricts to a set of column names (all are still decoded to advance the
    stream). Requires all columns numeric (Elegant's phase-space output has no string columns).
    """
    raw, _, off, endian, params, columns = _open(path)
    (nrows,) = struct.unpack_from(endian + "i", raw, off)
    off += 4
    for name, typ, is_fixed in params:  # skip parameters to reach the column block
        if is_fixed:
            continue
        _v, off = _read_value(raw, off, typ, endian)
    if any(t == "string" for _, t, _ in columns):
        raise ValueError(f"{path}: string columns not supported by the fast row reader")
    fmt = endian + "".join(_NUM[t][0] for _, t, _ in columns)
    names = [n for n, _, _ in columns]
    keep = set(names) if want is None else set(want)
    out = {n: [] for n in names if n in keep}
    block = raw[off: off + struct.calcsize(fmt) * nrows]
    for row in struct.iter_unpack(fmt, block):
        for n, v in zip(names, row):
            if n in keep:
                out[n].append(v)
    return out


if __name__ == "__main__":  # python read_sdds.py <file> [col1,col2]
    import sys

    if len(sys.argv) > 2:
        cols = read_columns(sys.argv[1], sys.argv[2].split(","))
        for k, v in cols.items():
            print(f"  {k}: n={len(v)} first={v[0]:.4e}")
    else:
        v = read_parameters(sys.argv[1])
        for k in ("Sx", "Sy", "ex", "ey", "Particles"):
            print(f"  {k:10s} = {v.get(k)}")
