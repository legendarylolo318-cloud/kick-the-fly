#!/usr/bin/env python3
"""Regenerate crates/fly_sim/src/ziggurat_tables.rs from the constants compiled into the installed NumPy.

NumPy's ziggurat tables (random/src/distributions/ziggurat_constants.h) are computed offline with more precision than the textbook
constants give, so they cannot be rebuilt bit for bit from the algorithm in f64 (the float32 `ki` table needs rounding, the float64
tables differ at 1e-10). The tables are plain data, so they are read out of the shared library and checked in:

    python crates/fly_sim/tools/extract_ziggurat.py > crates/fly_sim/src/ziggurat_tables.rs

The search anchors on values rebuilt from Marsaglia & Tsang's algorithm, then takes the 256-entry arrays that sit next to them.
"""
import math
import os
import struct
import sys

import numpy as np

R = 3.6541528853610088
V = 0.00492867323399
x = [0.0] * 256
f = [0.0] * 256
q = V / math.exp(-0.5 * R * R)
x[0], x[255], f[0], f[255] = q, R, 1.0, math.exp(-0.5 * R * R)
dn = R
for i in range(254, 0, -1):
    dn = math.sqrt(-2 * math.log(V / dn + math.exp(-0.5 * dn * dn)))
    x[i], f[i] = dn, math.exp(-0.5 * dn * dn)

lib = os.path.join(os.path.dirname(np.random.__file__), [n for n in os.listdir(os.path.dirname(np.random.__file__)) if n.startswith("_generator") and n.endswith(".so")][0])
data = open(lib, "rb").read()


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def find_f32(vals):
    i = data.find(struct.pack("<3f", *vals))
    assert i >= 0, "float32 table not found"
    return i


fi_f_at = find_f32([f32(v) for v in f[:3]])
wi_f_at = find_f32([f32(v / 8388608.0) for v in x[:3]])
ki_f_at = wi_f_at + 1024
ki_f = struct.unpack("<256I", data[ki_f_at:ki_f_at + 1024])
assert ki_f[1] == 0 and abs(ki_f[0] - int(round((R / q) * 8388608.0))) < 2

arr = np.frombuffer(data[: len(data) // 8 * 8], dtype="<f8")
u = np.frombuffer(data[: len(data) // 8 * 8], dtype="<u8")


def find_f64(vals):
    m = np.isclose(arr[:-3], vals[0], rtol=1e-9, atol=0) & np.isclose(arr[1:-2], vals[1], rtol=1e-9, atol=0) & np.isclose(arr[2:-1], vals[2], rtol=1e-9, atol=0)
    idx = np.flatnonzero(m)
    assert len(idx) >= 1, "float64 table not found"
    return int(idx[0])


fi_d_at = find_f64(f[:3])
wi_d_at = find_f64([v / 4503599627370496.0 for v in x[:3]])
ki0 = int((R / q) * 4503599627370496.0)
cand = np.flatnonzero((np.abs(u[:-3].astype(np.int64) - ki0) < 1 << 20) & (u[1:-2] == 0))
ki_d_at = int(cand[0])


def emit(name, ty, values, fmt):
    print(f"pub static {name}: [{ty}; 256] = [")
    for k in range(0, 256, 4):
        print("    " + ", ".join(fmt(v) for v in values[k:k + 4]) + ",")
    print("];\n")


print("//! NumPy's ziggurat tables, read out of the installed NumPy by tools/extract_ziggurat.py (bit patterns, so nothing is rounded twice).")
print("//! Generated file: do not edit.\n")
emit("KI_F", "u32", list(ki_f), lambda v: f"{v:#010x}")
emit("WI_F", "u32", list(struct.unpack("<256I", data[wi_f_at:wi_f_at + 1024])), lambda v: f"{v:#010x}")
emit("FI_F", "u32", list(struct.unpack("<256I", data[fi_f_at:fi_f_at + 1024])), lambda v: f"{v:#010x}")
emit("KI_D", "u64", [int(v) for v in u[ki_d_at:ki_d_at + 256]], lambda v: f"{v:#018x}")
emit("WI_D", "u64", [int(v) for v in u[wi_d_at:wi_d_at + 256]], lambda v: f"{v:#018x}")
emit("FI_D", "u64", [int(v) for v in u[fi_d_at:fi_d_at + 256]], lambda v: f"{v:#018x}")
print(f"// numpy {np.__version__}", file=sys.stderr)
