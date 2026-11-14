#!/usr/bin/env python3
"""et_from_civ.py -- print the total energy ET stored by the CI stage.

usage: et_from_civ.py fort.37

The CI vector file (NFCIV, unit 37) is gfortran unformatted sequential;
its second record holds ET as real*8 at byte offset 8 (found by scanning the
M5 outputs: the same value is in the CI summary file, unit 71). The printed
output has 10 decimals for boron but only 8 for f, too few for the 1e-9
check of L5.
"""
import struct
import sys


def record(path, index):
    with open(path, "rb") as fh:
        data = fh.read()
    pos = 0
    for i in range(index + 1):
        (n,) = struct.unpack_from("<i", data, pos)
        rec = data[pos + 4:pos + 4 + n]
        (m,) = struct.unpack_from("<i", data, pos + 4 + n)
        if m != n:
            raise ValueError(f"{path}: bad record markers at {pos}")
        if i == index:
            return rec
        pos += 8 + n
    raise ValueError(f"{path}: fewer than {index + 1} records")


if __name__ == "__main__":
    (et,) = struct.unpack_from("<d", record(sys.argv[1], 1), 8)
    print(repr(et))
