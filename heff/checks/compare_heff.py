#!/usr/bin/env python3
"""Compare two H_eff dumps (HEFF_DUMP npz: one matrix per level|term, Eh) and two phase15k result JSON.

  compare_heff.py A.npz B.npz [A.json B.json SPACE]

Prints max |dH_eff| over every element of every block (Eh), per level and overall, and T1..T4, B, C of both JSON with
their difference. Exit 1 if the keys or block shapes differ."""
import json
import sys

import numpy as np


def main():
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    if sorted(a.files) != sorted(b.files):
        print('keys differ', sorted(set(a.files) ^ set(b.files)))
        sys.exit(1)
    per = {}
    for k in a.files:
        if a[k].shape != b[k].shape:
            print('shape differs', k, a[k].shape, b[k].shape)
            sys.exit(1)
        lev = k.split('|')[0]
        per[lev] = max(per.get(lev, 0.0), float(np.abs(a[k] - b[k]).max()))
    n = sum(a[k].size for k in a.files)
    print(f'{len(a.files)} blocks, {n} elements; max |dH_eff| (Eh): ' + ', '.join(f'{k} {v:.3e}' for k, v in per.items())
          + f'; overall {max(per.values()):.3e}')
    if len(sys.argv) > 5:
        sp = sys.argv[5]
        ja, jb = (json.load(open(f))[sp]['exact']['proj'] for f in sys.argv[3:5])
        for k in ('T1', 'T2', 'T3', 'T4', 'B', 'C'):
            print(f'{k:3s} {ja[k]:+.9f} {jb[k]:+.9f} diff {ja[k] - jb[k]:+.2e}   printed {ja[k]:.3f} / {jb[k]:.3f}')


if __name__ == '__main__':
    main()
