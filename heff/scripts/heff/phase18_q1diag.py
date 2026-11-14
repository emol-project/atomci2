#!/usr/bin/env python3
"""Phase 18 (8 Nov, item 3): which Q2 sub-class, moved into Q1_rule, removes most of the Fe5+ T2 error (+0.293 mEh, Step 8).
Uses phase17c_scan.evaluate with C-e replaced by Q1_rule ('' = Q1_rule itself) and the Q2 classes refined:
  mx<l><l'>  3d^2 -> v v' with two different l (e.g. mxdg = one d and one g virtual)
  v3         three virtual electrons from 3d (3d^(N-3) v^3)
  otherwise  the phase17c_scan class (inner, pf2, 3s_v, 3p2_v, 3p_v, s, p, vd, f, g, h, rest)
A set is '+'-joined class names added to Q1_rule. Q1_rule itself is not changed.
usage: ION=fe5nd phase18_q1diag.py classes            (configurations per class, no H)
       ION=fe5nd phase18_q1diag.py eval <out.json> <set> ...  (needs the stored H, about 4.4 GB)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase17c_scan as S  # noqa: E402

C = S.C
LS = S.LS
_q0 = S.qclass


def qclass(label, N):
    c, vir, inner = C.occ(label)
    ls = sorted({k[-1] for k in vir}, key=LS.index)
    k = sum(vir.values())
    if inner and k == 2 and c == (2, 6, N - 2) and len(ls) == 2:
        return 'mx' + ''.join(ls)
    if inner and k == 3 and c == (2, 6, N - 3):
        return 'v3'
    return _q0(label, N)


def install(cfgs_all=None):
    q1s0 = C.q1_sets

    def q1_sets(cfgs):
        r = dict(q1s0(cfgs))
        r['C-e'] = sorted(S.rset('R', cfgs, C.N))
        return r
    C.q1_sets = q1_sets
    S.qclass = qclass


if __name__ == '__main__':
    install()
    if sys.argv[1] == 'classes':
        A = C.A
        bl, eav, hf, _ = A.blocks(os.environ.get('SPACE', 'besth_sall'))
        p0 = bl[0]['lab'][0]
        cfgs = sorted(set(eav) - {p0})
        R = set(S.rset('R', cfgs, C.N))
        from collections import Counter
        print('Q1_rule', len(R), 'Q2', len(cfgs) - len(R))
        print(sorted(Counter(qclass(c, C.N) for c in cfgs if c not in R).items(), key=lambda x: -x[1]))
    else:
        S.CLASSES[:] = sorted(set(S.CLASSES) | {x for s in sys.argv[3:] for x in s.split('+') if x})
        S.evaluate('besth_sall', sys.argv[2], [s if s != '-' else '' for s in sys.argv[3:]])
