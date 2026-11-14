#!/usr/bin/env python3
"""Phase 15p: Cr3+ d3 RAS k = 1 with only one l raised: all virtual d (T2), more g (T1), both.

Ion via env ION (as phase15n; 15p uses cr only).
  dims [spaces]; prep <spaces>; run_names <names>; analyze <tag> <spaces>   (as phase15n)
Spaces (on top of 15n rasK1, rasK1_nof; same RAS rules, f content = the compact f as in rasK1):
  rasDall       virtuals s 1, p 1, d all (5), f 1, g 1
  rasDall_nof   the same without f
  rasDall_1d    rasDall with at most one electron in the virtual d (the work order's fallback; also over the limit)
  rasDall_c     rasDall cut: the diffuse d (13d, 12d, 11d) up to 2 electrons with 3s2 3p6 intact, at most 1 electron
                with one 3s/3p hole, none with two holes; 15d, 14d uncut (so rasK1 < rasK2_d < rasDall_c < rasDall)
  rasDall_c0    rasDall cut harder: the diffuse d only with 3s2 3p6 intact
  rasDall_c_nof rasDall_nof with the rasDall_c cut
  rasG3_c       rasG3 with 13g cut as in rasDall_c (12g uncut)
  rasG2 / rasG2_nof / rasG3   g 2 / g 2 without f / g 3 (g chosen by <r>), other l k = 1
  rasDallG2     all virtual d + g 2, with f
  dpathSD       d path only, frozen [Ne] vcm deck: 3s2 3p6 x 3d -> d SD into all 5 virtual d
  dpathSD_k1 / dpathSD_k2   the same with the RAS k = 1 d (15d) / k = 2 d (15d, 14d) only
                (the 15e d5 space is the full CI of the same orbitals; it is stored as hf+d5 in phase15k_item1.json)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15n as N15  # noqa: E402
import phase15m as M  # noqa: E402
import phase15k as K  # noqa: E402

ION, N_, TAG_ = M.ION, M.N_, M.TAG_
VARIANTS = {'rasDall': {'s': 1, 'p': 1, 'd': 5, 'f': 1, 'g': 1},
            'rasDall_nof': {'s': 1, 'p': 1, 'd': 5, 'g': 1},
            'rasG2': {'s': 1, 'p': 1, 'd': 1, 'f': 1, 'g': 2},
            'rasG2_nof': {'s': 1, 'p': 1, 'd': 1, 'g': 2},
            'rasG3': {'s': 1, 'p': 1, 'd': 1, 'f': 1, 'g': 3},
            'rasDallG2': {'s': 1, 'p': 1, 'd': 5, 'f': 1, 'g': 2}}
DV = [f'{k}d' for k in range(11, 16)]
DIF = ['13d', '12d', '11d']        # the diffuse virtual d of cr15n3 (<r> 3.1-5.6 bohr; 14d 1.8, 15d 1.0)
CUTS = {'rasDall_c': ('rasDall', DIF, 1), 'rasDall_c0': ('rasDall', DIF, 0), 'rasDall_c_nof': ('rasDall_nof', DIF, 1),
        'rasG3_c': ('rasG3', ['13g'], 1)}


def holes(c):
    d = dict(c)
    return (2 - d.get('3s', 0)) + (6 - d.get('3p', 0))


def cut(cf, orbs, one_hole):
    """Keep configurations without electrons in orbs, or with 3s2 3p6 intact, or (one_hole) with one 3s/3p hole and
    at most one electron in orbs."""
    def n(c):
        return sum(q for o, q in c if o in orbs)
    return [c for c in cf if n(c) == 0 or holes(c) == 0 or (one_hole and holes(c) == 1 and n(c) <= 1)]

_sp0 = N15.spaces


def spaces():
    sp = _sp0()
    for k, v in VARIANTS.items():
        if k not in sp:
            sp[k] = ('allcore', N15.ras_cfgs(N15.virt_k(v)))
    if 'rasDall_1d' not in sp:
        kind, cf = sp['rasDall']
        sp['rasDall_1d'] = (kind, [c for c in cf if sum(q for o, q in c if o in DV) <= 1])
    for k, (base, orbs, one) in CUTS.items():
        if k not in sp:
            sp[k] = (sp[base][0], cut(sp[base][1], orbs, one))
    if 'dpathSD' not in sp and N_ == 3:
        import phase15e as E15
        for name, dv in (('dpathSD', DV), ('dpathSD_k1', ['15d']), ('dpathSD_k2', ['15d', '14d'])):
            cf = [d for d in E15.fci(N_, ['3d'] + dv) if d.get('3d', 0) >= N_ - 2]
            cf.sort(key=lambda d: d != {'3d': N_})            # P first
            sp[name] = ('vcm', [E15.full_cfg('vcm', d) for d in cf])
    return sp


N15.spaces = spaces
_rn0 = N15.rname


def rname(space, t):
    if space in VARIANTS or space in CUTS or space in ('rasDall_1d', 'dpathSD', 'dpathSD_k1', 'dpathSD_k2'):
        return f'v15p_{space}_d{N_}_{t}e_{TAG_}'
    return _rn0(space, t)


N15.rname = rname
K.rname_k = rname


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dims':
        N15.dims(sys.argv[2:])
    elif cmd == 'prep':
        N15.prep(sys.argv[2:])
    elif cmd == 'run_names':
        M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        N15.analyze(sys.argv[2], sys.argv[3:])
    elif cmd == 'virt':
        print(ION, {k: N15.virt_k(v) for k, v in VARIANTS.items()})
