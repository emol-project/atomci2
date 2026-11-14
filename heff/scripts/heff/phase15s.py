#!/usr/bin/env python3
"""Phase 15s: best space for V2+, Mn4+ (and Fe2+ d6 if it fits); RAS-limit sensitivity for Cr3+.

Ion via env ION (cr | v | mn | fe). Same route as 15q/15r (stop-after-H deck, iterative solver).
  rasDallG2_f2, rasDallG2_f0   best space (all virtual d, g 2, f 2 / no f, s p 1), 15r definitions
  rasK1_v3   RAS k = 1 with up to 3 electrons in the virtuals; a configuration with 3 virtual electrons may be a
             triple from the references (it cannot be less), all other rules unchanged
  rasK1_h3   RAS k = 1 with up to 3 holes in 3s3p (admits 3s 3p4 3d^(N+3), a double from 3s 3p6 3d^(N+1))
Run names v15s_<space>_d<N>_<t>e_<tag>. Commands: dims / prep / run_names / analyze / check."""
import collections
import itertools as it
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15r as R15  # noqa: E402  (15q route + rasDallG2_f* spaces)
import phase15q as Q  # noqa: E402
import phase15n as N15  # noqa: E402

N_ = Q.N_


def ras_cfgs_x(virt, maxh=2, maxv=2):
    """N15.ras_cfgs with the hole limit in 3s3p (maxh) and the virtual-electron limit (maxv) as parameters.
    Excitation level from the references stays <= 2, except <= 3 for configurations with 3 virtual electrons."""
    import phase15e as E15
    lof = {o: 'spdfghik'.index(o[-1]) for o in virt + ['3s', '3p', '3d']}
    refs = [{'3s': 2, '3p': 6, '3d': N_}, {'3s': 1, '3p': 6, '3d': N_ + 1}]
    nel = 8 + N_
    rv = []
    for n in range(maxv + 1):
        for comb in it.combinations_with_replacement(virt, n):
            v = dict(collections.Counter(comb))
            if all(q <= 2 * (2 * lof[o] + 1) for o, q in v.items()):
                rv.append(v)
    out = []
    for ns in range(0, 3):
        for np_ in range(0, 7):
            if (2 - ns) + (6 - np_) > maxh:
                continue
            for v in rv:
                nd = nel - ns - np_ - sum(v.values())
                if nd < 0 or nd > 10:
                    continue
                c = {'3s': ns, '3p': np_, '3d': nd, **v}
                if sum(lof[o] * q for o, q in c.items()) % 2:
                    continue
                lim = max(2, sum(v.values()))
                if min(sum(max(0, r.get(o, 0) - c.get(o, 0)) for o in set(r) | set(c)) for r in refs) <= lim:
                    out.append(c)
    P0 = {'3s': 2, '3p': 6, '3d': N_}
    out.sort(key=lambda c: c != P0)
    cf = [E15.full_cfg('all', c) for c in out]
    c2s = list(E15.core_spaces(N_)['c2s'])
    return cf + [c for c in c2s if c not in cf]


if Q.ION == 'crnd':          # addendum: plain Sapporo-QZP-2012 (9s7p5d3f2g1h); keep only the virtuals this basis has
    import gen_v
    import phase15j as J
    _nbf = Q.K.H.F.Bas(Q.M.BAS_, Q.M.Z_).nbf
    for _l, _lst in J.VIRT.items():
        _lst[:] = [o for o in _lst if gen_v.vec(o) <= _nbf['spdfgh'.index(_l)]]
Q.NEW['rasDallG2_fall'] = {'s': 1, 'p': 1, 'd': 5, 'f': 4, 'g': 2}   # all virtual d and f the basis has (addendum)
K1 = {l: 1 for l in 'spdfg'}
STEP3 = {'rasK1_v3': dict(maxv=3), 'rasK1_h3': dict(maxh=3)}
S1 = ['rasDallG2_f2', 'rasDallG2_f0', 'rasDallG2_fall']
_sp0 = N15.spaces


def dp_sets():
    """Addendum d path (15p dpathSD rule: 3d -> virtual d, 3d >= N - 2) in all-electron decks: all virtual d (dpA),
    each one left out (dpX_<o>), each one alone (dp1_<o>)."""
    import phase15j as J
    dv = sorted(J.VIRT['d'], key=N15.r_virt().get)       # compact first
    return {'dpA': dv, **{f'dpX_{o}': [x for x in dv if x != o] for o in dv}, **{f'dp1_{o}': [o] for o in dv}}


def spaces():
    sp = _sp0()
    for k, kw in STEP3.items():
        if k not in sp:
            sp[k] = ('allcore', ras_cfgs_x(N15.virt_k(K1), **kw))
    if N_ == 3 and 'dpA' not in sp:
        import phase15e as E15
        for k, dv in dp_sets().items():
            cf = [d for d in E15.fci(N_, ['3d'] + dv) if d.get('3d', 0) >= N_ - 2]
            cf.sort(key=lambda d: d != {'3d': N_})        # P first
            sp[k] = ('allcore', [E15.full_cfg('all', d) for d in cf])
    return sp


N15.spaces = spaces
Q.spaces = spaces            # Q.dims looks up its module-level spaces()
_rn = N15.rname


def rname(space, t):
    if space in S1 or space in STEP3 or space.startswith('dp'):
        return f'v15s_{space}_d{N_}_{t}e_{Q.TAG_}'
    return _rn(space, t)


N15.rname = rname
Q.K.rname_k = rname

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'check':      # ras_cfgs_x with the default limits must reproduce the 15n rules exactly
        for kmap in (K1, R15.NEW['rasDallG2_f2']):
            virt = N15.virt_k(kmap)
            assert ras_cfgs_x(virt) == N15.ras_cfgs(virt), kmap
        print('ras_cfgs_x(maxh=2, maxv=2) == phase15n.ras_cfgs for rasK1 and rasDallG2_f2:', Q.ION)
    elif cmd == 'dims':
        Q.dims(sys.argv[2:], 'phase15s_dims.json')
    elif cmd == 'prep':
        N15.prep(sys.argv[2:])
    elif cmd == 'run_names':
        Q.M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        N15.analyze(sys.argv[2], sys.argv[3:])
