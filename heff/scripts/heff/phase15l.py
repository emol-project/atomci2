#!/usr/bin/env python3
"""Phase 15l: Cr3+ d3 -- virtual f space and core-core correlation as candidates for the T1 excess.

Same basis (cr.bas), orbitals (cr15n3 = orbsets_15a '3|l') and P space as 15k.
  step0                 virtual f diagnostics (<r>, h eigenvalue, R^k with 3p, 3d)       -> phase15l_step0.json
  dims [spaces]         CSF counts per (L,S) block and time estimates                    -> phase15l_dims.json
  prep <names>          decks
  run_names <names>     ATOMCI runs (one per call)
  analyze <tag> <sp..>  15k analyze3 (one dense block at a time)                         -> phase15l_<tag>.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15k as K  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402

OUT = K.OUT


def orbital(bs, z, o):
    import gen_v
    L = 'spdfgh'.index(o[-1])
    c = z[f'3|{L}'][:, gen_v.vec(o) - 1]
    return L, c, bs.P(L, c)


def step0():
    import phase15j as J
    bs = K.H.F.Bas(K.BAS3, K.Z3)
    z = np.load(os.path.join(OUT, 'orbsets_15a.npz'))
    P, rows = {}, []
    for o in ['3p', '3d'] + J.VIRT['f']:
        L, c, p = orbital(bs, z, o)
        sgn = np.sign(p[np.argmax(np.abs(p))])
        P[o] = p * sgn
        rows.append((o, L, c))

    def rk(a, b, c, d, k):
        # R^k(ab; cd) = int int Pa(1) Pb(2) r<^k / r>^(k+1) Pc(1) Pd(2)
        return float(I.slater(P[a], P[c], P[b], P[d], k, bs.x, bs.r))

    res = {'3p': {}, '3d': {}, 'f': []}
    for o, L, c in rows:
        r_ = float(np.sum(P[o] ** 2 * bs.r * bs.w))
        nrm = float(np.sum(P[o] ** 2 * bs.w))
        if o in ('3p', '3d'):
            res[o] = dict(r=r_, norm=nrm)
            continue
        res['f'].append(dict(orb=o, r=r_, norm=nrm, h=float(c @ bs.h[L] @ c),
                             R1_3p3d_3dnf=rk('3p', '3d', '3d', o, 1), R3_3p3d_3dnf=rk('3p', '3d', '3d', o, 3),
                             R2_3p3d_nf3d=rk('3p', '3d', o, '3d', 2)))
    res['G1_3p3d'] = rk('3p', '3d', '3d', '3p', 1)
    res['F2_3d3d'] = rk('3d', '3d', '3d', '3d', 2)
    res['f'].sort(key=lambda d: d['r'])
    print(f"<r> 3p {res['3p']['r']:.4f}  3d {res['3d']['r']:.4f}  G1(3p,3d) {res['G1_3p3d']:.5f}  "
          f"F2(3d,3d) {res['F2_3d3d']:.5f} Eh")
    print(' orb    <r>     norm      h(f)     R1(3p3d;3d nf)  R3(3p3d;3d nf)  R2(3p3d;nf 3d)')
    for d in res['f']:
        print(f" {d['orb']:4s} {d['r']:7.4f} {d['norm']:8.5f} {d['h']:10.4f}  {d['R1_3p3d_3dnf']:+.5f}        "
              f"{d['R3_3p3d_3dnf']:+.5f}        {d['R2_3p3d_nf3d']:+.5f}")
    json.dump(res, open(os.path.join(OUT, 'phase15l_step0.json'), 'w'), indent=1)


_SP = {}
_rname_k15 = K.rname_k
_spaces15k = K.spaces3


def fsingles(fs):
    """3p -> f singles on the core path, the cFf7 classes: 3p5 f 3d3 and 3s 3p5 f 3d4."""
    import phase15e as E15
    return ([E15.full_cfg('all', {'3p': 5, '3d': 3, v: 1}) for v in fs] +
            [E15.full_cfg('all', {'3s': 1, '3p': 5, '3d': 4, v: 1}) for v in fs])


def core_core(cls):
    """Core-core class X on P (3s2 3p6 3d3) and on the core-route Q configuration 3s 3p6 3d4 ("X on P+Q").
    Virtuals by <r> (15k compact3)."""
    import itertools as it
    import phase15e as E15
    P, Q = [], []
    if cls == 'a':          # 3p2 -> 3d2
        P, Q = [{'3p': 4, '3d': 5}], [{'3s': 1, '3p': 4, '3d': 6}]
    elif cls == 'b':        # 3s2 -> 3d2 (P only: Q has a single 3s)
        P = [{'3s': 0, '3d': 5}]
    elif cls == 'c':        # 3p2 -> 3d n'd
        P = [{'3p': 4, '3d': 4, d: 1} for d in K.compact3('d', 2)]
        Q = [{'3s': 1, '3p': 4, '3d': 5, d: 1} for d in K.compact3('d', 2)]
    elif cls == 'd':        # 3p2 -> f f' (incl. f2)
        for a, b in it.combinations_with_replacement(K.compact3('f', 2), 2):
            P.append({'3p': 4, '3d': 3, a: 1, b: 1} if a != b else {'3p': 4, '3d': 3, a: 2})
            Q.append({'3s': 1, '3p': 4, '3d': 4, a: 1, b: 1} if a != b else {'3s': 1, '3p': 4, '3d': 4, a: 2})
    elif cls == 'e':        # 3s 3p -> 3d n'p
        P = [{'3s': 1, '3p': 5, '3d': 4, p: 1} for p in K.compact3('p', 2)]
        Q = [{'3s': 0, '3p': 5, '3d': 5, p: 1} for p in K.compact3('p', 2)]
    return [E15.full_cfg('all', d) for d in P + Q]


def core_core_f(cls, f):
    """Class X applied to the 3p5 f 3d3 state (valence dicts)."""
    import itertools as it
    if cls == 'a':
        return [{'3p': 3, '3d': 5, f: 1}]
    if cls == 'b':
        return [{'3s': 0, '3p': 5, '3d': 5, f: 1}]
    if cls == 'c':
        return [{'3p': 3, '3d': 4, f: 1, d: 1} for d in K.compact3('d', 2)]
    if cls == 'd':
        out = []
        for a, b in it.combinations_with_replacement(K.compact3('f', 2), 2):
            c = {'3p': 3, '3d': 3, f: 1}
            c[a] = c.get(a, 0) + 1
            c[b] = c.get(b, 0) + 1
            out.append(c)
        return out
    if cls == 'e':
        return [{'3s': 1, '3p': 4, '3d': 4, f: 1, p: 1} for p in K.compact3('p', 2)]


def spaces15l():
    """{name: (kind, [configs], P first)}: step 1 (f-space sensitivity) and step 2 (core-core classes)."""
    if _SP:
        return _SP
    import phase15e as E15
    import phase15j as J
    sp = dict(_spaces15k())
    c2s = list(E15.core_spaces(3)['c2s'])
    fs = sorted(J.VIRT['f'], key=K.r_virt3().get)
    for f in fs:
        sp[f'cF1_{f}'] = ('allcore', c2s + fsingles([f]))
    sp['cFno1'] = ('allcore', c2s + fsingles(fs[1:]))
    allx = []
    for x in 'abcde':
        X = core_core(x)
        allx += X
        sp[f'c2s_{x}'] = ('allcore', c2s + X)
        sp[f'cFf7_{x}'] = ('allcore', sp['cFf7'][1] + X)
    sp['c2s_f'] = ('allcore', c2s + allx)
    sp['cFf7_f'] = ('allcore', sp['cFf7'][1] + allx)
    # classes changing cFf7 by >= 0.3 mEh (a, b, f): the same class also on the 3p5 f 3d3 states (all 4 f)
    onF = {x: [E15.full_cfg('all', d) for f in fs for d in core_core_f(x, f)] for x in 'abcde'}
    sp['cFf7_aF'] = ('allcore', sp['cFf7_a'][1] + onF['a'])
    sp['cFf7_bF'] = ('allcore', sp['cFf7_b'][1] + onF['b'])
    sp['cFf7_fF'] = ('allcore', sp['cFf7_f'][1] + [c for x in 'abcde' for c in onF[x]])   # 83k CSFs: not run
    sp['cFf7_fFnod'] = ('allcore', sp['cFf7_f'][1] + [c for x in 'abce' for c in onF[x]])
    # diagnostics for (a): 3p2 -> 3d2 on P only / on 3s 3p6 3d4 only
    XA = core_core('a')
    sp['c2s_aP'] = ('allcore', c2s + XA[:1])
    sp['c2s_aQ'] = ('allcore', c2s + XA[1:])
    sp['c0_aP'] = ('allcore', c2s[:1] + XA[:1])      # P + 3p4 3d5 only, no core route
    _SP.update(sp)
    return _SP


def rname(space, t):
    return f'v15l_{space}_d3_{t}e_{K.TAG3}' if space.startswith(('cF1_', 'cFno1', 'c2s_', 'cFf7_', 'c0_')) \
        else _rname_k15(space, t)


K.spaces3 = spaces15l
K.rname_k = rname


def dims(names):
    import counts as CN
    import phase13d as D
    sp = spaces15l()
    res = {}
    for name in names:
        kind, cf = sp[name]
        per = {t: CN.count(__import__('phase15e').LMAP, cf, t)[0]
               for L, twoS, m, t in D.term_list(3)}
        mx = max(per.values())
        res[name] = dict(ncfg=len(cf), per=per, max=mx, sum=sum(per.values()),
                         est_s=sum(K.tsec(d)[0] for d in per.values()), maxjob_s=K.tsec(mx)[0])
        r = res[name]
        print(f'{name:10s} cfg {len(cf):4d} largest {mx:6d} sum {r["sum"]:7d} est {r["est_s"] / 60:7.1f} CPU-min '
              f'largest job {r["maxjob_s"] / 60:6.1f} min (x3 {3 * r["maxjob_s"] / 60:6.1f})', flush=True)
    old = json.load(open(os.path.join(OUT, 'phase15l_dims.json'))) if os.path.exists(
        os.path.join(OUT, 'phase15l_dims.json')) else {}
    old.update(res)
    json.dump(old, open(os.path.join(OUT, 'phase15l_dims.json'), 'w'), indent=1)


def ras_cfgs(k):
    """RAS-like space (step 3): RAS1 = 3s, 3p (<= 2 holes), RAS2 = 3d, RAS3 = k most compact virtuals per l
    (s, p, d, f, g; None = all), <= 2 electrons in RAS3, at most double excitation from 3s2 3p6 3d3 or
    3s 3p6 3d4; even parity; 1s2 2s2 2p6 frozen."""
    import itertools as it
    import phase15e as E15
    import phase15j as J
    virt = [o for l in 'spdfg' for o in (J.VIRT[l] if k is None else K.compact3(l, k))]
    lof = {o: 'spdfgh'.index(o[-1]) for o in virt + ['3s', '3p', '3d']}
    cap = {o: 2 * (2 * lof[o] + 1) for o in lof}
    refs = [{'3s': 2, '3p': 6, '3d': 3}, {'3s': 1, '3p': 6, '3d': 4}]
    out = []
    # RAS3 part: 0, 1 or 2 electrons
    r3 = [{}] + [{o: 1} for o in virt] + [{a: 1, b: 1} if a != b else {a: 2}
                                          for a, b in it.combinations_with_replacement(virt, 2)]
    for ns in range(0, 3):
        for np_ in range(4, 7):
            if (2 - ns) + (6 - np_) > 2:
                continue
            for v in r3:
                nv = sum(v.values())
                nd = 11 - ns - np_ - nv
                if nd < 0 or nd > 10:
                    continue
                c = {'3s': ns, '3p': np_, '3d': nd, **v}
                if sum(lof[o] * q for o, q in c.items()) % 2:
                    continue
                exc = min(sum(max(0, r.get(o, 0) - c.get(o, 0)) for o in set(r) | set(c)) for r in refs)
                if exc <= 2:
                    out.append(c)
    P0 = {'3s': 2, '3p': 6, '3d': 3}
    out.sort(key=lambda c: c != P0)
    return [E15.full_cfg('all', c) for c in out]


def mrsd_cfgs(k, fref):
    """Pure MRCI-SD (addendum): every even-parity configuration within a double excitation of any reference
    {3s2 3p6 3d3, 3s 3p6 3d4} + {3s2 3p5 f 3d3 for f in fref}; orbitals 3s, 3p, 3d + the k most compact virtuals
    per l (s, p, d, f, g; None = all) + the reference f; 1s2 2s2 2p6 frozen. P stays 3s2 3p6 3d3 (first)."""
    import itertools as it
    import phase15e as E15
    import phase15j as J
    virt = [o for l in 'spdfg' for o in (J.VIRT[l] if k is None else K.compact3(l, k))]
    virt += [f for f in fref if f not in virt]
    orbs = ['3s', '3p', '3d'] + virt
    lof = {o: 'spdfgh'.index(o[-1]) for o in orbs}
    cap = {o: 2 * (2 * lof[o] + 1) for o in orbs}
    refs = [{'3s': 2, '3p': 6, '3d': 3}, {'3s': 1, '3p': 6, '3d': 4}] + \
           [{'3s': 2, '3p': 5, '3d': 3, f: 1} for f in fref]
    seen = set()
    for r in refs:
        occ = [o for o in r if r[o]]
        for nx in (0, 1, 2):
            for rem in it.combinations_with_replacement(occ, nx):
                c = dict(r)
                ok = True
                for o in rem:
                    c[o] -= 1
                    ok &= c[o] >= 0
                if not ok:
                    continue
                for add in it.combinations_with_replacement(orbs, nx):
                    d = dict(c)
                    for o in add:
                        d[o] = d.get(o, 0) + 1
                    if any(d[o] > cap[o] for o in d) or sum(lof[o] * q for o, q in d.items()) % 2:
                        continue
                    seen.add(frozenset((o, q) for o, q in d.items() if q))
    P0 = frozenset({'3s': 2, '3p': 6, '3d': 3}.items())
    out = [P0] + sorted(seen - {P0}, key=lambda c: sorted(c))
    # keep explicit zeros for 3s, 3p, 3d: full_cfg fills a missing core shell with its closed-shell default
    return [E15.full_cfg('all', {'3s': 0, '3p': 0, '3d': 0, **dict(c)}) for c in out]


def step3b():
    """Addendum: MRCI-SD from {3s2 3p6 3d3, 3s 3p6 3d4} (+ 3s2 3p5 f 3d3, f = most compact / all f)."""
    import counts as CN
    import phase13d as D
    import phase15j as J
    res = json.load(open(os.path.join(OUT, 'phase15l_step3.json')))
    f1 = K.compact3('f', 1)
    core = ('1s', '2s', '2p', '3s', '3p', '3d')

    def capped(cf):
        # RAS caps: <= 2 holes in 3s 3p, <= 2 electrons in the virtuals
        return [c for c in cf if (2 - dict(c).get('3s', 0)) + (6 - dict(c).get('3p', 0)) <= 2
                and sum(q for o, q in c if o not in core) <= 2]

    for k in (1, 2):
        for lab, fref, cap in [(lb, fr, cp) for cp in (False, True)
                               for lb, fr in (('2ref', []), ('3ref_f1', f1), ('3ref_fall', list(J.VIRT['f'])))]:
            key = f'{"rascap" if cap else "mrsd"}_{lab}_k{k}'
            cf = mrsd_cfgs(k, fref)
            cf = capped(cf) if cap else cf
            per = {}
            for L, twoS, m, t in D.term_list(3):
                per[t] = CN.count(__import__('phase15e').LMAP, cf, t)[0]
            mx = max(per.values())
            res[key] = dict(ncfg=len(cf), per=per, max=mx, mem_block_GB=8 * mx * mx / 1e9,
                            mem_analysis_GB=6 * 8 * mx * mx / 1e9, t_largest_s=K.tsec(mx)[0],
                            t_sum_s=sum(K.tsec(d)[0] for d in per.values()))
            r = res[key]
            print(f'{key}: cfg {len(cf)} per {per} max {mx}: H {r["mem_block_GB"]:.1f} GB, analysis '
                  f'~{r["mem_analysis_GB"]:.0f} GB, largest job {r["t_largest_s"] / 3600:.2f} h, all '
                  f'{r["t_sum_s"] / 3600:.1f} CPU-h', flush=True)
            json.dump(res, open(os.path.join(OUT, 'phase15l_step3.json'), 'w'), indent=1)


def step3():
    import counts as CN
    import phase13d as D
    import phase15e as E15
    res = {}
    for k in (1, 2, None):
        cf = ras_cfgs(k)
        per = {}
        for L, twoS, m, t in D.term_list(3):
            per[t] = CN.count(E15.LMAP, cf, t)[0]
            print(k, t, per[t], flush=True)
        mx = max(per.values())
        res[str(k)] = dict(ncfg=len(cf), per=per, max=mx, mem_block_GB=8 * mx * mx / 1e9,
                           mem_analysis_GB=6 * 8 * mx * mx / 1e9, t_largest_s=K.tsec(mx)[0],
                           t_sum_s=sum(K.tsec(d)[0] for d in per.values()))
        r = res[str(k)]
        print(f'k={k}: cfg {len(cf)} per {per} max {mx}: H {r["mem_block_GB"]:.1f} GB, analysis ~{r["mem_analysis_GB"]:.0f} GB,'
              f' largest job {r["t_largest_s"] / 3600:.2f} h, all {r["t_sum_s"] / 3600:.1f} CPU-h', flush=True)
        json.dump(res, open(os.path.join(OUT, 'phase15l_step3.json'), 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] in ('step3', 'step3b'):
        {'step3': step3, 'step3b': step3b}[sys.argv[1]]()
        sys.exit()
    cmd = sys.argv[1]
    if cmd == 'step0':
        step0()
    elif cmd == 'dims':
        dims(sys.argv[2:] or [n for n in spaces15l() if rname(n, '4F').startswith('v15l_')])
    elif cmd == 'prep':
        K.prep3(sys.argv[2:])
    elif cmd == 'run_names':
        K.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        K.analyze3.__globals__['OUT'] = OUT
        K.analyze3(sys.argv[2], sys.argv[3:])
