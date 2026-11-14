#!/usr/bin/env python3
"""Phase 15n: the balanced RAS k = 1 space across ions, with and without the compact f.

Ion via env ION = v | cr | mn | fe (own 15e / 15h basis and orbitals, read from runs/vq_orb_<tag>/be.NO).
  rvirt                 <r> of the virtuals and the k = 1 choice per ion            -> phase15n_rvirt.json
  dims [spaces]         CSF counts per (L,S) block, time estimates                  -> phase15n_dims.json
  prep <spaces>; run_names <names>; analyze <tag> <spaces>
  radial                step 3 radial integrals per ion                             -> phase15n_radial.json
  k2plan                step 4 RAS k = 2 sizes (full and reduced variants)          -> phase15n_k2plan.json
Spaces: rasK1 (virtuals 1 per l by <r>), rasK1_nof (no f), rasK1_fdif (the most diffuse f instead of the compact one).
Same RAS rules as 15m step 1: 3s 3p <= 2 holes, <= 2 electrons in the virtuals, <= double from
{3s2 3p6 3d^N, 3s 3p6 3d^(N+1)}, plus 2s 3d^(N+1) so that the base is c2s; 1s2 2s(2) 2p6 otherwise frozen.
"""
import itertools as it
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase15m as M  # noqa: E402  (sets the ion: K.Z3, K.BAS3, K.TAG3)
import phase15k as K  # noqa: E402

OUT = K.OUT
ION, N_, TAG_ = M.ION, M.N_, M.TAG_
_RV = {}


def r_virt():
    """<r> (bohr) of every virtual of this ion's orbital set (be.NO of vq_orb_<tag>; column = gen_v.vec - 1)."""
    if not _RV:
        import gen_v
        import phase15j as J
        from atomci_heff.reader import read_vectors
        bs = K.H.F.Bas(M.BAS_, M.Z_)
        C = read_vectors(os.path.join(ROOT, 'runs', f'vq_orb_{TAG_}', 'be.NO'), bs.nbf)[0]
        for l, lst in J.VIRT.items():
            L = J.LV.index(l)
            for o in lst + ([f'3{l}'] if l in 'spd' else []):
                if o == '3s' or o == '3p' or o == '3d' or o in lst:
                    p = bs.P(L, C[L][:, gen_v.vec(o) - 1])
                    _RV[o] = float(np.sum(p * p * bs.r * bs.w))
    return _RV


K.r_virt3 = r_virt          # compact3 now ranks this ion's virtuals


def virt_k(kmap):
    """Virtual list: kmap {l: k (most compact k) | list of names}."""
    out = []
    for l in 'spdfg':
        v = kmap.get(l, 0)
        out += K.compact3(l, v) if isinstance(v, int) else list(v)
    return out


def ras_cfgs(virt):
    """RAS configurations for this ion's d^N (15l ras_cfgs generalised to N and an explicit virtual list), P first,
    plus 2s 3d^(N+1)."""
    import phase15e as E15
    lof = {o: 'spdfghik'.index(o[-1]) for o in virt + ['3s', '3p', '3d']}
    refs = [{'3s': 2, '3p': 6, '3d': N_}, {'3s': 1, '3p': 6, '3d': N_ + 1}]
    nel = 8 + N_
    r3 = [{}] + [{o: 1} for o in virt] + [{a: 1, b: 1} if a != b else {a: 2}
                                          for a, b in it.combinations_with_replacement(virt, 2)]
    out = []
    for ns in range(0, 3):
        for np_ in range(4, 7):
            if (2 - ns) + (6 - np_) > 2:
                continue
            for v in r3:
                nd = nel - ns - np_ - sum(v.values())
                if nd < 0 or nd > 10:
                    continue
                c = {'3s': ns, '3p': np_, '3d': nd, **v}
                if sum(lof[o] * q for o, q in c.items()) % 2:
                    continue
                if min(sum(max(0, r.get(o, 0) - c.get(o, 0)) for o in set(r) | set(c)) for r in refs) <= 2:
                    out.append(c)
    P0 = {'3s': 2, '3p': 6, '3d': N_}
    out.sort(key=lambda c: c != P0)
    cf = [E15.full_cfg('all', c) for c in out]
    c2s = list(E15.core_spaces(N_)['c2s'])
    assert set(c2s[:2]) <= set(cf) and c2s[0] == cf[0]
    return cf + [c for c in c2s if c not in cf]


_SP = {}


def spaces():
    if not _SP:
        import phase15j as J
        f_dif = sorted(J.VIRT['f'], key=r_virt().get)[-1]
        _SP['rasK1'] = ('allcore', ras_cfgs(virt_k({l: 1 for l in 'spdfg'})))
        _SP['rasK1_nof'] = ('allcore', ras_cfgs(virt_k({'s': 1, 'p': 1, 'd': 1, 'g': 1})))
        _SP['rasK1_fdif'] = ('allcore', ras_cfgs(virt_k({'s': 1, 'p': 1, 'd': 1, 'f': [f_dif], 'g': 1})))
    return _SP


def rname(space, t):
    if space == 'rasK1' and ION == 'cr':
        return f'v15m_rasK1_d3_{t}e_cr15n3'      # identical configuration set (checked), run in 15m
    if space in spaces():
        return f'v15n_{space}_d{N_}_{t}e_{TAG_}'
    return M.rname(space, t)


K.rname_k = rname


def dims(names, extra=None):
    import counts as CN
    import phase15e as E15
    sp = dict(spaces(), **(extra or {}))
    f = 1 / 0.6 if N_ == 6 else 1.0
    path = os.path.join(OUT, 'phase15n_dims.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in names or list(sp):
        kind, cf = sp[name]
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in M.term_list()}
        mx = max(per.values())
        r = dict(ncfg=len(cf), per=per, max=mx, sum=sum(per.values()), est_s=f * sum(K.tsec(d)[0] for d in per.values()),
                 maxjob_s=f * K.tsec(mx)[0], mem_block_GB=8 * mx * mx / 1e9)
        res[f'{ION}:{name}'] = r
        print(f'{ION}:{name:12s} cfg {len(cf):4d} largest {mx:6d} sum {r["sum"]:7d} est {r["est_s"] / 3600:6.2f} CPU-h '
              f'largest job {r["maxjob_s"] / 3600:5.2f} h (x3 {3 * r["maxjob_s"] / 3600:5.2f}) block {r["mem_block_GB"]:.1f} GB',
              flush=True)
    json.dump(res, open(path, 'w'), indent=1)
    return res


def prep(names):
    import phase15j as J
    J.BAS6 = M.BAS_
    sp = spaces()
    for name in names:
        kind, cf = sp[name]
        for L, twoS, m, t in M.term_list():
            (K.write_deck3 if N_ == 3 else J.write_deck)(rname(name, t), kind, cf, t)
    print('decks written:', ION, ' '.join(names))


def analyze(tag, names):
    if N_ == 3:
        K.analyze3(tag, names)
        return
    import phase15h as H
    import phase15j as J
    J.analyze2(names, lambda sp, t: H.rname(sp, 6, t, TAG_) if sp == 'c2s' else rname(sp, t), tag)


def radial():
    """Radial integrals that couple P, 3s 3d^(N+1) and 3p4 3d^(N+2) (frozen own orbitals), per ion."""
    from atomci_heff import integrals as I
    res = {}
    for ion, (N, Z, bas, tag) in M.IONS.items():
        bs, P, h = M.orbitals_of(tag, bas, Z)

        def R(a, b, c, d, k):     # R^k(ab; cd) = int int Pa(1) Pb(2) r<^k / r>^(k+1) Pc(1) Pd(2)
            return float(I.slater(P[a], P[c], P[b], P[d], k, bs.x, bs.r))
        res[ion] = dict(R1_3s3d_3p3p=R('3s', '3d', '3p', '3p', 1),      # 3s 3d^(N+1) <-> 3p4 3d^(N+2)
                        R2_3d3d_3d3s=R('3d', '3d', '3d', '3s', 2),      # P <-> 3s 3d^(N+1) (H2(3s,3d) of Trees)
                        R1_3p3p_3d3d=R('3p', '3p', '3d', '3d', 1),      # P <-> 3p4 3d^(N+2)
                        R3_3p3p_3d3d=R('3p', '3p', '3d', '3d', 3),
                        G1_3p3d=R('3p', '3d', '3d', '3p', 1), G3_3p3d=R('3p', '3d', '3d', '3p', 3),
                        G2_3s3d=R('3s', '3d', '3d', '3s', 2), F2_3d3d=R('3d', '3d', '3d', '3d', 2))
        print(ion, {k: round(v, 5) for k, v in res[ion].items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase15n_radial.json'), 'w'), indent=1)


def k2plan():
    variants = {'rasK2': {l: 2 for l in 'spdfg'},
                'rasK2_sdg': {'s': 2, 'p': 1, 'd': 2, 'f': 1, 'g': 2},
                'rasK2_sd': {'s': 2, 'p': 1, 'd': 2, 'f': 1, 'g': 1},
                'rasK2_dg': {'s': 1, 'p': 1, 'd': 2, 'f': 1, 'g': 2},
                'rasK2_d': {'s': 1, 'p': 1, 'd': 2, 'f': 1, 'g': 1}}
    extra = {k: ('allcore', ras_cfgs(virt_k(v))) for k, v in variants.items()}
    res = dims(list(extra), extra)
    out = {k: dict(res[f'{ION}:{k}'], kmap=variants[k], virt=virt_k(variants[k])) for k in variants}
    json.dump(out, open(os.path.join(OUT, f'phase15n_k2plan_{ION}.json'), 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'rvirt':
        rv = r_virt()
        print(ION, {l: K.compact3(l, 1) for l in 'spdfg'}, {o: round(r, 3) for o, r in sorted(rv.items(), key=lambda x: x[1])})
        path = os.path.join(OUT, 'phase15n_rvirt.json')
        d = json.load(open(path)) if os.path.exists(path) else {}
        d[ION] = dict(r=rv, k1={l: K.compact3(l, 1) for l in 'spdfg'})
        json.dump(d, open(path, 'w'), indent=1)
    elif cmd == 'dims':
        dims(sys.argv[2:])
    elif cmd == 'prep':
        prep(sys.argv[2:])
    elif cmd == 'run_names':
        M.run_names(sys.argv[2:])
    elif cmd == 'analyze':
        analyze(sys.argv[2], sys.argv[3:])
    elif cmd == 'radial':
        radial()
    elif cmd == 'k2plan':
        k2plan()
