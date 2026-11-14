#!/usr/bin/env python3
"""Phase 17 A1 for Fe2+ 3d6: the second-order variants of phase17a (C2, S2-cav, S2-diag; per Q configuration) on the
largest stored Fe2+ space, 15n RAS k = 1 (v15n_rasK1_d6_<t>e_fe15h6; standard Sapporo-QZP-2012, own orbitals
fe15h6). Same P / HF placement and T1, T2 projection as phase15j.analyze2 (exact, S2-full = 'sylvester' and the
15j 'closure' are read from phase15j_o_fe.json). -> results/phase17/a_fe_rasK1.npz / .json
usage: phase17a_fe.py [space]
Own-orbital runs (30 Oct): env A_RUNFMT (run name with {t}) and A_ION; the HF P block is then the run's own PHP
(phase17.own_hf), output a_<A_ION>_<space>.npz / .json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase13b as B  # noqa: E402
import phase13d as D  # noqa: E402
import phase15h as H  # noqa: E402
import phase15j as J  # noqa: E402
import phase17a as A  # noqa: E402  (classify; imports the d3 setup, unused here)
from atomci_heff.sparse import SymLower  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase17')


def t12(by):
    O1, O2 = H.o_axes()
    c, _ = H.decompose(6, B.vec_from_blocks(6, by))
    T1, T2, out = H.t1t2(c, O1, O2)
    return 1e3 * T1, 1e3 * T2


def main(space='rasK1'):
    tl = D.term_list(6)
    fmt = os.environ.get('A_RUNFMT')
    hf = {} if fmt else {t: v[1] for t, v in J.path_blocks('dSD').items()}
    tagout = os.environ.get('A_ION', 'fe')
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    bl, acc = [], {}
    for L, twoS, m, t in tl:
        name = fmt.format(t=t) if fmt else f'v15n_{space}_d6_{t}e_fe15h6'
        d = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
        lab = [str(x) for x in d['labels']]
        dg = SymLower.from_npz(d).d.copy()
        g = (2 * L + 1) * (twoS + 1)
        for i, la in enumerate(lab):
            s_, w = acc.get(la, (0.0, 0.0))
            acc[la] = (s_ + g * dg[i], w + g)
        bl.append(dict(t=t, L=L, twoS=twoS, m=m, name=name, lab=lab, U=np.array(maps[f'vcm_d6_{t}e'])))
    eav = {k: s_ / w for k, (s_, w) in acc.items()}
    p0 = bl[0]['lab'][0]
    cfgs = sorted(set(eav) - {p0})
    ci = {c: k for k, c in enumerate(cfgs)}
    ea = np.array([eav[c] for c in cfgs])
    per = {}
    for b in bl:
        m, t, U = b['m'], b['t'], b['U']
        op = SymLower.from_npz(np.load(os.path.join(ROOT, 'results', b['name'] + '.npz'), allow_pickle=True))
        lab = b['lab']
        assert all(la == p0 for la in lab[:m]) and all(la != p0 for la in lab[m:])
        P, Qi = np.arange(m), np.arange(m, op.n)
        Hpp = op.sub(P).dense()
        if fmt:
            hf[t] = U.T @ Hpp @ U
        V = op.block(P, Qi).toarray()
        dq, cq = op.d[m:], np.array([ci[la] for la in lab[m:]])
        p, Am = np.linalg.eigh(Hpp)
        W = Am.T @ V
        off = np.diag(U.T @ Hpp @ U - hf[t])
        assert np.abs(U.T @ Hpp @ U - hf[t] - off.mean() * np.eye(m)).max() < 1e-7, t
        for meth, Dn in {'C2': np.broadcast_to(eav[p0] - ea[cq], (m, len(cq))), 'S2-cav': p[:, None] - ea[cq][None, :],
                         'S2-diag': p[:, None] - dq[None, :]}.items():
            Y = W / Dn
            G = np.zeros((len(cfgs), m, m))
            np.add.at(G, cq, np.transpose(V[:, :, None] * Y.T[None, :, :], (1, 0, 2)))
            G = G @ Am.T
            G = 0.5 * (G + np.transpose(G, (0, 2, 1)))
            per[f'{meth}|{t}'] = U.T @ G @ U
        per[f'hf|{t}'] = hf[t]
        print(space, t, op.n, flush=True)
        del op, V, W
    np.savez(os.path.join(OUT, f'a_{tagout}_{space}.npz'), cfgs=np.array(cfgs), **per)
    terms = [t for L, twoS, m, t in tl]
    key = {t: (L, twoS) for L, twoS, m, t in tl}
    res = {}
    cl = np.array([A.classify(str(c), N=6) for c in cfgs])
    T0 = t12({key[t]: per[f'hf|{t}'] for t in terms})
    for meth in A.METH:
        tot = t12({key[t]: per[f'hf|{t}'] + per[f'{meth}|{t}'].sum(axis=0) for t in terms})
        cls = {}
        for c in A.CLASSES:
            mk = cl == c
            if mk.any():
                x = t12({key[t]: per[f'hf|{t}'] + per[f'{meth}|{t}'][mk].sum(axis=0) for t in terms})
                cls[c] = [x[0] - T0[0], x[1] - T0[1], int(mk.sum())]
        res[meth] = dict(T1=tot[0], T2=tot[1], classes=cls)
        print(meth, round(tot[0], 4), round(tot[1], 4), flush=True)
    res['hf'] = T0
    json.dump(res, open(os.path.join(OUT, f'a_{tagout}_{space}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
