#!/usr/bin/env python3
"""Phase 21 A2 (run 14 Nov, ~11 h approved): Cr3+ fam (fam_d5f2g2h1, crnd3 orbitals) with and without the job-A
three-hole class, plus the balanced second-order variants in the same space.

Space famA = fam + the class: every even-parity 3s1 3p4 3d^a configuration with <= 2 electrons in the fam virtuals and
excitation level <= 2 from R2 = 3s 3p6 3d^(N+1) (3s1 3p4 3d6; 3s1 3p4 3d5 v; 3s1 3p4 3d4 v v'), 46 configurations, none
in fam. Run names v21_famA_d3_<t>e_crnd3 (ATOMCI, crnd3 vectors). 'without' = the principal sub-matrix on the fam
CSFs (checked against the stored v15u_fam_d5f2g2h1_d3_<t>e_crnd3 run: same CSF labels / couplings, same diagonal).
Placement as phase21c.scan (hf = P19.hf_blocks, vcm U).

Variants per block (H_eff, m x m, by = hf + H_eff - U^T Hpp U):
  exact:famA, exact:fam               des Cloizeaux, full space
  bal:famQ1                           balanced: Q1 = every fam Q CSF, Q2 = the class (D = p_j - H_qq)
  R:fam, R:famA, R+class:famA         balanced with Q1 = Q1_rule (phase17c_scan.rset 'R'); R:famA has the class in
                                      Q2 except members of Q1_rule; R+class:famA has the whole class in Q1
  S2-diag / C2 (additive class increment) = 0 by structure if H(P, class) = 0: checked (|H(P, class)|_max stored).
Commands: dims / prep / block <t> (-> a2/a2_<t>.npz + a2/a2_<t>.json) / final (-> phase21f.json)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
HERE = os.path.join(ROOT, 'results', 'phase21')  # outputs (was this script's directory)
os.makedirs(HERE, exist_ok=True)
_C.setup_path()
A2D = os.environ.get('A2_OUT') or os.path.join(HERE, 'a2')   # A2_OUT / A2_SRC: code test on another npz
JS = os.path.join(HERE, 'phase21f.json')
SP = 'famA'
BASE = 'fam_d5f2g2h1'
EH = 219474.63


def run_name(t):
    return f'v21_{SP}_d3_{t}e_crnd3'


def setup_decks():
    import itertools
    import phase21c as C
    C.MY = C.MY + (SP,)
    M, N15, spaces, rname = C.setup('crnd')
    import phase15s as S
    kind, cf = spaces()[BASE]
    V = sorted({o for c in cf for o, q in c if int(o[:-1]) >= 11}, key=lambda o: (o[-1], o))
    core = {('1s', 2), ('2s', 2), ('2p', 6), ('3s', 1), ('3p', 4)}
    cls = [frozenset(core | {('3d', 6)})] + [frozenset(core | {('3d', 5), (v, 1)}) for v in V]
    for v, w in itertools.combinations_with_replacement(V, 2):
        cls.append(frozenset(core | {('3d', 4), (v, 2)} if v == w else core | {('3d', 4), (v, 1), (w, 1)}))
    par = lambda c: sum(q * 'spdfghik'.index(o[-1]) for o, q in c) % 2  # noqa: E731
    cls = [c for c in cls if par(c) == 0]
    fs = set(cf)
    new = [c for c in cls if c not in fs]
    assert len(new) == len(cls) == 46, (len(cls), len(new))
    base_spaces = N15.spaces

    def spaces2():
        sp = base_spaces()
        sp[SP] = (kind, list(cf) + new)
        return sp
    N15.spaces = spaces2
    S.Q.spaces = spaces2
    assert rname(SP, '2P') == run_name('2P')
    return M, N15, spaces2, rname, V, new


def dims():
    M, N15, spaces, rname, V, new = setup_decks()
    import counts as CN
    import phase15e as E15
    import phase13d as D
    sp = spaces()
    out = {}
    for L, twoS, m, t in D.term_list(3):
        out[t] = dict(fam=CN.count(E15.LMAP, sp[BASE][1], t)[0], famA=CN.count(E15.LMAP, sp[SP][1], t)[0])
    print('virtuals', V, 'class', len(new), out, flush=True)
    return out


def prep():
    M, N15, spaces, rname, V, new = setup_decks()
    import phase15k as K
    import phase13d as D
    kind, cf = spaces()[SP]
    for L, twoS, m, t in D.term_list(3):
        K.write_deck3(rname(SP, t), kind, cf, t)
    print('decks written', [rname(SP, t) for L, twoS, m, t in D.term_list(3)], flush=True)


def _env():
    _C.setup_path()
    os.environ['ION'] = 'crnd'
    os.environ['HEFF_SOLVER'] = 'iter'


def block(t):
    _env()
    import resource
    import phase19b as P19
    import phase21a as A21
    import phase17c_scan as CS
    from atomci_heff import iterative as IT
    from atomci_heff.sparse import SymLower
    os.makedirs(A2D, exist_ok=True)
    (L, twoS, m) = [(L, twoS, m) for L, twoS, m, tt in P19.terms(3) if tt == t][0]
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hf = P19.hf_blocks(dict(N=3))[t]
    U = np.array(maps[f'vcm_d3_{t}e'])
    t0 = time.time()
    z = np.load(os.environ.get('A2_SRC') or os.path.join(ROOT, 'results', run_name(t) + '.npz'), allow_pickle=True)
    lab = [str(x) for x in z['labels']]
    cpl = [str(x) for x in z['couplings']]
    op = SymLower.from_npz(z)
    del z
    assert all(la == lab[0] for la in lab[:m]) and all(la != lab[0] for la in lab[m:]), t
    isc = {la: (lambda d: d['3s'] == 1 and d['3p'] == 4)(A21.occ(la)) for la in set(lab)}
    cl = np.array([i for i, la in enumerate(lab) if isc[la]], dtype=int)
    kpos = np.array([i for i, la in enumerate(lab) if not isc[la]], dtype=int)
    assert list(kpos[:m]) == list(range(m))
    info = dict(n_famA=int(op.n), n_fam=len(kpos), n_class=len(cl), class_cfgs=len({lab[i] for i in cl}))
    # check: fam CSFs == the stored fam run (labels, couplings, diagonal)
    zf = np.load(os.environ.get('A2_SRC') or os.path.join(ROOT, 'results', f'v15u_{BASE}_d3_{t}e_crnd3.npz'), allow_pickle=True)
    lf, cf_ = [str(x) for x in zf['labels']], [str(x) for x in zf['couplings']]
    rows, cols, vals = zf['H_rows'], zf['H_cols'], zf['H_vals']
    dfam = np.zeros(len(lf))
    dg = rows == cols
    dfam[rows[dg]] = vals[dg]
    del zf, rows, cols, vals, dg
    same = (lf == [lab[i] for i in kpos]) and (cf_ == [cpl[i] for i in kpos])
    info['fam_check'] = dict(same_csfs=bool(same), max_diag_dev=float(np.abs(dfam - op.d[kpos]).max()) if same else None)
    print(t, info, f'{time.time() - t0:.0f} s', flush=True)
    assert os.environ.get('A2_SRC') or (same and info['fam_check']['max_diag_dev'] < 1e-8), info['fam_check']
    Hpp = op.sub(np.arange(m)).dense()
    php = U.T @ Hpp @ U
    by, inf = {}, {}
    info['HPclass_max'] = float(np.abs(op.block(np.arange(m), cl).toarray()).max()) if len(cl) else 0.0
    ex, i_ = IT.heff_block(dict(P=list(range(m)), Q=list(range(m, op.n)), m=m, U=U), op)
    by['exact:famA'] = hf + ex - php
    inf['exact:famA'] = dict(wP=float(i_['wP']), s=time.time() - t0)
    print(t, 'exact:famA', inf['exact:famA'], flush=True)
    # balanced, Q1 = fam Q, Q2 = class
    h, ib = CS.heff(op, m, kpos[m:], Hpp)
    by['bal:famQ1'] = hf + U.T @ h @ U - php
    inf['bal:famQ1'] = dict(wP=ib['wP'], it=ib['it'], s=time.time() - t0)
    print(t, 'bal:famQ1', f'{time.time() - t0:.0f} s', flush=True)
    labA = lab
    cfgs = sorted(set(labA[m:]))
    R = CS.rset('R', cfgs, 3)
    clc = {lab[i] for i in cl}
    q1R = np.array([i for i in range(m, op.n) if labA[i] in R], dtype=int)
    q1Rc = np.array([i for i in range(m, op.n) if labA[i] in R or labA[i] in clc], dtype=int)
    for v, q1 in (('R:famA', q1R), ('R+class:famA', q1Rc)):
        h, ir = CS.heff(op, m, q1, Hpp)
        by[v] = hf + U.T @ h @ U - php
        inf[v] = dict(n1=ir['n1'], wP=ir['wP'], it=ir['it'], s=time.time() - t0)
        print(t, v, inf[v]['n1'], f'{time.time() - t0:.0f} s', flush=True)
    info['class_in_R'] = sorted(clc & R)
    sub = op.sub(kpos)
    del op
    labk = [lab[i] for i in kpos]
    exk, ik = IT.heff_block(dict(P=list(range(m)), Q=list(range(m, sub.n)), m=m, U=U), sub)
    by['exact:fam'] = hf + exk - php
    inf['exact:fam'] = dict(wP=float(ik['wP']), s=time.time() - t0)
    q1Rk = np.array([i for i in range(m, sub.n) if labk[i] in R], dtype=int)
    h, ir = CS.heff(sub, m, q1Rk, Hpp)
    by['R:fam'] = hf + U.T @ h @ U - php
    inf['R:fam'] = dict(n1=ir['n1'], wP=ir['wP'], it=ir['it'], s=time.time() - t0)
    del sub
    e0 = np.linalg.eigvalsh(by['exact:famA'])
    info['eig_vs_exact_famA_cm'] = {v: ((np.linalg.eigvalsh(b) - e0) * EH).tolist() for v, b in by.items()}
    info['variants'] = inf
    info['peak_rss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    info['seconds'] = time.time() - t0
    np.savez(os.path.join(A2D, f'a2_{t}.npz'), **{v.replace(':', '__'): b for v, b in by.items()})
    json.dump(dict(t=t, L=L, twoS=twoS, m=m, **info), open(os.path.join(A2D, f'a2_{t}.json'), 'w'), indent=1, default=float)
    print(t, 'eig - exact:famA (cm-1)', {v: np.round(x, 1).tolist() for v, x in info['eig_vs_exact_famA_cm'].items()},
          f"peak {info['peak_rss_gb']:.1f} GB, {info['seconds']:.0f} s", flush=True)


def final():
    _env()
    import phase19b as P19
    V = ['exact:famA', 'exact:fam', 'bal:famQ1', 'R:fam', 'R:famA', 'R+class:famA']
    by = {v: {} for v in V}
    blk = {}
    for L, twoS, m, t in P19.terms(3):
        z = np.load(os.path.join(A2D, f'a2_{t}.npz'))
        for v in V:
            by[v][(L, twoS)] = z[v.replace(':', '__')]
        blk[t] = json.load(open(os.path.join(A2D, f'a2_{t}.json')))
    out = {}
    for v in V:
        p = P19.decompose(dict(N=3), by[v])
        out[v] = {k: float(p[k]) for k in ('T1', 'T2', 'T3', 'T4', 'B', 'C', 'alpha', 'beta') if k in p}
    ex = out['exact:famA']
    dT1 = ex['T1'] - out['exact:fam']['T1']
    for v in V:
        out[v]['err_T1'] = out[v]['T1'] - ex['T1']
        out[v]['err_T2'] = out[v]['T2'] - ex['T2']
    rec = {'bal:famQ1 vs exact:fam': out['bal:famQ1']['T1'] - out['exact:fam']['T1'],
           'R:famA vs R:fam': out['R:famA']['T1'] - out['R:fam']['T1'],
           'R+class:famA vs R:fam': out['R+class:famA']['T1'] - out['R:fam']['T1']}
    res = dict(variants=out, dT1_exact=dT1, dT2_exact=ex['T2'] - out['exact:fam']['T2'],
               recovery={k: dict(dT1=x, frac=x / dT1) for k, x in rec.items()},
               P4=dict(threshold=0.86, k1fh_dT1=0.8588, holds=bool(dT1 > 0.86)), blocks=blk)
    json.dump(res, open(JS, 'w'), indent=1, default=float)
    for v in V:
        print(f"{v:14s} T1 {out[v]['T1']:.4f} T2 {out[v]['T2']:.4f} err {out[v]['err_T1']:+.4f} / {out[v]['err_T2']:+.4f}")
    print(f'dT1 exact {dT1:+.4f} (k1fh +0.8588) -> P4', 'HOLDS' if dT1 > 0.86 else 'FAILS')
    for k, x in res['recovery'].items():
        print(f'recovery {k:24s} {x["dT1"]:+.4f} = {100 * x["frac"]:.0f} %')


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'dims':
        dims()
    elif c == 'prep':
        prep()
    elif c == 'block':
        block(sys.argv[2])
    elif c == 'final':
        final()
