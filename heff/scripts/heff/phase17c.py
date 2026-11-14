#!/usr/bin/env python3
"""Phase 17 C: partially dressed H_eff on the stored H (no ATOMCI). Which part Q1 of Q must be treated to all orders,
with the rest Q2 at second order?

Per term block (P = the m CSFs of 3s2 3p6 3d^N, first; Q1 = the CSFs of the configurations named by the variant;
Q2 = the rest of Q), for each P root j (p_j, a_j = eigenpairs of PHP; reference energy E_ref = p_j, state specific,
as S2-diag in phase17a):
  balanced  H'_j = H(P+Q1) + H(P+Q1, Q2) D_j^-1 H(Q2, P+Q1),   D_j = diag(p_j - H_qq), q in Q2
  pp        H'_j = H(P+Q1) with only the PP block dressed: H'_PP += H(P, Q2) D_j^-1 H(Q2, P)
H'_j is diagonalized densely; of its roots the one with the largest (a_j . x_P)^2 is taken, (E_j, x_j) for j = 1..m
give the des Cloizeaux H_eff over P (S^-1/2 X diag(E) X^T S^-1/2, X = P components), placed on the HF P block as in
15k analyze3 (HF + H_eff - PHP) and projected (decompose3: B, C, beta, T1..T4).
Q1 = {} gives S2-diag (exactly for m = 1 blocks); Q1 = all of Q gives exact (check: variant 'full', small spaces).

Q1 variants (configurations, N = 3):
  C-a   n = 3 complex: no virtual electron, no 1s/2s/2p hole (3s 3d4, 3p4 3d5, 3s0 3d5 in the Step 8 space)
  C-b   C-a + the 3p -> f singles 3s2 3p5 3d3 nf (every f)
  C-c   C-b + one C-a configuration with one more 3p -> f or 3d -> f (3s 3p5 3d4 f, 3s 3d3 f, 3p3 3d5 f, 3p4 3d4 f, ...)
  C-d   C-b + the 3d^(N-1) ns buffer (3s2 3p6 3d2 ns, every s; the V2+ 3d2 11s intruder)
Ion via env ION (vnd | crnd | mnnd), as phase17a; A_RUNFMT / A_TAG / A_ION for the older runs.
Round 2: forms own-diag / own-cav ('bal-own', see levels), C-e = C-d + 3s2 3p6 3d2 nd (every virtual d), C-e:<nd> =
C-d + that one nd; env C_TAG names the output (c_<ion>_<space>_<C_TAG>.json).
usage: phase17c.py <space> [variants ...]  -> results/phase17/c_<ion>_<space>.json"""
import json
import os
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase17a as A  # noqa: E402
import atomci_heff.approx as AP  # noqa: E402
from atomci_heff.sparse import SymLower  # noqa: E402

N = int(os.environ.get('C_N', '3'))     # 6: Fe2+ d6 (A_RUNFMT runs, own PHP as HF block, T1 / T2 only)


def blocks6(fmt):
    """d6 counterpart of phase17a.blocks: blocks of the runs fmt.format(t=t), configuration averages, and the HF P
    block = the run's own PHP (canonical seniority basis; phase17.own_hf)."""
    import json
    import phase13d as D
    maps = json.load(open(os.path.join(A.ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    bl, acc, hf = [], {}, {}
    for L, twoS, m, t in D.term_list(6):
        name = fmt.format(t=t)
        op = SymLower.from_npz(np.load(os.path.join(A.ROOT, 'results', name + '.npz'), allow_pickle=True))
        lab = [str(x) for x in np.load(os.path.join(A.ROOT, 'results', name + '.npz'), allow_pickle=True)['labels']]
        g = (2 * L + 1) * (twoS + 1)
        for i, la in enumerate(lab):
            s_, w = acc.get(la, (0.0, 0.0))
            acc[la] = (s_ + g * op.d[i], w + g)
        U = np.array(maps[f'vcm_d6_{t}e'])
        hf[t] = U.T @ op.sub(np.arange(m)).dense() @ U
        bl.append(dict(t=t, L=L, twoS=twoS, m=m, name=name, lab=lab, U=U))
        del op
    return bl, {k: s_ / w for k, (s_, w) in acc.items()}, hf, None


def params6(by):
    """{t: m x m} -> T1, T2 (mEh) by the 15h d6 projection (as phase17a_fe.t12); B, C, beta not defined here."""
    import phase13b as B13
    import phase13d as D
    import phase15h as H
    key = {t: (L, twoS) for L, twoS, m, t in D.term_list(6)}
    O1, O2 = H.o_axes()
    c, _ = H.decompose(6, B13.vec_from_blocks(6, {key[t]: v for t, v in by.items()}))
    T1, T2, _ = H.t1t2(c, O1, O2)
    return dict(T1=1e3 * T1, T2=1e3 * T2, **{q: float('nan') for q in ('T3', 'T4', 'B', 'C', 'alpha', 'beta')})
Q5 = ('T1', 'T2', 'T3', 'T4', 'B', 'C', 'alpha', 'beta')


def occ(label):
    o = AP.parse_label(label)
    vir = {k: q for k, q in o.items() if int(k[:-1]) >= 11}
    inner = (o.get('1s', 0), o.get('2s', 0), o.get('2p', 0)) == (2, 2, 6)
    return (o.get('3s', 0), o.get('3p', 0), o.get('3d', 0)), vir, inner


def n3(label):
    c, vir, inner = occ(label)
    return not vir and inner and c != (2, 6, N)


def pf1(label):
    c, vir, inner = occ(label)
    return inner and c == (2, 5, N) and len(vir) == 1 and sum(vir.values()) == 1 and next(iter(vir))[-1] == 'f'


def cc(label, ca):
    c, vir, inner = occ(label)
    if not (inner and len(vir) == 1 and sum(vir.values()) == 1 and next(iter(vir))[-1] == 'f'):
        return False
    par = {(s, p - 1, d) for s, p, d in ca} | {(s, p, d - 1) for s, p, d in ca}
    return c in par and c != (2, 5, N)


def buf(label, l='s'):
    c, vir, inner = occ(label)
    return inner and c == (2, 6, N - 1) and len(vir) == 1 and sum(vir.values()) == 1 and next(iter(vir))[-1] == l


def q1_sets(cfgs):
    a = [c for c in cfgs if n3(c)]
    ca = {occ(c)[0] for c in a}
    b = a + [c for c in cfgs if pf1(c)]
    d = b + [c for c in cfgs if buf(c)]
    vd = [c for c in cfgs if buf(c, 'd')]
    out = {'C0': [], 'C-a': a, 'C-b': b, 'C-c': b + [c for c in cfgs if cc(c, ca)], 'C-d': d, 'C-e': d + vd,
           'full': list(cfgs)}
    for c in vd:                                   # C-d + one virtual d (split of C-e)
        out['C-e:' + next(iter(occ(c)[1]))] = d + [c]
    return out


KINDS = ('balanced', 'pp', 'own-diag', 'own-cav')


def levels(op, m, q1, Hpp, cav=None):
    """{kind: H_dC (m x m, CSF basis)} for P = first m rows, Q1 index array q1 (in Q), Q2 the rest.
    own-diag / own-cav (round 2, 'bal-own'): element (a, b) of P+Q1 dressed with
    1/2 [(E_a - H_qq)^-1 + (E_b - H_qq)^-1], E_a = p_j for a in P, H_aa (diag) or the configuration average of a's
    configuration (cav: array over all CSFs) for a in Q1."""
    n = op.n
    q1 = np.sort(q1)
    i1 = np.r_[np.arange(m), q1].astype(int)
    q2 = np.setdiff1d(np.arange(m, n), i1)
    H11 = op.sub(i1).dense()
    X = op.block(i1, q2) if len(q2) else sp.csr_matrix((len(i1), 0))
    d2 = op.d[q2]
    p, Am = np.linalg.eigh(Hpp)
    out, info = {}, {}
    Xc = X.tocoo()
    for kind in KINDS:
        E, Xp, wsel, gaps, dmin = [], [], [], [], []
        for j in range(m):
            Dinv = sp.diags(1.0 / (p[j] - d2))
            Hj = H11.copy()
            if not len(q2):
                pass
            elif kind.startswith('own'):
                Ea = np.r_[np.full(m, p[j]), op.d[q1] if kind == 'own-diag' else cav[q1]]
                den = Ea[Xc.row] - d2[Xc.col]
                F = sp.csr_matrix((Xc.data / den, (Xc.row, Xc.col)), shape=X.shape)
                G = (F @ X.T).toarray()
                Hj += 0.5 * (G + G.T)
                dmin.append(float(np.abs(den[Xc.row >= m]).min()) if (Xc.row >= m).any() else None)
            elif kind == 'balanced':
                Hj += (X @ Dinv @ X.T).toarray()
            else:
                Hj[:m, :m] += (X[:m] @ Dinv @ X[:m].T).toarray()
            e, C = np.linalg.eigh(Hj)
            ov = (Am[:, j] @ C[:m]) ** 2
            k = int(np.argmax(ov))
            E.append(e[k])
            Xp.append(C[:m, k])
            wsel.append(float((C[:m, k] ** 2).sum()))
            gaps.append(float(min(abs(e[k] - e[kk]) for kk in range(len(e)) if kk != k)) if len(e) > 1 else np.inf)
        Xp = np.array(Xp).T
        w, Us = np.linalg.eigh(Xp @ Xp.T)
        Sm = Us @ np.diag(w ** -0.5) @ Us.T
        out[kind] = Sm @ Xp @ np.diag(E) @ Xp.T @ Sm
        info[kind] = dict(wP=wsel, gap_eV=[g * 27.211386 for g in gaps], dD_min=float(np.min(np.abs(p[:, None] - d2[None, :])))
                          if len(d2) else None, dQ1_min=dmin or None)
    return out, info, len(i1)


def main(space, variants):
    t0 = time.time()
    ion = os.environ.get('A_ION', A.Q.ION)
    bl, eav, hf, _ = blocks6(os.environ['A_RUNFMT']) if N == 6 else A.blocks(space)
    p0 = bl[0]['lab'][0]
    cfgs = sorted(set(eav) - {p0})
    sets = q1_sets(cfgs)
    variants = variants or ['C0', 'C-a', 'C-b', 'C-d']
    res = dict(space=space, ion=ion, q1={v: sets[v] for v in variants}, ncsf_total=0, variants={})
    by = {v: {k: {} for k in KINDS} for v in variants}
    dims = {v: {} for v in variants}
    inf_all = {v: {} for v in variants}
    for b in bl:
        m, t, U = b['m'], b['t'], b['U']
        op = SymLower.from_npz(np.load(os.path.join(A.ROOT, 'results', b['name'] + '.npz'), allow_pickle=True))
        lab = b['lab']
        res['ncsf_total'] += op.n
        Hpp = op.sub(np.arange(m)).dense()
        php = U.T @ Hpp @ U
        for v in variants:
            s1 = set(sets[v])
            q1 = np.array([i for i in range(m, op.n) if lab[i] in s1], dtype=int)
            hv, inf, n1 = levels(op, m, q1, Hpp, cav=np.array([eav[la] if la in eav else np.nan for la in lab]))
            for kind in hv:
                by[v][kind][t] = hf[t] + U.T @ hv[kind] @ U - php
            dims[v][t] = dict(n1=int(n1), n=int(op.n))
            inf_all[v][t] = inf
            print(f'{ion} {space} {t} {v}: dim(P+Q1) {n1} / {op.n}, {time.time() - t0:.0f} s', flush=True)
        del op
    for v in variants:
        n1 = sum(x['n1'] for x in dims[v].values())
        nt = sum(x['n'] for x in dims[v].values())
        r = dict(dim_P_Q1=n1, dim_total=nt, frac=n1 / nt, per=dims[v], info=inf_all[v])
        for kind in KINDS:
            pr = (params6 if N == 6 else A.params)(by[v][kind])
            r[kind] = {q: float(pr[q]) for q in Q5}
        res['variants'][v] = r
        print(v, ' | '.join(f"{k} T1 {r[k]['T1']:.3f} T2 {r[k]['T2']:.3f}" for k in KINDS), f'| dim {n1} ({100 * n1 / nt:.2f} %)',
              flush=True)
    res['wall_s'] = time.time() - t0
    res['maxrss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    tag = '_'.join(variants) if variants == ['full'] else os.environ.get('C_TAG', '')
    json.dump(res, open(os.path.join(A.OUT, f'c_{ion}_{space}{"_" + tag if tag else ""}.json'), 'w'), indent=1)
    print('done', ion, space, f'{res["wall_s"]:.0f} s, maxrss {res["maxrss_MB"]:.0f} MB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
