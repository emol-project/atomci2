#!/usr/bin/env python3
"""Phase 21 addendum A1 / A3 (stored H only, no ATOMCI runs). Spaces as phase21a (configuration subsets of the stored
u3c1 run, RAS k = 1 + h): k1fh, A = k1fh + the job-A class (3s1 3p4, nv <= 2, level <= 2 from R2 = 3s 3p6 3d^(N+1)).

A1 (3s-hole root): per term block, the lowest root dominated by R2 (3s 3p6 3d^(N+1)) in k1fh and in A, by a
root-homing Davidson on H(space) (target: the R2 CSFs; Ritz vector = the lowest one with R2 weight >= 1/2 of the
largest; diagonal preconditioner), and the exact P roots (eigenvalues of the des Cloizeaux H_eff, iterative.heff_block).
Gap = E(3s-hole root) - E(P root), per P root; narrowing = gap(A) - gap(k1fh).

A3 (Cr3+): in A, with the class = the CSFs of A not in k1fh,
  (i)   exact                 heff_block on H(A)
  (ii)  class in Q2           a) S2-diag / C2: exact H_eff(k1fh) + the additive second-order increments of the class
                                 (phase17a formulas: D = p_j - H_qq / closure Ebar_P - Ebar_c(q))
                              b) balanced: Q1 = all of Q(k1fh), Q2 = the class, D_j = p_j - H_qq (phase17c_scan.heff)
  (iii) Q1_rule               phase17c_scan rule (rset 'R') on the A configurations, rest in Q2 (balanced),
                              and Q1_rule + class; reference: Q1_rule on k1fh (class absent)
All placed on the vcm HF P block (HF + H_eff - PHP) and projected (phase19b.decompose).
usage: phase21e.py root <ion>        (A1 3s-hole root, k1fh and A)
       phase21e.py a3                (A3, Cr3+)                -> phase21e.json"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
OUT = os.path.join(ROOT, 'results', 'phase21')
JS = os.path.join(OUT, 'phase21e.json')
sys.path[:0] = [OUT]
import phase21a as A21  # noqa: E402

EH = 219474.6314


def jload():
    return json.load(open(JS)) if os.path.exists(JS) else {}


def jsave(d):
    json.dump(d, open(JS, 'w'), indent=1, default=float)


class SubOp:
    def __init__(self, op, idx):
        self.op, self.idx, self.n = op, idx, len(idx)
        self.d = op.d[idx]

    def matvec(self, X):
        X = X.reshape(self.n, -1)
        Z = np.zeros((self.op.n, X.shape[1]))
        Z[self.idx] = X
        return self.op.matvec(Z)[self.idx]


def homing(op, tgt, tol=1e-5, maxit=400, msub=80, keep=12):
    """lowest root of op dominated by the CSFs tgt (local indices)."""
    n = op.n
    V = np.zeros((n, len(tgt)))
    V[tgt, np.arange(len(tgt))] = 1.0
    AV = op.matvec(V)
    for it in range(maxit):
        Hs = V.T @ AV
        th, S = np.linalg.eigh(0.5 * (Hs + Hs.T))
        w = (V[tgt] @ S) ** 2
        w = w.sum(0)
        k = int(np.min(np.where(w >= 0.5 * w.max())[0]))
        x, ax = V @ S[:, k], AV @ S[:, k]
        r = ax - th[k] * x
        rn = float(np.linalg.norm(r))
        if rn < tol:
            break
        den = th[k] - op.d
        den[np.abs(den) < 1e-3] = np.sign(den[np.abs(den) < 1e-3] + 1e-30) * 1e-3
        t = r / den
        if V.shape[1] >= msub:
            sel = np.argsort(-w)[:keep]
            sel = np.unique(np.r_[k, sel])
            V, AV = V @ S[:, sel], AV @ S[:, sel]
        for _ in range(2):
            t -= V @ (V.T @ t)
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            break
        t /= nt
        V = np.c_[V, t]
        AV = np.c_[AV, op.matvec(t[:, None])]
    return dict(E=float(th[k]), wR2=float(w[k]), res=rn, it=it + 1)


def load(c, t):
    from atomci_heff.sparse import SymLower
    k, u = A21.runs(c, t)
    lk = A21.labels(k)
    z = np.load(os.path.join(ROOT, 'results', u + '.npz'), allow_pickle=True)
    lu = [str(x) for x in z['labels']]
    op = SymLower.from_npz(z)
    del z
    sel, _ = A21.select(c, lu, set(lk))
    return op, lu, sel


def r2_label(lab, N):
    d = A21.occ(lab)
    return (A21.nv(d) == 0 and (d['1s'], d['2s'], d['2p'], d['3s'], d['3p'], d['3d']) == (2, 2, 6, 1, 6, N + 1))


def root(ion):
    c = A21.setup(ion)
    import phase19b as P19
    from atomci_heff import iterative as IT
    N = c['N']
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    res, t0 = {}, time.time()
    for L, twoS, m, t in [x for x in P19.terms(N) if not os.environ.get('E_TERMS') or x[3] in os.environ['E_TERMS'].split(',')]:
        op, lu, sel = load(c, t)
        U = np.array(maps[f'vcm_d{N}_{t}e'])
        res[t] = {}
        for s in ('k1fh', 'A'):
            idx = sel[s]
            sub = op.sub(idx)
            ex, inf = IT.heff_block(dict(P=list(range(m)), Q=list(range(m, sub.n)), m=m, U=U), sub)
            eP = np.linalg.eigvalsh(ex)
            tgt = np.array([j for j, i in enumerate(idx) if r2_label(lu[i], N)])
            h = homing(SubOp(op, idx), tgt)
            res[t][s] = dict(n=int(len(idx)), nR2=int(len(tgt)), EP=eP.tolist(), E3s=h['E'], wR2=h['wR2'], res=h['res'],
                             it=h['it'], gap=[(h['E'] - e) for e in eP], gap_cm=[(h['E'] - e) * EH for e in eP],
                             diagR2=float(sub.d[tgt].min()))
            print(f"{ion} {t} {s}: n {len(idx)} E3s {h['E']:.6f} wR2 {h['wR2']:.3f} res {h['res']:.1e} it {h['it']} "
                  f"gap {[round(g * EH) for g in res[t][s]['gap']]} cm-1 ({time.time() - t0:.0f} s)", flush=True)
            del sub
        res[t]['narrow_cm'] = [(a - b) * EH for a, b in zip(res[t]['A']['gap'], res[t]['k1fh']['gap'])]
        print(f"{ion} {t}: gap(A) - gap(k1fh) {[round(x) for x in res[t]['narrow_cm']]} cm-1", flush=True)
        del op
    d = jload()
    d.setdefault('root', {})[ion] = res
    jsave(d)


def a3():
    c = A21.setup('cr')
    import phase19b as P19
    from atomci_heff import iterative as IT
    _C.setup_path()
    import phase17c_scan as CS
    N = 3
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hf = P19.hf_blocks(dict(N=N))
    V = ['exact:k1fh', 'exact:A', 'S2-diag', 'C2', 'bal:k1fhQ1', 'R:k1fh', 'R:A', 'R+class:A']
    by = {v: {} for v in V}
    info = {v: {} for v in V}
    keep = {}
    acc = {}
    t0 = time.time()
    for L, twoS, m, t in [x for x in P19.terms(N) if not os.environ.get('E_TERMS') or x[3] in os.environ['E_TERMS'].split(',')]:
        op, lu, sel = load(c, t)
        U = np.array(maps[f'vcm_d{N}_{t}e'])
        g = (2 * L + 1) * (twoS + 1)
        for i, la in enumerate(lu):            # configuration averages over the u3c1 CSFs (C2; as phase17a, all terms)
            s_, w = acc.get(la, (0.0, 0.0))
            acc[la] = (s_ + g * op.d[i], w + g)
        iA, ik = sel['A'], sel['k1fh']
        sub = op.sub(iA)
        del op
        labA = [lu[i] for i in iA]
        kpos = np.where(np.isin(iA, ik))[0]                      # k1fh CSFs inside A (sorted, P first)
        cl = np.where(~np.isin(iA, ik))[0]                       # the class
        assert list(kpos[:m]) == list(range(m)) and len(kpos) == len(ik)
        Hpp = sub.sub(np.arange(m)).dense()
        php = U.T @ Hpp @ U
        ex, inf = IT.heff_block(dict(P=list(range(m)), Q=list(range(m, sub.n)), m=m, U=U), sub)
        by['exact:A'][t] = hf[t] + ex - php
        info['exact:A'][t] = dict(wP=inf['wP'], n=int(sub.n))
        nk = len(kpos)
        subk = sub.sub(kpos)
        labk = [labA[i] for i in kpos]
        exk, infk = IT.heff_block(dict(P=list(range(m)), Q=list(range(m, nk)), m=m, U=U), subk)
        by['exact:k1fh'][t] = hf[t] + exk - php
        info['exact:k1fh'][t] = dict(wP=infk['wP'], n=nk)
        # (ii-a) additive increments of the class, P = first m
        Vpc = sub.block(np.arange(m), cl).toarray()
        p, Am = np.linalg.eigh(Hpp)
        keep[t] = dict(V=Vpc, dq=sub.d[cl].copy(), p=p, A=Am, U=U, labs=[labA[i] for i in cl], lab0=labA[0], exk=exk, php=php, m=m)
        # (ii-b) balanced, Q1 = Q(k1fh)
        h, ib = CS.heff(sub, m, kpos[m:], Hpp)
        by['bal:k1fhQ1'][t] = hf[t] + U.T @ h @ U - php
        info['bal:k1fhQ1'][t] = dict(wP=ib['wP'], it=ib['it'])
        # (iii) Q1_rule
        cfgA = sorted(set(labA[m:]))
        R = CS.rset('R', cfgA, N)
        clc = set(keep[t]['labs'])
        q1R = np.array([i for i in range(m, sub.n) if labA[i] in R], dtype=int)
        q1Rc = np.array([i for i in range(m, sub.n) if labA[i] in R or labA[i] in clc], dtype=int)
        q1Rk = np.array([i for i in range(m, nk) if labk[i] in R], dtype=int)
        for v, op_, q1 in (('R:A', sub, q1R), ('R+class:A', sub, q1Rc), ('R:k1fh', subk, q1Rk)):
            h, ir = CS.heff(op_, m, q1, Hpp)
            by[v][t] = hf[t] + U.T @ h @ U - php
            info[v][t] = dict(n1=ir['n1'], wP=ir['wP'], it=ir['it'])
        info.setdefault('class_in_R', {})[t] = sorted(clc & R)
        e0 = np.linalg.eigvalsh(by['exact:A'][t])
        print(t, ' '.join(f"{v} {np.round((np.linalg.eigvalsh(by[v][t]) - e0) * 219474.63, 1).tolist()}" for v in V
                          if t in by[v] and v != 'exact:A'), 'cm-1 vs exact:A', flush=True)
        print(f'{t}: n(A) {sub.n} n(k1fh) {nk} class CSFs {len(cl)}, R cfgs {len(R)} (class in R: {len(clc & R)}) '
              f'{time.time() - t0:.0f} s', flush=True)
        del sub, subk
    eav = {k: s_ / w for k, (s_, w) in acc.items()}
    for t, k in keep.items():
        m, Vpc, p, Am, U = k['m'], k['V'], k['p'], k['A'], k['U']
        W = Am.T @ Vpc
        p0 = None
        for meth in ('S2-diag', 'C2'):
            if meth == 'S2-diag':
                Dn = p[:, None] - k['dq'][None, :]
            else:
                p0 = eav[k['lab0']]
                Dn = np.broadcast_to(p0 - np.array([eav[x] for x in k['labs']]), W.shape)
            Y = W / Dn
            G = Vpc @ Y.T @ Am.T
            G = 0.5 * (G + G.T)
            by[meth][t] = hf[t] + k['exk'] + U.T @ G @ U - k['php']   # fixed 13 Nov (was U^T (exk + G) U; see a3add)
    d = jload()
    out = {}
    for v in V:
        pr = P19.decompose(dict(N=N), {(L, twoS): by[v][t] for L, twoS, m, t in P19.terms(N)})
        out[v] = dict(T1=float(pr['T1']), T2=float(pr['T2']), B=float(pr['B']), C=float(pr['C']), info=info.get(v))
    ex = out['exact:A']
    dex = ex['T1'] - out['exact:k1fh']['T1']
    for v in V:
        out[v]['err_T1'] = out[v]['T1'] - ex['T1']
        out[v]['err_T2'] = out[v]['T2'] - ex['T2']
    rec = {'S2-diag': out['S2-diag']['T1'] - out['exact:k1fh']['T1'], 'C2': out['C2']['T1'] - out['exact:k1fh']['T1'],
           'bal:k1fhQ1': out['bal:k1fhQ1']['T1'] - out['exact:k1fh']['T1'],
           'R+class vs R:k1fh': out['R+class:A']['T1'] - out['R:k1fh']['T1'], 'R:A vs R:k1fh': out['R:A']['T1'] - out['R:k1fh']['T1']}
    out['recovery'] = {k: dict(dT1=v, frac=v / dex) for k, v in rec.items()}
    out['dT1_exact'] = dex
    out['class_in_R'] = info.get('class_in_R')
    d['a3'] = out
    jsave(d)
    for v in V:
        print(f"{v:12s} T1 {out[v]['T1']:.4f} T2 {out[v]['T2']:.4f} err {out[v]['err_T1']:+.4f} / {out[v]['err_T2']:+.4f}")
    for k, v in out['recovery'].items():
        print(f'recovery {k:20s} dT1 {v["dT1"]:+.4f} = {100 * v["frac"]:.0f} % of exact {dex:+.4f}')


def a3add():
    """(ii-a) redone (13 Nov): a3 put the canonical-basis exact H_eff(k1fh) through U twice in the S2-diag / C2 rows.
    Here only the class increments G (CSF basis) are recomputed; T = T(exact:k1fh) + [dec(hf + U^T G U) - dec(hf)]
    (the projection is linear: checked with 2G)."""
    c = A21.setup('cr')
    import phase19b as P19
    N = 3
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hf = P19.hf_blocks(dict(N=N))
    acc, keep, t0 = {}, {}, time.time()
    for L, twoS, m, t in P19.terms(N):
        op, lu, sel = load(c, t)
        g = (2 * L + 1) * (twoS + 1)
        for i, la in enumerate(lu):
            s_, w = acc.get(la, (0.0, 0.0))
            acc[la] = (s_ + g * op.d[i], w + g)
        iA, ik = sel['A'], sel['k1fh']
        cl = iA[~np.isin(iA, ik)]
        Hpp = op.sub(np.arange(m)).dense()
        p, Am = np.linalg.eigh(Hpp)
        keep[t] = dict(V=op.block(np.arange(m), cl).toarray(), dq=op.d[cl].copy(), p=p, A=Am, m=m, lab0=lu[0],
                       labs=[lu[i] for i in cl], U=np.array(maps[f'vcm_d{N}_{t}e']), L=L, twoS=twoS)
        print(t, len(cl), f'{time.time() - t0:.0f} s', flush=True)
        del op
    eav = {k: s_ / w for k, (s_, w) in acc.items()}
    d = jload()
    out = d['a3']
    tl = P19.terms(N)
    base = P19.decompose(dict(N=N), {(L, twoS): hf[t] for L, twoS, m, t in tl})
    for meth in ('S2-diag', 'C2'):
        inc = {}
        for t, k in keep.items():
            W = k['A'].T @ k['V']
            if meth == 'S2-diag':
                Dn = k['p'][:, None] - k['dq'][None, :]
            else:
                Dn = np.broadcast_to(eav[k['lab0']] - np.array([eav[x] for x in k['labs']]), W.shape)
            G = k['V'] @ (W / Dn).T @ k['A'].T
            inc[t] = k['U'].T @ (0.5 * (G + G.T)) @ k['U']
        p1 = P19.decompose(dict(N=N), {(L, twoS): hf[t] + inc[t] for L, twoS, m, t in tl})
        p2 = P19.decompose(dict(N=N), {(L, twoS): hf[t] + 2 * inc[t] for L, twoS, m, t in tl})
        lin = max(abs((p2[q] - base[q]) - 2 * (p1[q] - base[q])) for q in ('T1', 'T2'))
        r = {q: out['exact:k1fh'][q] + p1[q] - base[q] for q in ('T1', 'T2', 'B', 'C')}
        r.update(err_T1=r['T1'] - out['exact:A']['T1'], err_T2=r['T2'] - out['exact:A']['T2'],
                 incr_T1=p1['T1'] - base['T1'], incr_T2=p1['T2'] - base['T2'], lin_check=lin, fixed='a3add 13 Nov')
        out[meth] = r
        out['recovery'][meth] = dict(dT1=r['incr_T1'], frac=r['incr_T1'] / out['dT1_exact'])
        print(f"{meth}: T1 {r['T1']:.4f} T2 {r['T2']:.4f} err {r['err_T1']:+.4f} / {r['err_T2']:+.4f} incr T1 "
              f"{r['incr_T1']:+.4f} = {100 * r['incr_T1'] / out['dT1_exact']:.0f} % of {out['dT1_exact']:+.4f} (lin {lin:.1e})")
    jsave(d)


if __name__ == '__main__':
    if sys.argv[1] == 'a3add':
        a3add()
    elif sys.argv[1] == 'root':
        root(sys.argv[2])
    elif sys.argv[1] == 'a3':
        a3()
