#!/usr/bin/env python3
"""Phase 11a: perturbative counterparts of the 3s -> 3d three-body term (V QZP, [Ne] frozen, vq_orb set 2).

Per term (all 14 targets; d0/d1 only enter the third difference), from the existing vc_T2a H exports
(which contain T0 = S + 3s 3d^(n+1) as a subspace):
  P = 3s2 3p6 3d^n CSFs, Q = 3s 3p6 3d^(n+1) CSFs of the same LS (none for d0 1S, d2 1S).
Step 0: Q (and P) counts from coupling rules (d^m terms by determinant M_L/M_S counting, times 3s),
  asserted equal to the CSFs present in the export.
Effective Hamiltonians on P (Hermitian):
  A  CI(P): H_PP
  B  closure: H_PP - H_PQ H_QP / Delta. Delta_n = E_avQ(n) - E_av(3d^n), g-weighted averages of the
     diagonal H; E_avQ is over the Q CSFs of the target LS only (the other LS of 3s 3d^(n+1) are not in any
     export). B_sep: Delta_n per n; B_com: one Delta = mean(Delta_2, Delta_3); B_com x0.5 / x2: scaling check.
  C  H_PP + sum_q H_Pq H_qP / (E_av(3d^n) - H_qq)
  D  symmetrized EN: H_PP + sum_q H_pq H_qp' (1/(H_pp - H_qq) + 1/(H_p'p' - H_qq)) / 2
  E  CI(P + Q), des Cloizeaux on P (phase10c.solve)
Decomposition, 2D seniority map, metric and q basis: scripts/phase9.py / phase10.py (P10.decompose), U from A.
u = R(T0) - R(S) from results/phase10/phase10c_r.json.

Usage: phase11a.py   -> results/phase11/phase11a.txt/.json
"""
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase9 as P  # noqa: E402
import phase10 as P10  # noqa: E402
from phase10c import rms, solve  # noqa: E402
from atomci_heff.sparse import SymLower  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase11')
KEYS = ('B', 'C', 'alpha', 'beta')


# ---------------------------------------------------------------- step 0: counts from coupling rules
def d_terms(m):
    """{(L, 2S): multiplicity} of d^m, from determinant (M_L, 2M_S) counting."""
    so = [(ml, s) for ml in range(-2, 3) for s in (1, -1)]
    N = Counter()
    for det in itertools.combinations(so, m):
        N[(sum(x[0] for x in det), sum(x[1] for x in det))] += 1
    out = {}
    for (ML, MS2), _ in N.items():
        if ML < 0 or MS2 < 0:
            continue
        k = N[(ML, MS2)] - N[(ML + 1, MS2)] - N[(ML, MS2 + 2)] + N[(ML + 1, MS2 + 2)]
        if k:
            out[(ML, MS2)] = k
    return out


def n_q(n, L, S2):
    """Number of 3s 3d^(n+1) CSFs of (L, 2S): d^(n+1) parents (L, 2Sp) coupled with s, 2S = 2Sp +- 1."""
    return sum(k for (Lp, Sp2), k in d_terms(n + 1).items() if Lp == L and abs(Sp2 - S2) == 1)


def n_p(n, L, S2):
    return d_terms(n).get((L, S2), 0) if n else int((L, S2) == (0, 0))


# ---------------------------------------------------------------- effective Hamiltonians
def blocks(n, t):
    z = np.load(os.path.join(ROOT, 'results', f'vc_T2a_{P10.ION[n]}_{t}e.npz'))
    labels = [str(s) for s in z['labels']]
    op = SymLower.from_npz(z)
    pref = {'3s': 2, '3p': 6, **({'3d': n} if n else {})}
    qref = {'3s': 1, '3p': 6, '3d': n + 1}
    Pi = [i for i, lb in enumerate(labels) if P10.occ(lb) == pref]
    Qi = [i for i, lb in enumerate(labels) if P10.occ(lb) == qref]
    idx = sorted(Pi + Qi)
    H = op.sub(np.array(idx)).dense()
    pos = {i: k for k, i in enumerate(idx)}
    p, q = [pos[i] for i in Pi], [pos[i] for i in Qi]
    return dict(op=op, labels=labels, Pi=Pi, Qi=Qi, idx=idx, HPP=H[np.ix_(p, p)], HPQ=H[np.ix_(p, q)],
                hq=np.diag(H)[q])


def pack(Heff, two):
    Heff = 0.5 * (Heff + Heff.T)
    return dict(Heff=Heff) if two else dict(E=float(Heff[0, 0]))


def main():
    os.makedirs(OUT, exist_ok=True)
    w = []
    a = w.append
    a('# Phase 11a: perturbative counterparts of the 3s -> 3d three-body term; V QZP, [Ne] frozen, vq_orb set 2; mEh')
    a('P = 3s2 3p6 3d^n CSFs, Q = 3s 3p6 3d^(n+1) CSFs of the same LS; exports vc_T2a (contain T0).')

    # step 0
    a('\n## Step 0: CSF counts (coupling rules vs export)')
    a('| term | P rule | P export | Q rule | Q export |')
    a('|---|---|---|---|---|')
    B = {}
    for n, t in P10.TERMS:
        L, S2 = 'SPDFGH'.index(t[1]), int(t[0]) - 1
        b = blocks(n, t)
        B[(n, t)] = b
        rp, rq = n_p(n, L, S2), n_q(n, L, S2)
        a(f'| d{n} {t} | {rp} | {len(b["Pi"])} | {rq} | {len(b["Qi"])} |')
        print(w[-1], flush=True)
        assert (rp, rq) == (len(b['Pi']), len(b['Qi'])), (n, t, rp, rq, len(b['Pi']), len(b['Qi']))
    a('All counts agree.')

    # averages and Delta
    Eav, EavQ = {}, {}
    for n in (0, 1, 2, 3):
        tl = [t for m, t in P10.TERMS if m == n]
        gp = sum(P.g(t) * B[(n, t)]['HPP'].trace() for t in tl)
        gn = sum(P.g(t) * len(B[(n, t)]['Pi']) for t in tl)
        Eav[n] = gp / gn
        qn = sum(P.g(t) * len(B[(n, t)]['Qi']) for t in tl)
        EavQ[n] = sum(P.g(t) * B[(n, t)]['hq'].sum() for t in tl) / qn if qn else None
    Dl = {n: EavQ[n] - Eav[n] for n in (1, 2, 3)}
    Dcom = 0.5 * (Dl[2] + Dl[3])

    def closure(D):
        return lambda n, b: b['HPP'] - b['HPQ'] @ b['HPQ'].T / (D if np.isscalar(D) else D[n])

    def termdep(n, b):
        return b['HPP'] + (b['HPQ'] / (Eav[n] - b['hq'])[None, :]) @ b['HPQ'].T

    def en(n, b):
        hp = np.diag(b['HPP'])
        R = 1.0 / (hp[:, None] - b['hq'][None, :])            # p x q
        V = b['HPQ']
        return b['HPP'] + 0.5 * ((V * R) @ V.T + V @ (V * R).T)

    variants = {'A': lambda n, b: b['HPP'], 'B_sep': closure(Dl), 'B_com': closure(Dcom),
                'B_com x0.5': closure(0.5 * Dcom), 'B_com x2': closure(2 * Dcom), 'C': termdep, 'D': en}
    E = {k: {} for k in list(variants) + ['E']}
    for (n, t), b in B.items():
        two = (n == 3 and t == '2D')
        for k, f in variants.items():
            if n == 0 or not b['Qi']:
                E[k][(n, t)] = pack(b['HPP'], two)
            else:
                E[k][(n, t)] = pack(f(n, b), two)
        r = solve(b['op'], b['labels'], b['idx'], n, two, [])
        E['E'][(n, t)] = dict(Heff=r['Heff']) if two else dict(E=r['E'])
    Ea = E['A']
    d2a = {t: Ea[(2, t)]['E'] - sum(P.g(x) * Ea[(2, x)]['E'] for x in P.T2) / 45 for t in P.T2}
    Ha = Ea[(3, '2D')]['Heff']
    Eav3a = (sum(P.g(t) * Ea[(3, t)]['E'] for t in P.T3) + 10 * np.trace(Ha)) / 120
    U, mis = P.seniority_map(Ha, Eav3a * np.eye(2) + P.pairsum(d2a)['2D'])
    assert mis < 1e-8, mis
    Dc = {k: P10.decompose(E[k], U) for k in E}
    RA = np.asarray(Dc['A']['R'])
    u = np.asarray(json.load(open(os.path.join(ROOT, 'results', 'phase10', 'phase10c_r.json')))['u'])

    a(f'\nCheck: rms R(A) = {1e3 * rms(RA):.2e} mEh (3d^n alone is exactly two-body); 2D seniority map mismatch {mis:.1e} Eh.')
    a('\n## Delta (Eh)')
    a(f'E_av(3d^n): ' + ', '.join(f'n={n} {Eav[n]:.6f}' for n in (0, 1, 2, 3)))
    a(f'E_avQ (target-LS Q CSFs only): ' + ', '.join(f'n={n} {EavQ[n]:.6f}' for n in (1, 2, 3)))
    a('Delta_n = E_avQ - E_av(3d^n): ' + ', '.join(f'n={n} {Dl[n]:.6f} Eh ({Dl[n] * P.EV:.3f} eV)' for n in (1, 2, 3))
      + f'; common Delta = mean(n=2, 3) = {Dcom:.6f} Eh ({Dcom * P.EV:.3f} eV)')

    a('\n## 1. Three-body vector delta = R(X) - R(A) vs u = R(T0) - R(S) (rms u 1.2735 mEh)')
    a('| X | rms delta | angle to u | <delta,u>/<u,u> | angle to delta(E) | R3 (q1..q4) |')
    a('|---|---|---|---|---|---|')
    dE = np.asarray(Dc['E']['R']) - RA
    res = {}
    for k in ['B_sep', 'B_com', 'B_com x0.5', 'B_com x2', 'C', 'D', 'E']:
        d = np.asarray(Dc[k]['R']) - RA
        c = P.ip(d, u) / P.ip(u, u)
        res[k] = dict(rms=rms(d), angle_u=P.wangle(d, u), coef_u=c, angle_E=P.wangle(d, dE),
                      R3=[P.ip(d, q) for q in P.Q3])
        a(f'| {k} | {1e3 * rms(d):.4f} | {P.wangle(d, u):.1f} | {c:.3f} | {P.wangle(d, dE):.1f} | '
          + ', '.join(f'{1e3 * x:+.3f}' for x in res[k]['R3']) + ' |')
    a('| u | 1.2735 | 0.0 | 1.000 | ' + f'{P.wangle(u, dE):.1f} | ' + ', '.join(f'{1e3 * P.ip(u, q):+.3f}' for q in P.Q3) + ' |')
    a(f'delta(E) vs u: rms ratio {rms(dE) / rms(u):.3f}, angle {P.wangle(dE, u):.1f} deg, coefficient '
      f'{res["E"]["coef_u"]:.3f}. delta(E) is on top of P alone, u on top of S: the difference is the effect of S.')

    a('\n## 2. Two-body content: d2 parameters (X - A), d3 refit dpar, rms of t3\' and of R')
    a('| X | dB(d2) | dC(d2) | dalpha(d2) | dbeta(d2) | refit dB | refit dC | refit dalpha | refit dbeta | rms t3\' | rms R | 3rd diff |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|')
    two = {}
    for k in ['A', 'B_sep', 'B_com', 'C', 'D', 'E']:
        d = Dc[k]
        dp = {x: d['par'][x] - Dc['A']['par'][x] for x in KEYS}
        two[k] = dict(dpar_d2=dp, refit=dict(d['dpar']), rms_t3=float(d['rms_t3']), rms_R=float(d['rms_R']),
                      third=float(d['third']))
        a(f'| {k} | ' + ' | '.join(f'{1e3 * dp[x]:+.4f}' for x in KEYS) + ' | '
          + ' | '.join(f'{1e3 * d["dpar"][x]:+.4f}' for x in KEYS)
          + f' | {1e3 * d["rms_t3"]:.4f} | {1e3 * d["rms_R"]:.4f} | {1e3 * d["third"]:+.3f} |')

    a('\n## 3. Delta-independence of the closure direction')
    r1, r05, r2 = (np.asarray(Dc[k]['R']) - RA for k in ('B_com', 'B_com x0.5', 'B_com x2'))
    a(f'angle(B_com, B_com x0.5) {P.wangle(r1, r05):.2e} deg, angle(B_com, B_com x2) {P.wangle(r1, r2):.2e} deg; '
      f'rms ratios x0.5: {rms(r05) / rms(r1):.6f}, x2: {rms(r2) / rms(r1):.6f}')
    a(f'angle(B_sep, B_com) {P.wangle(np.asarray(Dc["B_sep"]["R"]) - RA, r1):.2f} deg (different Delta for d2 and d3)')
    txt = '\n'.join(w) + '\n'
    open(os.path.join(OUT, 'phase11a.txt'), 'w').write(txt)
    json.dump(dict(Delta=Dl, Delta_common=Dcom, E_av=Eav, E_avQ=EavQ, results=res, two_body=two,
                   R={k: np.asarray(Dc[k]['R']).tolist() for k in Dc}), open(os.path.join(OUT, 'phase11a.json'), 'w'),
              indent=1, default=float)
    print(txt)


if __name__ == '__main__':
    main()
