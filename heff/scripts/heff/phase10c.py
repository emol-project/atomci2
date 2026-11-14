#!/usr/bin/env python3
"""Phase 10c: the 3s -> 3d channel on uniform S (all spaces with uniform parents).

Spaces (scripts/phase10_recon/counts.py, tspaces), p = 3s 3d^(n+1):
  T0 = S + p;  T1a = T0 + singles out of p + doubles out of p with <= 1 virtual electron;
  T2a = T1a + C3a core doubles;  T1r/T2r: T1a/T2a + doubles out of p with two virtual electrons in the
  kept core-correlating NOs (runs/vq_cno, scripts/phase10c_nos.py).
Reference: C1, C2, C3a with uniform parents (results/phase10/phase10b_part2u.json).
Per space: residual (rms, angle to R(C3a)), coefficient along u = D(K_3s:3d) = R(T0) - R(S)
(c = <R - R(S), u> / <u, u>; T0 = 1 by definition), B, C, alpha, beta, dB, dC, the weight of p in each target.

Usage: phase10c.py a          (T2a runs)         -> results/phase10/phase10c_a.txt/.json
       phase10c.py r          (T2a and T2r runs) -> results/phase10/phase10c_r.txt/.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
_C.setup_path()
_C.setup_path()
import counts as CN  # noqa: E402
import phase9 as P  # noqa: E402
import phase10 as P10  # noqa: E402
import trial  # noqa: E402
from atomci_heff.sparse import SymLower  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase10')
KEYS = ('B', 'C', 'alpha', 'beta')


def solve(op, labels, idx, n, two, pidx):
    """phase10.solve plus the weight of the CSFs pidx (positions in the full list) in the targets."""
    sub = op.sub(np.array(idx))
    lab = [labels[i] for i in idx]
    pref = {'3s': 2, '3p': 6, **({'3d': n} if n else {})}
    Pi = [k for k, lb in enumerate(lab) if P10.occ(lb) == pref]
    pos = {i: k for k, i in enumerate(idx)}
    Qi = [pos[i] for i in pidx if i in pos]
    k = min(4 if two else 2, sub.n)
    e, C, res = P10.lowest(sub, k)
    wP = (C[Pi] ** 2).sum(0)
    wQ = (C[Qi] ** 2).sum(0) if Qi else np.zeros(C.shape[1])
    if not two:
        i = int(np.argmax(wP > 0.5)) if (wP > 0.5).any() else 0
        return dict(E=float(e[i]), w=float(wP[i]), wp=float(wQ[i]), res=res)
    roots = sorted(np.argsort(-wP)[:2].tolist())
    X = C[np.ix_(Pi, roots)]
    u, sv, vt = np.linalg.svd(X)
    Xt = u @ vt
    return dict(Heff=Xt @ np.diag(e[roots]) @ Xt.T, w=[float(wP[r]) for r in roots], wp=[float(wQ[r]) for r in roots],
                condX=float(sv[0] / sv[-1]), res=res, E2=[float(e[r]) for r in roots])


def compute(fam, order):
    """fam: 'T2a' or 'T2r'; order: nested space names, e.g. ['S', 'T0', 'T1a', 'T2a']."""
    E = {s: {} for s in ['a'] + order}
    for n, t in P10.TERMS:
        run = f'vc_{fam}_{P10.ION[n]}_{t}e'
        z = np.load(os.path.join(ROOT, 'results', f'{run}.npz'))
        labels = [str(s) for s in z['labels']]
        op = SymLower.from_npz(z)
        del z
        keep = trial.KEEP if fam == 'T2r' else None
        sp = CN.tspaces(n, P10.V, t, keep=keep)[1]
        keys = [frozenset(P10.occ(lb).items()) for lb in labels]
        assert all(k in sp[order[-1]] for k in keys), run
        p = frozenset({'3s': 1, '3p': 6, '3d': n + 1}.items())
        pidx = [i for i, k in enumerate(keys) if k == p]
        two = (n == 3 and t == '2D')
        pref = {'3s': 2, '3p': 6, **({'3d': n} if n else {})}
        E['a'][(n, t)] = solve(op, labels, [i for i, lb in enumerate(labels) if P10.occ(lb) == pref], n, two, pidx)
        for s in order:
            E[s][(n, t)] = solve(op, labels, [i for i, k in enumerate(keys) if k in sp[s]], n, two, pidx)
        print(f'{run}: N={op.n}', flush=True)
    Ea = E['a']
    d2a = {t: Ea[(2, t)]['E'] - sum(P.g(x) * Ea[(2, x)]['E'] for x in P.T2) / 45 for t in P.T2}
    Ha = Ea[(3, '2D')]['Heff']
    Eav3a = (sum(P.g(t) * Ea[(3, t)]['E'] for t in P.T3) + 10 * np.trace(Ha)) / 120
    U, mis = P.seniority_map(Ha, Eav3a * np.eye(2) + P.pairsum(d2a)['2D'])
    assert mis < 1e-8, mis
    return {s: P10.decompose(E[s], U) for s in order}, E


def rms(x):
    return float(np.sqrt(P.ip(x, x) / 120))


def main(mode):
    ref = json.load(open(os.path.join(OUT, 'phase10b_part2u.json')))['decomposition']
    RR = {s: np.asarray(ref[s]['R']) for s in ('S', 'C1', 'C2', 'C3a', 'S+K_3s:3d')}
    D, E = compute('T2a', ['S', 'T0', 'T1a', 'T2a'])
    seq = ['S', 'T0', 'T1a', 'T2a']
    if mode == 'r':
        trial.KEEP = set(json.load(open(os.path.join(OUT, 'phase10c_keep.json')))['keep'])
        Dr, Er = compute('T2r', ['S', 'T0', 'T1r', 'T2r'])
        chk_r = max(abs(Dr[s]['E2'][t] - D[s]['E2'][t]) for s in ('S', 'T0') for t in P.T2)
        D.update({s: Dr[s] for s in ('T1r', 'T2r')})
        E.update({s: Er[s] for s in ('T1r', 'T2r')})
        seq += ['T1r', 'T2r']
    R = {s: np.asarray(D[s]['R']) for s in seq}
    u = R['T0'] - R['S']
    chkS = float(np.abs(R['S'] - RR['S']).max())
    chkT0 = float(np.abs(u - (RR['S+K_3s:3d'] - RR['S'])).max())
    w = []
    a = w.append
    a('# Phase 10c: the 3s -> 3d channel (p = 3s 3d^(n+1)) on uniform S; V QZP, [Ne] frozen; mEh')
    a(f'\nChecks: R(S) vs phase10b_part2u {chkS:.1e} Eh; R(T0) - R(S) vs D(K_3s:3d) of part2u {chkT0:.1e} Eh'
      + (f'; S, T0 energies of the T2r runs (NO orbitals) vs T2a runs {chk_r:.1e} Eh' if mode == 'r' else ''))
    a(f'u = R(T0) - R(S): rms {1e3 * rms(u):.4f} mEh.')
    a('\n## Spaces')
    a('| space | rms residual | angle to R(C3a) | coefficient along u | perp. rms | B | C | alpha | beta | dB | dC | dbeta | 3rd diff |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows = [(s, R[s], D[s]) for s in seq] + [(s + ' (part2u)', RR[s], ref[s]) for s in ('C1', 'C2', 'C3a')]
    coef = {}
    for nm, r, d in rows:
        x = r - RR['S']
        c = P.ip(x, u) / P.ip(u, u)
        coef[nm] = c
        perp = x - c * u
        a(f'| {nm} | {1e3 * rms(r):.4f} | {P.wangle(r, RR["C3a"]):.1f} | {c:.3f} | {1e3 * rms(perp):.4f} | '
          + ' | '.join(f'{1e3 * d["par"][k]:.4f}' for k in KEYS) + ' | '
          + ' | '.join(f'{1e3 * d["dpar"][k]:+.4f}' for k in ('B', 'C', 'beta')) + f' | {1e3 * d["third"]:+.3f} |')
    a('\n## Weight of p = 3s 3d^(n+1) in each target (and w(3s2 3p6 3d^n))')
    a('| term | ' + ' | '.join(seq[1:]) + ' |')
    a('|---|' + '---|' * (len(seq) - 1))
    for n, t in P10.TERMS:
        cells = []
        for s in seq[1:]:
            r = E[s][(n, t)]
            if isinstance(r['wp'], list):
                cells.append('/'.join(f'{x:.5f}' for x in r['wp']) + ' (' + '/'.join(f'{x:.3f}' for x in r['w']) + ')')
            else:
                cells.append(f'{r["wp"]:.5f} ({r["w"]:.3f})')
        a(f'| d{n} {t} | ' + ' | '.join(cells) + ' |')
    a('\n## Channel energy per term (E(T) - E(S)), mEh')
    a('| term | ' + ' | '.join(seq[1:]) + ' |')
    a('|---|' + '---|' * (len(seq) - 1))
    for n, t in P10.TERMS:
        if n == 3 and t == '2D':
            for i in range(2):
                ev = {s: np.linalg.eigvalsh(D[s]['H2D'])[i] for s in seq}
                a(f'| d3 2D{"-+"[i]} | ' + ' | '.join(f'{1e3 * (ev[s] - ev["S"]):+.3f}' for s in seq[1:]) + ' |')
        else:
            a(f'| d{n} {t} | ' + ' | '.join(f'{1e3 * (E[s][(n, t)]["E"] - E["S"][(n, t)]["E"]):+.3f}' for s in seq[1:]) + ' |')
    tag = f'phase10c_{mode}'
    open(os.path.join(OUT, f'{tag}.txt'), 'w').write('\n'.join(w) + '\n')
    js = dict(coef=coef, R={s: R[s].tolist() for s in seq}, u=u.tolist(),
              par={s: {k: float(D[s]['par'][k]) for k in KEYS} for s in seq},
              dpar={s: {k: float(D[s]['dpar'][k]) for k in KEYS} for s in seq},
              rms_R={s: rms(R[s]) for s in seq})
    json.dump(js, open(os.path.join(OUT, f'{tag}.json'), 'w'), indent=1)
    print('\n'.join(w))


if __name__ == '__main__':
    main(sys.argv[1])
