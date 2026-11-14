#!/usr/bin/env python3
"""Phase 10: core (3s, 3p) contributions to the d^3 three-body residual, V QZP, [Ne] frozen.

Runs vc_C2_<ion>_<term> (scripts/phase10_recon/trial.py; d0 uses vc_C1_v5p_1Se, where C2 = C1).
Every space is a subspace of the term's C2 CSF list, selected by configuration:
  a     3s2 3p6 3d^n
  S     Phase 9 level s with 3s2 3p6 closed (reproduces Phase 9b)
  K_3s  one 3s hole (3s -> 3d, ns, nd, ng), out of a b configuration
  K_pp, K_pf, K_ph   one 3p hole, the electron in a virtual p, f, h orbital (out of b)
  K_cv  C2 only: one core hole and one valence move (C2 minus C1)
  C1 = S + K_3s + K_pp + K_pf + K_ph;  C2 = C1 + K_cv
Targets as in Phase 9 (lowest root; 2D: des Cloizeaux H_eff on the 3s2 3p6 3d3 2D model space from
the two roots of largest model-space weight, in the seniority basis identified at level a).
Decomposition, refit, residual and q basis: scripts/phase9.py (same functions).
Eigenvalues: Davidson (atomci_heff.sparse) above 300 CSFs, dense below.

Usage: scripts/phase10.py   -> results/phase10/phase10.txt/.json
"""
import json
import os
import re
import struct
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
from atomci_heff.reader import records  # noqa: E402
from atomci_heff.sparse import SymLower, davidson  # noqa: E402

V = (7, 6, 6, 4, 3, 2)
CLS = ['K_3s', 'K_pp', 'K_pf', 'K_ph', 'K_cv']
ION = {0: 'v5p', 1: 'v4p', 2: 'v3p', 3: 'v2p'}
TERMS = [(0, '1S'), (1, '2D')] + [(2, t) for t in P.T2] + [(3, t) for t in P.T3 + ['2D']]
KEYS = ('B', 'C', 'alpha', 'beta')


def occ(label):
    return {m.group(1): int(m.group(2)) for m in (re.match(r'(\d+[spdfgh])(\d+)', s) for s in label.split())}


def classify(labels, n, term):
    L, sp = CN.spaces(n, V, term)
    C1 = sp['C1']
    S = sp['S']
    out = []
    for lb in labels:
        o = occ(lb)
        key = frozenset(o.items())
        qs, qp = o.get('3s', 0), o.get('3p', 0)
        if qs == 2 and qp == 6:
            assert key in S, lb
            out.append('S')
        elif key not in C1:
            out.append('K_cv')
        elif qs == 1:
            out.append('K_3s')
        else:
            assert qp == 5, lb
            odd = [k for k in o if k not in ('3s', '3p') and k[-1] in 'pfh']
            assert len(odd) == 1, lb
            out.append({'p': 'K_pp', 'f': 'K_pf', 'h': 'K_ph'}[odd[0][-1]])
    return out


def lowest(op, k):
    if op.n <= 300:
        e, C = np.linalg.eigh(op.dense())
        return e[:k], C[:, :k], 0.0
    E, Cv, rn, it, nmv = davidson(op, k, tol=1e-8, maxit=600)
    return np.asarray(E), np.asarray(Cv), float(np.max(rn))


def solve(op, labels, idx, n, two):
    sub = op.sub(np.array(idx))
    lab = [labels[i] for i in idx]
    pref = {'3s': 2, '3p': 6, **({'3d': n} if n else {})}
    Pi = [k for k, lb in enumerate(lab) if occ(lb) == pref]
    k = min(4 if two else 2, sub.n)
    e, C, res = lowest(sub, k)
    wP = (C[Pi] ** 2).sum(0)
    if not two:
        i = int(np.argmax(wP > 0.5)) if (wP > 0.5).any() else 0
        return dict(E=float(e[i]), w=float(wP[i]), root=i, res=res)
    roots = sorted(np.argsort(-wP)[:2].tolist())
    X = C[np.ix_(Pi, roots)]
    u, sv, vt = np.linalg.svd(X)
    Xt = u @ vt
    return dict(Heff=Xt @ np.diag(e[roots]) @ Xt.T, w=[float(wP[r]) for r in roots], roots=roots,
                condX=float(sv[0] / sv[-1]), res=res, E2=[float(e[r]) for r in roots])


def spaces(cls):
    sets = {'a': None, 'S': {'S'}, 'C1': {'S', 'K_3s', 'K_pp', 'K_pf', 'K_ph'}, 'C2': set(['S'] + CLS)}
    for c in CLS:
        sets[f'S+{c}'] = {'S', c}
    for i, a in enumerate(CLS):
        for b in CLS[i + 1:]:
            sets[f'S+{a}+{b}'] = {'S', a, b}
    return sets


def decompose(E, U):
    """E: {(n, t): solve result}. Phase 9 quantities for one space."""
    E0 = E[(0, '1S')]['E']
    E1 = E[(1, '2D')]['E']
    E2 = {t: E[(2, t)]['E'] for t in P.T2}
    Eav2 = sum(P.g(t) * E2[t] for t in P.T2) / 45
    d2 = {t: E2[t] - Eav2 for t in P.T2}
    par = P.d2_params(E2)
    H = U.T @ E[(3, '2D')]['Heff'] @ U
    E3 = {t: E[(3, t)]['E'] for t in P.T3}
    Eav3 = (sum(P.g(t) * E3[t] for t in P.T3) + 10 * np.trace(H)) / 120
    pred = P.pairsum(d2)
    t3 = {t: E3[t] - Eav3 - pred[t] for t in P.T3}
    t3['2D'] = H - Eav3 * np.eye(2) - pred['2D']
    tv = P.tovec(t3)
    assert abs(P.meanv(tv)) < 1e-9
    G = np.array([[P.ip(P.RVEC[i], P.RVEC[j]) for j in P.RVEC] for i in P.RVEC])
    dpar = np.linalg.solve(G, np.array([P.ip(P.RVEC[i], tv) for i in P.RVEC]))
    R = tv - sum(c * P.RVEC[k] for c, k in zip(dpar, P.RVEC))
    return dict(E0=E0, E1=E1, E2=E2, E3=E3, H2D=H, Eav2=Eav2, Eav3=Eav3, par=par, dpar=dict(zip(P.RVEC, dpar)),
                t3=t3, R=R, R3=[P.ip(R, q) for q in P.Q3], rms_R=np.sqrt(P.ip(R, R) / 120),
                rms_t3=np.sqrt(P.ip(tv, tv) / 120), third=Eav3 - 3 * Eav2 + 3 * E1 - E0)


def run_name(n, t):
    return f'vc_C1_v5p_1Se' if n == 0 else f'vc_C2_{ION[n]}_{t}e'


def main():
    sets = spaces(CLS)
    E = {s: {} for s in sets}
    ver, sizes, models = [], {}, {}
    for n, t in TERMS:
        run = run_name(n, t)
        z = np.load(os.path.join(ROOT, 'results', f'{run}.npz'))
        labels = [str(s) for s in z['labels']]
        op = SymLower.from_npz(z)
        cl = classify(labels, n, t)
        two = (n == 3 and t == '2D')
        sizes[(n, t)] = {c: cl.count(c) for c in ['S'] + CLS}
        for s, members in sets.items():
            if s == 'a':
                pref = {'3s': 2, '3p': 6, **({'3d': n} if n else {})}
                idx = [i for i, lb in enumerate(labels) if occ(lb) == pref]
            else:
                idx = [i for i, c in enumerate(cl) if c in members]
            E[s][(n, t)] = solve(op, labels, idx, n, two)
        # verification of the full run against ATOMCI
        e37 = struct.unpack('<d', records(os.path.join(ROOT, 'runs', run, 'fort.37'))[1][8:16])[0]
        out = open(os.path.join(ROOT, 'runs', run, 'be.out'), errors='replace').read()
        eci = [float(x) for x in re.findall(r'STATE\s+CI ENERGY =\s+(-?\d+\.\d+)', out)]
        full = E['C2'][(n, t)]
        e0 = full['E2'][0] if two else full['E']
        ver.append(dict(run=run, N=op.n, d37=abs(e0 - e37) if (two or full['root'] == 0) else None,
                        d_root2=(abs(full['E2'][1] - eci[1]) if two else None), res=full['res']))
        models[(n, t)] = {s: E[s][(n, t)].get('w') for s in ('S', 'C1', 'C2')}
        models[(n, t)]['condX'] = {s: E[s][(n, t)].get('condX') for s in ('S', 'C1', 'C2')} if two else None
        print(f'{run}: N={op.n} classes {sizes[(n, t)]}', flush=True)
    # seniority basis at level a (CSF order may differ from Phase 9)
    Ea = E['a']
    d2a = {t: Ea[(2, t)]['E'] - sum(P.g(x) * Ea[(2, x)]['E'] for x in P.T2) / 45 for t in P.T2}
    Ha = Ea[(3, '2D')]['Heff']
    Eav3a = (sum(P.g(t) * Ea[(3, t)]['E'] for t in P.T3) + 10 * np.trace(Ha)) / 120
    U, mis = P.seniority_map(Ha, Eav3a * np.eye(2) + P.pairsum(d2a)['2D'])
    D = {s: decompose(E[s], U) for s in sets}
    # cross-checks
    p9 = json.load(open(os.path.join(ROOT, 'results', 'phase9', 'phase9_qzp.json')))['ivo']['s']
    chkS = max([abs(D['S']['E2'][t] - p9['E2'][t]) for t in P.T2] + [abs(D['S']['E3'][t] - p9['E3'][t]) for t in P.T3])
    chkC1 = []
    for n, t in TERMS:
        f = os.path.join(ROOT, 'results', f'vc_C1_{ION[n]}_{t}e.npz')
        if n in (2, 3) and t != '2D' and os.path.exists(f):
            z = np.load(f)
            o1 = SymLower.from_npz(z)
            e1 = lowest(o1, 1)[0][0]
            chkC1.append(abs(e1 - E['C1'][(n, t)]['E']))
    m = lambda x: f'{1e3 * x:+.3f}'   # noqa: E731
    w = []
    a = w.append
    a('# Phase 10: core (3s, 3p) channels of the d^3 three-body residual, V QZP, [Ne] frozen; mEh')
    a('\n## Verification')
    for v in ver:
        r2 = '' if v['d_root2'] is None else f", root 2 vs ATOMCI (6 decimals) {v['d_root2']:.1e}"
        d37 = 'n/a (target is not root 0)' if v['d37'] is None else f"{v['d37']:.1e}"
        a(f"  {v['run']}: N {v['N']}, |E - ATOMCI| {d37}{r2}, Davidson residual {v['res']:.1e}")
    a(f'  S reproduces Phase 9b level s: max |dE| {chkS:.1e} Eh; C1 as a subspace of C2 vs the C1 runs: max |dE| '
      f'{max(chkC1):.1e} Eh ({len(chkC1)} terms)')
    a(f'  2D seniority basis at level a: U = {np.round(U, 6).tolist()}, mismatch {mis:.1e} Eh')
    a('\nClass sizes (CSFs) per term: S / K_3s / K_pp / K_pf / K_ph / K_cv')
    for k, s in sizes.items():
        a(f'  d{k[0]} {k[1]}: ' + ' / '.join(str(s[c]) for c in ['S'] + CLS))
    a('\n## 1. Model space: w(3s2 3p6 3d^n) of the targets (S / C1 / C2), cond(X) for 2D; < 0.9 flagged')
    for k, mm in models.items():
        ws = [mm[s] for s in ('S', 'C1', 'C2')]
        fl = ' <-- below 0.9' if any((min(x) if isinstance(x, list) else x) < 0.9 for x in ws) else ''
        txt = ' / '.join(('/'.join(f'{y:.4f}' for y in x) if isinstance(x, list) else f'{x:.4f}') for x in ws)
        cx = '' if mm['condX'] is None else '; cond(X) ' + ' / '.join(f'{mm["condX"][s]:.4f}' for s in ('S', 'C1', 'C2'))
        a(f'  d{k[0]} {k[1]}: {txt}{cx}{fl}')
    a('\n## 2. Core correlation per term (E - E(S)), mEh')
    a('| term | C1 | C2 |')
    a('|---|---|---|')
    for n, t in TERMS:
        ks = (n, t)
        if n == 3 and t == '2D':
            for i in range(2):
                ev = {s: np.linalg.eigvalsh(D[s]['H2D'])[i] for s in ('S', 'C1', 'C2')}
                a(f'| d3 2D{"-+"[i]} | {m(ev["C1"] - ev["S"])} | {m(ev["C2"] - ev["S"])} |')
        else:
            a(f'| d{n} {t} | {m(E["C1"][ks]["E"] - E["S"][ks]["E"])} | {m(E["C2"][ks]["E"] - E["S"][ks]["E"])} |')
    a('\nParameters, refit and residual:')
    a('| space | B | C | alpha | beta | dB | dC | dalpha | dbeta | rms t3\' | rms residual | q1..q4 | 3rd diff |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s in ['a', 'S', 'C1', 'C2']:
        d = D[s]
        a(f'| {s} | ' + ' | '.join(f'{1e3 * d["par"][k]:.4f}' for k in KEYS) + ' | '
          + ' | '.join(f'{1e3 * d["dpar"][k]:+.4f}' for k in KEYS)
          + f' | {1e3 * d["rms_t3"]:.3f} | {1e3 * d["rms_R"]:.4f} | ' + ', '.join(f'{1e3 * x:+.3f}' for x in d['R3'])
          + f' | {m(d["third"])} |')
    a('\n## 3. Core channels: increments over S (one class at a time)')
    a('| class | d(dB) | d(dC) | d(dalpha) | d(dbeta) | d(3rd diff) | rms D | D in q basis |')
    a('|---|---|---|---|---|---|---|---|')
    Dv = {}
    for c in CLS:
        dd, d0 = D[f'S+{c}'], D['S']
        Dv[c] = dd['R'] - d0['R']
        a(f'| {c} | ' + ' | '.join(f'{1e3 * (dd["dpar"][k] - d0["dpar"][k]):+.4f}' for k in KEYS)
          + f' | {m(dd["third"] - d0["third"])} | {1e3 * np.sqrt(P.ip(Dv[c], Dv[c]) / 120):.4f} | '
          + ', '.join(f'{1e3 * P.ip(Dv[c], q):+.3f}' for q in P.Q3) + ' |')
    rms = lambda x: float(np.sqrt(P.ip(x, x) / 120))   # noqa: E731
    for tgt, members in (('C1', CLS[:4]), ('C2', CLS)):
        est = D['S']['R'] + sum(Dv[c] for c in members)
        a(f'Additivity, {tgt}: R(S) + sum of increments vs R({tgt}): angle {P.wangle(est, D[tgt]["R"]):.1f} deg, '
          f'norm ratio {rms(est) / rms(D[tgt]["R"]):.3f}, rms of the difference {1e3 * rms(est - D[tgt]["R"]):.4f}; '
          f'dB: {1e3 * (D["S"]["dpar"]["B"] + sum(D[f"S+{c}"]["dpar"]["B"] - D["S"]["dpar"]["B"] for c in members)):+.4f} vs '
          f'{1e3 * D[tgt]["dpar"]["B"]:+.4f}; 3rd diff: '
          f'{1e3 * (D["S"]["third"] + sum(D[f"S+{c}"]["third"] - D["S"]["third"] for c in members)):+.3f} vs '
          f'{1e3 * D[tgt]["third"]:+.3f}')
    a('Pair cross terms, rms of D(S+X+Y) - D_X - D_Y (mEh):')
    cross = {}
    for i, x in enumerate(CLS):
        for y in CLS[i + 1:]:
            dxy = D[f'S+{x}+{y}']['R'] - D['S']['R']
            cross[f'{x}+{y}'] = rms(dxy - Dv[x] - Dv[y])
    a('  ' + ', '.join(f'{k} {1e3 * v:.4f}' for k, v in cross.items()))
    c9 = json.load(open(os.path.join(ROOT, 'results', 'phase9', 'phase9c.json')))['QZP']['D']
    val = {'D_d': np.array(c9['d']), 'D_g': np.array(c9['g'])}
    a('Angles (deg) between core-channel increments and the Phase 9c valence increments:')
    a('| | ' + ' | '.join(CLS) + ' | D_d | D_g |')
    a('|---|' + '---|' * (len(CLS) + 2))
    allv = {**Dv, **val}
    for x in CLS + list(val):
        a(f'| {x} | ' + ' | '.join('' if x == y else f'{P.wangle(allv[x], allv[y]):.0f}' for y in CLS + list(val)) + ' |')
    a('\n## 4. d2 core correlation per term and class (E(S + class) - E(S)), mEh')
    a('| term | ' + ' | '.join(CLS) + ' | C1 | C2 |')
    a('|---|' + '---|' * (len(CLS) + 2))
    for t in P.T2:
        ks = (2, t)
        eS = E['S'][ks]['E']
        a(f'| {t} | ' + ' | '.join(m(E[f"S+{c}"][ks]["E"] - eS) for c in CLS)
          + f' | {m(E["C1"][ks]["E"] - eS)} | {m(E["C2"][ks]["E"] - eS)} |')
    os.makedirs(os.path.join(ROOT, 'results', 'phase10'), exist_ok=True)
    open(os.path.join(ROOT, 'results', 'phase10', 'phase10.txt'), 'w').write('\n'.join(w) + '\n')

    def js(x):
        if isinstance(x, dict):
            return {str(k): js(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [js(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x
    json.dump(js(dict(decomposition=D, increments=Dv, cross=cross, sizes=sizes, models=models, verification=ver)),
              open(os.path.join(ROOT, 'results', 'phase10', 'phase10.json'), 'w'), indent=1)
    print('\n'.join(w))


if __name__ == '__main__':
    main()
