#!/usr/bin/env python3
"""Phase 12a: the three-body vectors in the Judd-Leavitt operator basis (J. Phys. B 19, 485 (1986), Table 6).

Components 4F 4P 2H 2G 2F 2P; 2D 11, 33, 13 (seniority basis of cfp_l; 13 = <v1|H|v3>); metric phase9.ip.
Checks: orthogonality and norms of e0..e4, t1..t4; span{e} = span{1, r_B, r_C, r_alpha, r_beta}; span{t} = span{q}
(and the same with the 2D 13 sign of e4/t1 flipped, to fix the phase convention).
Basis for the analysis (orthogonal): u = 3 t1 + 2 t2, t4, w = t1 - t2, t3. span{u, t4} = "second-order" part.
Vectors: the seven second-order angular vectors (11b, 11d), and every CI three-body vector stored in results/
(Phase 9, 9b, 9c, 10b/10d, 11a, 11c-11e). Series of 11d/11e (d path) are recomputed from the exports (seconds).

Usage: phase12a.py   -> results/phase12/phase12a.txt/.json
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase9 as P  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase12')
R189 = np.sqrt(189)
JL = {
    'e0': [1, 1, 1, 1, 1, 1, 1, 1, 0],
    'e1': [3, 3, 3, 3, 3, 3, -33, 3, 0],
    'e2': [21, 21, -12, -12, -12, -12, 0, -12, 0],
    'e3': [-3, 7, -12, -2, 6, 16, 0, 12, 0],
    'e4': [-3, 7, 3, -7, 9, -11, 0, 0, R189],
    't1': [-3, 7, 3, -7, 9, -11, 0, 0, -R189],
    't2': [-9, 21, 3, 8, -12, 3, 0, -6, 0],
    't3': [0, 0, 6, -9, -9, 6, 0, 12, 0],
    't4': [0, 0, -30, 55, -33, -198, 0, 132, 0],
}
JL = {k: np.array(v, float) for k, v in JL.items()}
ip = P.ip


def span_res(A, B):
    """max over b in B of the relative norm of b outside span(A) (metric ip)."""
    G = np.array([[ip(a, c) for c in A] for a in A])
    out = 0.0
    for b in B:
        c = np.linalg.solve(G, [ip(a, b) for a in A])
        r = b - sum(x * a for x, a in zip(c, A))
        out = max(out, np.sqrt(ip(r, r) / ip(b, b)))
    return out


def frac(x, dmax=10 ** 7):
    f = Fraction(x).limit_denominator(dmax)
    return f if abs(float(f) - x) < 1e-12 * max(1, abs(x)) else None


def main():
    os.makedirs(OUT, exist_ok=True)
    w = []
    a = w.append
    a('# Phase 12a: three-body vectors in the Judd-Leavitt basis (J. Phys. B 19, 485 (1986), Table 6)')
    a('Components 4F 4P 2H 2G 2F 2P; 2D 11, 33, 13 (cfp_l seniority basis, 13 = <v1|H|v3>); metric weights '
      + ', '.join(f'{x:g}' for x in P.GV) + '.')
    names = list(JL)
    G = np.array([[ip(JL[i], JL[j]) for j in names] for i in names])
    off = np.abs(G - np.diag(np.diag(G))).max()
    a('\n## Checks')
    a(f'Gram matrix of e0..e4, t1..t4: max |off-diagonal| {off:.1e}; norms ' + ', '.join(
        f'{k} {G[i, i]:.6g}' for i, k in enumerate(names)))
    E = [JL[k] for k in ('e0', 'e1', 'e2', 'e3', 'e4')]
    TT = [JL[k] for k in ('t1', 't2', 't3', 't4')]
    ours = [P.ONE] + [P.RVEC[k] for k in ('B', 'C', 'alpha', 'beta')]
    s1, s2 = span_res(E, ours), span_res(ours, E)
    s3, s4 = span_res(TT, P.Q3), span_res(P.Q3, TT)
    a(f'span{{e0..e4}} vs span{{1, r_B, r_C, r_alpha, r_beta}}: ours outside e {s1:.1e}, e outside ours {s2:.1e}')
    a(f'span{{t1..t4}} vs span{{q1..q4}}: q outside t {s3:.1e}, t outside q {s4:.1e}')
    fl = {k: v.copy() for k, v in JL.items()}
    fl['e4'][8] *= -1
    fl['t1'][8] *= -1
    f1 = span_res([fl[k] for k in ('e0', 'e1', 'e2', 'e3', 'e4')], ours)
    f3 = span_res([fl[k] for k in ('t1', 't2', 't3', 't4')], P.Q3)
    a(f'With the 2D 13 sign of e4 and t1 flipped: ours outside e {f1:.2f}, q outside t {f3:.2f} -> the conventions agree '
      f'only with the sign as given (our r_B has 2D 13 = {P.RVEC["B"][8]:+.6f} = +3 sqrt(21) = {3 * np.sqrt(21):.6f}).')
    chk = dict(offdiag=off, norms={k: G[i, i] for i, k in enumerate(names)}, span_e=[s1, s2], span_t=[s3, s4],
               flipped=[f1, f3])

    u = 3 * JL['t1'] + 2 * JL['t2']
    wv = JL['t1'] - JL['t2']
    BAS = {'u': u, 't4': JL['t4'], 'w': wv, 't3': JL['t3']}
    a(f'u = 3 t1 + 2 t2, w = t1 - t2: <u,w> = {ip(u, wv):.1e}; norms <u,u> {ip(u, u):.6g}, <w,w> {ip(wv, wv):.6g}')

    def dec(x):
        c = {k: ip(x, b) / ip(b, b) for k, b in BAS.items()}
        n2 = {k: c[k] ** 2 * ip(b, b) for k, b in BAS.items()}
        tot = ip(x, x)
        res = x - sum(c[k] * b for k, b in BAS.items())
        return c, n2, tot, float(np.sqrt(ip(res, res) / tot)) if tot > 0 else 0.0

    # 1. second-order angular vectors
    j11b = json.load(open(os.path.join(ROOT, 'results', 'phase11', 'phase11b.json')))
    j11d = json.load(open(os.path.join(ROOT, 'results', 'phase11', 'phase11d.json')))
    ang = {'3s->3d R2xR2': np.array(list(j11b['pairs'].values())[0]['unit_R'])}
    for X in ('d', 'g'):
        for nm, v in j11d['angular'][X].items():
            ang[f'{X} {nm}'] = np.array(v['values'])
    a('\n## 1. Second-order angular vectors (unit integrals, Delta = 1 Eh): coefficients on t1..t4')
    a('| vector | t1 | t2 | t3 | t4 | t1:t2 | coefficient on u | on t4 | residual outside {t} |')
    a('|---|---|---|---|---|---|---|---|---|')
    js_ang = {}
    for nm, x in ang.items():
        ct = [ip(x, t) / ip(t, t) for t in TT]
        fr = [frac(c) for c in ct]
        res = x - sum(c * t for c, t in zip(ct, TT))
        ratio = ct[0] / ct[1] if abs(ct[1]) > 1e-15 else float('nan')
        cu, c4 = ip(x, u) / ip(u, u), ip(x, JL['t4']) / ip(JL['t4'], JL['t4'])
        a(f'| {nm} | ' + ' | '.join(str(f) if f is not None else f'{c:.3e}' for f, c in zip(fr, ct))
          + f' | {ratio:.12f} | {frac(cu) or cu} | {frac(c4) or c4} | {np.sqrt(ip(res, res) / ip(x, x)):.1e} |')
        js_ang[nm] = dict(t=ct, t_exact=[str(f) for f in fr], u=cu, t4=c4)
    a('max |t3| over the 7 vectors: %.1e; max |t1/t2 - 3/2|: %.1e' % (
        max(abs(v['t'][2]) for v in js_ang.values()),
        max(abs(v['t'][0] / v['t'][1] - 1.5) for v in js_ang.values())))

    # 2. CI vectors
    def L(*p):
        return json.load(open(os.path.join(ROOT, 'results', *p)))
    V = {}
    for tag, f in (('TZP', ('phase9', 'phase9b.json')), ('QZP', ('phase9', 'phase9b.json'))):
        r = L(*f)['residual'][tag]
        for lv in ('b', 's', 'd', 'c', 'c_nog'):
            V[f'9/9b {tag} R({lv})'] = r[lv]['R']
    for tag in ('TZP', 'QZP'):
        c = L('phase9', 'phase9c.json')[tag]
        for X in ('s', 'd', 'g'):
            V[f'9c {tag} D_{X}'] = c['D'][X]
        V[f'9c {tag} R(c)-R(c_nog)'] = c['Dfull']
    p2u = L('phase10', 'phase10b_part2u.json')['decomposition']
    for s in ('S', 'C1', 'C2', 'C3a'):
        V[f'10b R({s}) (core-open, [Ne])'] = p2u[s]['R']
    d10 = L('phase10', 'phase10d.json')['R']
    V['10d R(C3k) (total, [Ne])'] = d10['C3k']
    V['10d R(C3k) - R(S) (core)'] = list(np.array(d10['C3k']) - np.array(d10['S']))
    V['10c u = R(T0) - R(S)'] = L('phase10', 'phase10c_r.json')['u']
    a11 = L('phase11', 'phase11a.json')['R']
    for k in ('B_sep', 'E'):
        V[f'11a 3s->3d {k}'] = a11[k]
    c11 = L('phase11', 'phase11c.json')['paths']
    e11 = L('phase11', 'phase11e.json')['paths']
    for X in ('d', 'g'):
        V[f'11c {X} B2'] = c11[X]['R']['B2']
        V[f'11d {X} E'] = j11d['paths'][X]['R']['full']
        V[f'11d {X} E_blk'] = j11d['paths'][X]['R']['blk']
        V[f'11d {X} E_diag'] = j11d['paths'][X]['R']['diag']
        V[f'11d {X} E_noPrel'] = j11d['paths'][X]['R']['noPrel']
        V[f'11e {X} E_blk can'] = e11[X]['R']['E_blk can']
        for v in ('B2 can', 'B3 orig', 'B3 can', 'B3+ orig', 'B4 orig'):
            V[f'11e {X} {v}'] = e11[X]['R'][v]
        V[f'11e {X} B4 can'] = e11[X]['R']['B4 can']
    V['11d d+g E'] = j11d['both']['R_dg']
    a('\n## 2. CI three-body vectors in (u, t4, w, t3)')
    a('Coefficients x_b = <R,b>/<b,b> (microEh per unit of b); fractions of |R|^2; "beyond" = (w, t3) fraction '
      '= outside span{u, t4}.')
    a('| vector | rms (mEh) | x_u | x_t4 | x_w | x_t3 | f_u | f_t4 | f_w | f_t3 | beyond |')
    a('|---|---|---|---|---|---|---|---|---|---|---|')
    js_v = {}
    for nm, x in V.items():
        x = np.asarray(x, float)
        c, n2, tot, res = dec(x)
        assert res < 1e-6, (nm, res)
        f = {k: n2[k] / tot for k in n2}
        js_v[nm] = dict(coef=c, frac=f, rms=float(np.sqrt(tot / 120)))
        a(f'| {nm} | {1e3 * np.sqrt(tot / 120):.4f} | ' + ' | '.join(f'{1e6 * c[k]:+.4f}' for k in BAS) + ' | '
          + ' | '.join(f'{f[k]:.3f}' for k in BAS) + f' | {f["w"] + f["t3"]:.3f} |')

    # 3. series, d path (and g), original and canonical orbitals
    from phase11c import TERMS, decomp, pack, term_data
    from phase11d import series
    from scipy.linalg import block_diag
    virt = {X: sorted(L('phase11', 'phase11c.json')['paths'][X]['labels'], key=lambda o: int(o[:-1])) for X in ('d', 'g')}
    T = {k: term_data(*k, virt) for k in TERMS}

    def blk(d, r, cc):
        return d['H'][np.ix_([d['pos'][i] for i in r], [d['pos'][i] for i in cc])]
    EA = {k: pack(blk(d, d['Pi'], d['Pi']), k == (3, '2D')) for k, d in T.items()}
    d2a = {t: EA[(2, t)]['E'] - sum(P.g(y) * EA[(2, y)]['E'] for y in P.T2) / 45 for t in P.T2}
    Ha = EA[(3, '2D')]['Heff']
    Eav3a = (sum(P.g(t) * EA[(3, t)]['E'] for t in P.T3) + 10 * np.trace(Ha)) / 120
    U, _ = P.seniority_map(Ha, Eav3a * np.eye(2) + P.pairsum(d2a)['2D'])
    RA, _ = decomp(EA, U)
    a('\n## 3. Series (orders 2..6), coefficients on (u, t4, w, t3) in microEh and the beyond fraction')
    js_s = {}
    for X in ('d', 'g'):
        orbs = virt[X]
        no = len(orbs)
        C = np.array(e11[X]['C'])
        for bn, Cb in (('orig', np.eye(no)), ('can', C)):
            for sn in ('EN', 'BLK'):
                S = {}
                for k, d in T.items():
                    Qa = [i for o in orbs for i in d['Q'][o]]
                    H0 = blk(d, d['Pi'] + Qa, d['Pi'] + Qa)
                    p, m = len(d['Pi']), len(d['Q'][orbs[0]])
                    Tm = block_diag(np.eye(p), np.kron(Cb, np.eye(m)))
                    Hs = Tm.T @ H0 @ Tm
                    gr = [[q] for q in range(no * m)] if sn == 'EN' else [list(range(i * m, (i + 1) * m)) for i in range(no)]
                    S[k] = series(Hs, p, gr, nmax=6)
                a(f'\n{X} path, {bn} orbitals, {sn}:')
                a('| order | x_u | x_t4 | x_w | x_t3 | beyond | coefficient on delta(E) |')
                a('|---|---|---|---|---|---|---|')
                dE = np.asarray(j11d['paths'][X]['R']['full'])
                js_s[f'{X} {bn} {sn}'] = {}
                for o in range(2, 7):
                    r = decomp({k: pack(S[k][o], k == (3, '2D')) for k in T}, U)[0] - RA
                    c, n2, tot, _ = dec(r)
                    js_s[f'{X} {bn} {sn}'][o] = dict(coef=c, beyond=(n2['w'] + n2['t3']) / tot)
                    a(f'| {o} | ' + ' | '.join(f'{1e6 * c[k]:+.4f}' for k in BAS)
                      + f' | {(n2["w"] + n2["t3"]) / tot:.4f} | {ip(r, dE) / ip(dE, dE):.3f} |')
                c, n2, tot, _ = dec(dE)
                a('| E | ' + ' | '.join(f'{1e6 * c[k]:+.4f}' for k in BAS) + f' | {(n2["w"] + n2["t3"]) / tot:.4f} | 1 |')
    open(os.path.join(OUT, 'phase12a.txt'), 'w').write('\n'.join(w) + '\n')
    json.dump(dict(checks=chk, angular=js_ang, vectors=js_v, series=js_s,
                   basis={k: v.tolist() for k, v in BAS.items()}), open(os.path.join(OUT, 'phase12a.json'), 'w'),
              indent=1, default=float)
    print('\n'.join(w))


if __name__ == '__main__':
    main()
