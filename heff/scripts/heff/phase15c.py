#!/usr/bin/env python3
"""Phase 15c: Uylings' (1984) three-body operators O2', O2'' built algebraically; projection of the measured t.

Step 1 (algebra, no CI data; second quantization on the d shell plus one extra shell X, spin orbitals of X after
the d ones, so a P determinant d^N (+ closed X) maps 1:1 with sign +1 onto the dshell determinant):
  (a) H_SL = lift of the d^2 term projector |SL><SL| to d^N; H^c_SL = H_SL - C(N,2) [S,L]/45 (traceless, the
      orthogonalization to the identity under Tr); Gram matrix under the Uylings inner product (= dshell.ip, the
      trace over all states, weight [S,L] per term).
  (b) O2': t-part (projection on t1..t4 in d^3) of the closure -V+V of one excitation X -> d (X = inner closed s)
      with rank k = 2; lower-body lifts lie in span{e0..e4} in d^3, so the t-part of the d^3 closure matrix is its
      orthogonalized pure three-body part. Check: d -> s' (empty s') gives the same direction.
  (c) span of the t-parts of -V+V for d -> d' (k = 0,2,4), d -> g' (k = 2,4,6), d -> s' (k = 2), inner s -> d,
      with arbitrary radial weights (all symmetrized products O_k+ O_k'); its dimension; O2'' = the direction in it
      orthogonal to O2'. The one-body d -> d' term is tried separately.
  (d) span{O2', O2''} vs span{u = 3t1 + 2t2, t4} (Phase 12); orthogonality of e/t/f under the Uylings product.
Step 2: the measured t (own orbitals, d3 and d4, Ti..Fe, 5 spaces) as T1 O2' + T2 O2'' + rest; O2', O2'' unit-norm
  pure three-body operators under the trace product in d^3; signs: O2' along the s -> d closure -V+V (positive
  denominator), O2'' along the d -> d' (k = 2) closure component orthogonal to O2'.
Step 3: two-body side: the e-part of the sector H_eff as a pure two-body operator W = sum_k c_k e_k; its d^2 term
  energies E_SL and E_SL - CG (CG = sum [S,L] E_SL / 45), for e^(1) (H_PP) and De, V and Cr, N = 2, 3, 4.
-> results/phase15/phase15c.txt/.json
"""
import json
import os
import sys
import time
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import cfp_l  # noqa: E402
from atomci_heff import dshell as S  # noqa: E402
from atomci_heff.cfp import ck  # noqa: E402
import phase13b as B  # noqa: E402
import phase14b as P14  # noqa: E402
import phase15b as Q15  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase15')
SOd = cfp_l.spin_orbitals(2)
T3 = S.jl_table(3)


def ops_for(lx, inner, k):
    """V_QP of the two-body interaction with one electron X <-> d (radial weight 1 for rank k), as a dict of
    (p, q, r, s) -> <pq|rs> over spin orbitals (d: 0..9, X: 10..)."""
    SOx = cfp_l.spin_orbitals(lx)
    orb = [(2, m, s) for m, s in SOd] + [(lx, m, s) for m, s in SOx]
    isX = [False] * 10 + [True] * len(SOx)
    g = {}
    for p, (lp, mp, sp) in enumerate(orb):
        for q, (lq, mq, sq) in enumerate(orb):
            for r, (lr, mr, sr) in enumerate(orb):
                for s_, (ls, ms, ss) in enumerate(orb):
                    nXc, nXa = isX[p] + isX[q], isX[r] + isX[s_]
                    if inner:
                        if not (nXc == 0 and nXa == 1):
                            continue
                    elif not (nXc == 1 and nXa == 0):
                        continue
                    if sp != sr or sq != ss or mp + mq != mr + ms:
                        continue
                    v = ck(k, lp, mp, lr, mr) * ck(k, ls, ms, lq, mq)
                    if abs(v) > 1e-14:
                        g[(p, q, r, s_)] = v
    return g, len(SOx)


def closure_t(N, lx, inner, ks, onebody=False):
    """{(k, k'): t-vector (d^3 components restricted to t1..t4 coefficients)} of -sym(O_k+ O_k') for d^N."""
    nx = 2 * (2 * lx + 1)
    dN = S.dets(N)
    if inner:
        Pd = [d + tuple(range(10, 10 + nx)) for d in dN]
    else:
        Pd = list(dN)
    Ops = {}
    for k in ks:
        g, _ = ops_for(lx, inner, k)
        cols = {}
        for j, d in enumerate(Pd):
            out = {}
            for (p, q, r, s_), v in g.items():
                res = cfp_l._apply([('c', p), ('c', q), ('a', s_), ('a', r)], d)
                if res:
                    out[res[1]] = out.get(res[1], 0.0) + 0.5 * v * res[0]
            cols[j] = out
        Ops[k] = cols
    if onebody and not inner:
        # one-body d -> X (same m, s): sum_m a+_X(m s) a_d(m s)
        cols = {}
        for j, d in enumerate(Pd):
            out = {}
            for i, (m, s) in enumerate(SOd):
                p = 10 + cfp_l.spin_orbitals(lx).index((m, s)) if lx == 2 else None
                if p is None:
                    continue
                res = cfp_l._apply([('c', p), ('a', i)], d)
                if res:
                    out[res[1]] = out.get(res[1], 0.0) + res[0]
            cols[j] = out
        Ops['h'] = cols
    qidx = {}
    for cols in Ops.values():
        for out in cols.values():
            for qd in out:
                qidx.setdefault(qd, len(qidx))
    M = {}
    for k, cols in Ops.items():
        A = np.zeros((len(qidx), len(Pd)))
        for j, out in cols.items():
            for qd, v in out.items():
                A[qidx[qd], j] = v
        M[k] = A
    res = {}
    keys = list(Ops)
    for i, a in enumerate(keys):
        for b in keys[i:]:
            X = -(M[a].T @ M[b] + M[b].T @ M[a]) / (1 if a != b else 2)
            res[(a, b)] = X
    return res


def tvec(N, X):
    """t-part of a d^N matrix as coefficients on the Fock t1..t4 (Gram solve on the 14b basis), mapped to d^3."""
    names, vecs = P14.basis_ops(N)
    x = S.to_vec(N, X)
    G = np.array([[S.ip(N, p, q) for q in vecs] for p in vecs])
    c = dict(zip(names, np.linalg.solve(G, [S.ip(N, p, x) for p in vecs])))
    return np.array([c[k] for k in B.TK]), {k: v for k, v in c.items() if k.startswith('f')}


def to_d3(cvec):
    return sum(c * T3[k] for c, k in zip(cvec, B.TK))


def nrm3(v):
    return float(np.sqrt(max(S.ip(3, v, v), 0.0)))


def step1(w, js):
    a = w.append
    # (a)
    a('\n## Step 1(a). H_SL and H^c_SL (traceless) under the Uylings / JHR trace inner product')
    t2 = S.terms(2)
    SL = [(L, twoS) for L, twoS, vs in t2]
    labs = [f'{twoS + 1}{S.LCH[L]}' for L, twoS in SL]
    for N in (2, 3, 4, 5):
        H, Hc = [], []
        for i, (L, twoS) in enumerate(SL):
            v = np.zeros(len(S.components(2)))
            v[i] = 1.0
            HN = S.to_vec(N, S.lift(S.from_vec(2, v), 2, N))
            g = (2 * L + 1) * (twoS + 1)
            I = S.to_vec(N, np.eye(len(S.dets(N))))
            Hc.append(HN - comb(N, 2) * g / 45 * I)
            H.append(HN)
        tr = [S.ip(N, h, S.to_vec(N, np.eye(len(S.dets(N))))) for h in Hc]
        G = np.array([[S.ip(N, p, q) for q in Hc] for p in Hc])
        ev = np.linalg.eigvalsh(G)
        a(f'- d{N}: Tr H^c_SL max {max(abs(x) for x in tr):.1e}; Gram of H^c_SL ({", ".join(labs)}): rank '
          f'{int((ev > 1e-9 * ev.max()).sum())}; diagonal ' + ', '.join(f'{G[i, i]:.4f}' for i in range(5)) +
          '; off-diagonal ' + ' '.join(f'{G[i, j]:+.4f}' for i in range(5) for j in range(i + 1, 5)))
        js.setdefault('Hc_gram', {})[N] = G.tolist()
    a('- Uylings Table 1 (alpha(N), beta(N), x(N)) and Table 2 (u_SL) not checked: the paper is not on this '
      'machine. The Gram matrices above are what those tables tabulate (normalization and overlaps of H^c_SL).')
    # (b)
    a('\n## Step 1(b). O2\' from the s -> d closure (k = 2), and the d -> s\' check')
    o_in = {}
    for N in (2, 3, 4):
        X = closure_t(N, 0, True, [2])[(2, 2)]
        o_in[N] = X
    tin, fin = tvec(3, o_in[3])
    Xout = closure_t(3, 0, False, [2])[(2, 2)]
    tout, _ = tvec(3, Xout)
    O1 = to_d3(tin)
    O1 /= nrm3(O1)
    vo = to_d3(tout)
    cosang = S.ip(3, O1, vo) / nrm3(vo)
    a(f'- s -> d closure, d3: t-part coefficients (t1..t4) ' + ' '.join(f'{x:+.6f}' for x in tin) +
      f'; d -> s\' closure: ' + ' '.join(f'{x:+.6f}' for x in tout) +
      f'; angle between the two t-parts {np.degrees(np.arccos(min(1, abs(cosang)))):.2e} deg '
      f'(sign {"same" if cosang > 0 else "opposite"})')
    # same direction from d4 (t-part via lifted Fock t)
    t4in, f4 = tvec(4, closure_t(4, 0, True, [2])[(2, 2)])
    v4 = to_d3(t4in)
    a(f'- consistency: the s -> d closure in d4 has t-coefficients along the same direction '
      f'(angle {np.degrees(np.arccos(min(1, abs(S.ip(3, O1, v4)) / nrm3(v4)))):.2e} deg) and f-part norm '
      f'{np.sqrt(sum(x ** 2 for x in f4.values())):.2e} (coefficient units)')
    # (c)
    a('\n## Step 1(c). Span of second-order three-body parts and O2\'\'')
    span = {}
    for name, lx, inner, ks in (('s -> d (inner)', 0, True, [2]), ('d -> s\'', 0, False, [2]),
                                ('d -> d\'', 2, False, [0, 2, 4]), ('d -> g\'', 4, False, [2, 4, 6])):
        res = closure_t(3, lx, inner, ks)
        vs = [to_d3(tvec(3, X)[0]) for X in res.values()]
        nonzero = [(kk, nrm3(v)) for kk, v in zip(res, vs)]
        span[name] = vs
        M = np.array(vs)
        Gm = np.array([[S.ip(3, p, q) for q in vs] for p in vs])
        ev = np.linalg.eigvalsh(Gm)
        a(f'- {name}: products {", ".join(f"{kk}:{n:.3e}" for kk, n in nonzero)}; dimension of the t-span '
          f'{int((ev > 1e-10 * max(ev.max(), 1e-300)).sum())}')
    allv = [v for vs in span.values() for v in vs]
    Gm = np.array([[S.ip(3, p, q) for q in allv] for p in allv])
    ev, U = np.linalg.eigh(Gm)
    dim = int((ev > 1e-10 * ev.max()).sum())
    a(f'- all excitations together: dimension {dim} (eigenvalues of the Gram matrix '
      + ', '.join(f'{x:.2e}' for x in sorted(ev)[::-1][:5]) + ')')
    res1 = closure_t(3, 2, False, [0, 2, 4], onebody=True)
    vs1 = [to_d3(tvec(3, X)[0]) for X in res1.values()]
    Gm1 = np.array([[S.ip(3, p, q) for q in vs1] for p in vs1])
    ev1 = np.linalg.eigvalsh(Gm1)
    a(f'- d -> d\' including the one-body d -> d\' term and its cross products: dimension '
      f'{int((ev1 > 1e-10 * ev1.max()).sum())}')
    # O2'' : component of d->d' (k=2) orthogonal to O2'
    d2 = span['d -> d\''][[k for k in closure_t(3, 2, False, [0, 2, 4])].index((2, 2))]
    O2 = d2 - S.ip(3, d2, O1) * O1
    O2 /= nrm3(O2)
    # check that every vector lies in span{O1, O2}
    resid = max(nrm3(v - S.ip(3, v, O1) * O1 - S.ip(3, v, O2) * O2) / max(nrm3(v), 1e-300) for v in allv
                if nrm3(v) > 1e-12)
    a(f'- O2\'\' = d -> d\' (k = 2) closure t-part orthogonalized to O2\'. Max relative residual of all closure '
      f't-parts outside span{{O2\', O2\'\'}}: {resid:.1e}')
    c1 = [S.ip(3, O1, T3[k]) / S.ip(3, T3[k], T3[k]) for k in B.TK]
    c2 = [S.ip(3, O2, T3[k]) / S.ip(3, T3[k], T3[k]) for k in B.TK]
    a('- O2\' = ' + ' '.join(f'{x:+.6f} {k}' for x, k in zip(c1, B.TK)) + ' (unit trace norm in d3)')
    a('- O2\'\' = ' + ' '.join(f'{x:+.6f} {k}' for x, k in zip(c2, B.TK)))
    # (d)
    a('\n## Step 1(d). Relation to t1..t4 and span{u, t4}; orthogonality of e/t/f')
    u = 3 * T3['t1'] + 2 * T3['t2']
    r_ut4 = []
    for v, nm in ((u, 'u'), (T3['t4'], 't4')):
        r = v - S.ip(3, v, O1) * O1 - S.ip(3, v, O2) * O2
        r_ut4.append(nrm3(r) / nrm3(v))
        a(f'- {nm} outside span{{O2\', O2\'\'}}: relative residual {nrm3(r) / nrm3(v):.1e}')
    Q2 = [u / nrm3(u)]
    t4p = T3['t4'] - S.ip(3, T3['t4'], Q2[0]) * Q2[0]
    Q2.append(t4p / nrm3(t4p))
    for v, nm in ((O1, "O2'"), (O2, "O2''")):
        r = v - sum(S.ip(3, v, q) * q for q in Q2)
        a(f'- {nm} outside span{{u, t4}}: relative residual {nrm3(r):.1e}')
    a(f'- O2\' = {S.ip(3, O1, u) / S.ip(3, u, u):+.6f} u {S.ip(3, O1, T3["t4"]) / S.ip(3, T3["t4"], T3["t4"]):+.6f} t4 '
      f'(u, t4 orthogonal: <u,t4> = {S.ip(3, u, T3["t4"]):.1e}); O2\'\' = {S.ip(3, O2, u) / S.ip(3, u, u):+.6f} u '
      f'{S.ip(3, O2, T3["t4"]) / S.ip(3, T3["t4"], T3["t4"]):+.6f} t4')
    for N in (3, 4, 5):
        names, vecs = P14.basis_ops(N)
        Gm = np.array([[S.ip(N, p, q) for q in vecs] for p in vecs])
        off = np.abs(Gm - np.diag(np.diag(Gm))).max() / np.abs(np.diag(Gm)).max()
        grp = lambda n: 'e' if n.startswith('e') else n[0]  # noqa: E731
        cross = max((abs(Gm[i, j]) for i in range(len(names)) for j in range(len(names))
                     if grp(names[i]) != grp(names[j])), default=0.0) / np.abs(np.diag(Gm)).max()
        a(f'- d{N}: max relative off-diagonal of the Gram matrix of {{{", ".join(names)}}} under the trace product '
          f'{off:.1e}; between groups e / t / f {cross:.1e}')
    js['O1'], js['O2'] = c1, c2
    js['dim'] = dim
    js['r_ut4'] = r_ut4
    return O1, O2


def step2(w, js, O1, O2):
    a = w.append
    a('\n## Step 2. Measured t = T1 O2\' + T2 O2\'\' + rest (mEh; d3 trace-normalized operators); angle = '
      'atan2(T2, T1); out = |rest| / |t|')
    a('| series | space | ion | abs(t) | T1 | T2 | angle (deg) | out of plane | T2/T1 |')
    a('|---|---|---|---|---|---|---|---|---|')
    s2 = {}
    for N in (3, 4):
        for sp in Q15.SPACES:
            for el in Q15.EL:
                t = Q15.tag_of(el, N)
                x, dg = P14.sector(sp, N, t)
                c = P14.decompose(N, x)['c']
                v = to_d3([c[k] for k in B.TK])
                T1, T2 = S.ip(3, v, O1), S.ip(3, v, O2)
                rest = v - T1 * O1 - T2 * O2
                nt = nrm3(v)
                ang = float(np.degrees(np.arctan2(T2, T1)))
                s2[f'{N} {sp} {el}'] = dict(t=nt, T1=T1, T2=T2, ang=ang, out=nrm3(rest) / nt, flag=dg['flag'])
                a(f'| d{N} | {sp} | {el}{" F" if dg["flag"] else ""} | {1e3 * nt:.4f} | {1e3 * T1:+.4f} | '
                  f'{1e3 * T2:+.4f} | {ang:+.2f} | {nrm3(rest) / nt:.4f} | {T2 / T1:+.3f} |')
    js['step2'] = s2
    a('\nAngle drift along each series (well-posed ions): max - min of the angle (deg); mean out-of-plane fraction')
    a('| series | space | drift (deg) | out of plane (mean, max) |')
    a('|---|---|---|---|')
    for N in (3, 4):
        for sp in Q15.SPACES:
            r = [s2[f'{N} {sp} {el}'] for el in Q15.EL if not s2[f'{N} {sp} {el}']['flag']]
            angs = [x['ang'] for x in r]
            outs = [x['out'] for x in r]
            a(f'| d{N} | {sp} | {max(angs) - min(angs):.2f} | {np.mean(outs):.4f}, {max(outs):.4f} |')


def step3(w, js):
    a = w.append
    a('\n## Step 3. Two-body side: e-part as d^2 term energies E_SL (pure two-body W = sum_k c_k e_k), '
      'E_SL - CG (mEh); e^(1) = H_PP (HF), De = e - e^(1)')
    E2 = B.e_twobody()[0]
    t2 = S.terms(2)
    labs = [f'{twoS + 1}{S.LCH[L]}' for L, twoS, vs in t2]
    g = np.array([(2 * L + 1) * (twoS + 1) for L, twoS, vs in t2], float)
    a('| ion | N | part | space | ' + ' | '.join(f'E-CG {l}' for l in labs) + ' | CG |')
    a('|---|---|---|---|' + '---|' * (len(labs) + 1))
    s3 = {}
    for el in ('V', 'Cr'):
        for N in (2, 3, 4):
            t = Q15.tag_of(el, N) if N > 2 else {'V': 'o14n2', 'Cr': 'cr15n2'}[el]
            for sp in ('3d', 's6d', 'core', 'g'):
                x, dg = P14.sector(sp, N, t)
                ce = P14.decompose(N, x)['c']
                by = {}
                for b in P14.sector_blocks(sp, N, t):
                    by[(b['L'], b['twoS'])] = b['U'].T @ b['H'][np.ix_(b['P'], b['P'])] @ b['U']
                c1 = P14.decompose(N, B.vec_from_blocks(N, by))['c']
                for part, cc in (('e(1)', c1), ('De', {k: ce[k] - c1[k] for k in B.EK})):
                    if part == 'e(1)' and sp != '3d':
                        continue
                    W = sum(cc[k] * E2[k][1] for k in B.EK[1:])
                    E = S.to_vec(2, W)
                    CG = float(np.sum(g * E) / g.sum())
                    s3[f'{el} {N} {part} {sp}'] = dict(E=E.tolist(), CG=CG)
                    a(f'| {el} | {N} | {part} | {sp if part == "De" else "all"} | ' +
                      ' | '.join(f'{1e3 * (e - CG):+.3f}' for e in E) + f' | {1e3 * CG:+.3f} |')
    js['step3'] = s3
    # HF check: E_SL - CG of e(1) in d2 vs the Condon-Shortley d2 term energies with the own F2, F4
    orbj = json.load(open(os.path.join(OUT, 'phase15a_orb.json')))
    a('\nHF check: d^2 term energies relative to CG from the own F^2, F^4 (Slater-Condon, F2 = F^2/49, F4 = F^4/441) '
      'vs e(1) above:')
    cs = {'1S': (14, 126), '3P': (7, -84), '1D': (-3, 36), '3F': (-8, -9), '1G': (4, 1)}
    for el in ('V', 'Cr'):
        for N in (2, 3, 4):
            F2, F4 = orbj[el][str(N)]['F2'] / 49, orbj[el][str(N)]['F4'] / 441
            Es = np.array([cs[l][0] * F2 + cs[l][1] * F4 for l in labs])
            cg = float(np.sum(g * Es) / g.sum())
            e1 = np.array(s3[f'{el} {N} e(1) 3d']['E']) - s3[f'{el} {N} e(1) 3d']['CG']
            a(f'- {el} d{N}: Slater-Condon ' + ', '.join(f'{l} {1e3 * (e - cg):+.3f}' for l, e in zip(labs, Es)) +
              f'; max |difference| from e(1) {1e3 * np.abs(Es - cg - e1).max():.2e} mEh')


def main():
    t0 = time.time()
    js = {}
    w = ['# Phase 15c: Uylings three-body operators O2\', O2\'\' and the measured t',
         open(__file__).read().split('"""')[1].split('-> results')[0].strip()]
    O1, O2 = step1(w, js)
    print(f'step1 {time.time() - t0:.0f} s', flush=True)
    step2(w, js, O1, O2)
    print(f'step2 {time.time() - t0:.0f} s', flush=True)
    step3(w, js)
    w.append(f'\nWall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase15c.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    open(os.path.join(OUT, 'phase15c.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
