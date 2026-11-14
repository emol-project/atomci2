#!/usr/bin/env python3
"""Phase 13h addendum: second order in T (V T T V) for b' (core, g as controls).

Same model as 13h (H0 + H_PQ + lambda sum_c T_c; configuration-average denominators, rho = 0). The second-order-in-V
Hermitian effective Hamiltonian, exact in T, is H2 = 1/2 (Y V+ + V Y+) with Y solving the Sylvester equation
  (E_P + T_PP) Y - Y (eps_Q + T_QQ) = V_PQ.
Expanding in T (Y0 = -V/Delta, Y_{k+1} = (T_PP Y_k - Y_k T_QQ) / Delta, column-wise) gives
  order T^1:  V T V / Delta^2 - 1/2 {V V / Delta^2, T_PP}                        (= 13h H3)
  order T^2:  V T Delta^-1 T V / Delta^2 (Q-Q, "V T T V")  + folded  T_PP^2 and  T_PP (V T V)  mixed terms
so H4 = 1/2 (Y2 V+ + V Y2+) is the complete second-order-in-T, second-order-in-V term of the 13h expansion.
step A  r4 and r5 (5-body: lift V0..V4 from d0..d4, residual in d^5) of H3 and H4, and of H4 split into Q-only,
        P-only and mixed parts; also all orders in T at O(V^2) (Sylvester) and the T^3, T^4 terms.
step B  lambda^2 coefficient of the exact response (all classes, lambda +-0.125, +-0.25, +-0.5, 1; Richardson) vs r4(H4).
step C  prediction at lambda = 1: base + H3 + H4 + rho closure (13g) vs exact.
usage: phase13h_add.py <core,b' s3d,b' s6d | g> [g Nmax]
-> results/phase13/phase13h_add_cb.txt/.json, phase13h_add_g.txt/.json
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.linalg import solve_sylvester

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
import phase13b as B  # noqa: E402
import phase13d as D  # noqa: E402
import phase13e as E  # noqa: E402
import phase13f as F  # noqa: E402
import phase13g as G  # noqa: E402
import phase13h as H  # noqa: E402

OUT = H.OUT


def build(which, gmax):
    t0 = time.time()
    paths = {}
    if 'g' in which:
        return {'g': build_g(t0, gmax)}
    cb = F.core_blocks()
    for b in cb:
        b['W'] = G.tape_W(f'vcm_d{b["N"]}_{b["t"]}e')
        b['H0'] = b['H']
    paths['core'] = dict(blocks=cb, d0=None, V={'3s', '3d'}, lf=G.fit_labels(cb))
    data = {sp: {(N, t): D.load(sp, N, t) for N in range(1, 6) for L, twoS, m, t in D.term_list(N)} for sp in D.SPACES}
    lf = E.label_fit(data)
    dgp, _, _ = E.exact_d_gaps(data['s6d'], lf)
    shift = {o: {N: dgp[o][3] - dgp[o][N] for N in dgp[o]} for o in dgp}
    for sp in D.SPACES:
        f, _ = D.split(data[sp])
        bl = F.singles_blocks(data[sp], f, 'b', shift)
        for b in bl:
            b['W'] = data[sp][(b['N'], b['t'])]['W']
            b['H0'] = data[sp][(b['N'], b['t'])]['H']
        paths[f"b' {sp}"] = dict(blocks=bl, d0=B.ECORE, V={'3d'} | set(D.SPACES[sp]))
        paths[f"b' {sp}"]['lf'] = G.fit_labels(bl)
    print(f'core/singles built {time.time() - t0:.0f} s', flush=True)
    return paths


def build_g(t0, gmax):
    gb = [b for b in F.g_blocks() if b['N'] <= gmax]
    gruns = {1: 'vq_v4p_{}e_ivo', 2: 'vq_v3p_{}e_ivo', 3: 'vq_v2p_{}e_ivo', 4: 'vq_v1p_{}e_ivo_r1d1g',
             5: 'vq_v0p_{}e_ivo_r3d1g'}
    for bg in gb:
        run_ = gruns[bg['N']].format(bg['t'])
        z = np.load(os.path.join(ROOT, 'results', run_ + '.npz'), allow_pickle=True)
        lab = [str(x) for x in z['labels']]
        keep = [i for i, l in enumerate(lab) if set(B.cfg_orbitals(l)) <= {'3d', '11g'}]
        bg['W'] = H.tape_W_multi(run_, {'g': keep})['g']
        bg['H0'] = bg['H']
        print(f'  g d{bg["N"]} {bg["t"]} {time.time() - t0:.0f} s', flush=True)
    print(f'g built {time.time() - t0:.0f} s', flush=True)
    return dict(blocks=gb, d0=B.ECORE, V={'3d', '11g'}, lf=G.fit_labels(gb))


def resid(blocks, d0, mats, Nmax):
    """Pure Nmax-body residual of a linear H_eff contribution (P x P matrices per block, CSF basis)."""
    vecs = {} if d0 is None else {0: np.array([0.0])}
    by = defaultdict(dict)
    for b in blocks:
        if b['N'] > Nmax or (b['N'] == 0 and d0 is not None):
            continue
        by[b['N']][(b['L'], b['twoS'])] = b['U'].T @ mats[id(b)] @ b['U']
    for N, bl in by.items():
        vecs[N] = B.vec_from_blocks(N, bl)
    V = S.nbody_parts([S.from_vec(N, vecs[N]) for N in range(Nmax)])
    return vecs[Nmax] - S.to_vec(Nmax, S.apply_parts(V, Nmax))


def nrm(N, x):
    return float(np.sqrt(S.ip(N, x, x)))


def series(blocks, eps, classes, kmax=4):
    """Per block: {k: H_k} (order T^k, O(V^2)), H4 parts, and the Sylvester (all orders in T) matrix."""
    out = {k: {} for k in range(kmax + 1)}
    parts = {'Q': {}, 'P': {}, 'mix': {}}
    syl = {}
    for b in blocks:
        P, Q = b['P'], b['Q']
        T = sum(b['T'][c] for c in classes)
        if not Q:
            for d in list(out.values()) + list(parts.values()) + [syl]:
                d[id(b)] = np.zeros((len(P), len(P)))
            continue
        e = eps[id(b)]
        EP = e[P[0]]
        dl = e[Q] - EP
        V = b['H'][np.ix_(P, Q)]
        TP, TQ = T[np.ix_(P, P)], T[np.ix_(Q, Q)]
        herm = lambda Y: 0.5 * (Y @ V.T + V @ Y.T)  # noqa: E731

        def step(Y, tp, tq):
            return (tp @ Y - Y @ tq) / dl
        Y = -V / dl
        for k in range(kmax + 1):
            out[k][id(b)] = herm(Y)
            Y = step(Y, TP, TQ)
        Y0 = -V / dl
        z = np.zeros_like(TP)
        hQ = herm(step(step(Y0, z, TQ), z, TQ))
        hP = herm(step(step(Y0, TP, 0 * TQ), TP, 0 * TQ))
        parts['Q'][id(b)], parts['P'][id(b)] = hQ, hP
        parts['mix'][id(b)] = out[2][id(b)] - hQ - hP
        Ys = solve_sylvester(EP * np.eye(len(P)) + TP, -(np.diag(e[Q]) + TQ), V)
        syl[id(b)] = herm(Ys)
    return out, parts, syl


def main():
    which = sys.argv[1].split(',')        # core,b' s3d,b' s6d  or  g
    gmax = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tag = 'g' if which == ['g'] else 'cb'
    PATHS = [x for x in ('core', 'g', "b' s3d", "b' s6d") if x in which]
    t0 = time.time()
    w = []
    a = w.append
    js = {}
    a('# Phase 13h addendum: second order in T')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip())
    paths = build(which, gmax)
    a(f'\nBuild {time.time() - t0:.0f} s.')
    for nm, p in paths.items():
        p['const'], _ = H.decompose(p['blocks'], p['V'], p['lf'])
        pres = {c: max(np.abs(b['T'][c]).max() for b in p['blocks']) for c in H.CLASSES}
        p['classes'] = [c for c, x in pres.items() if x > 1e-12]
        p['eps0'], _ = H.base_eps(p['blocks'], p['const'], 0.0)
        p['eps1'], _ = H.base_eps(p['blocks'], p['const'], 1.0)
    print(f'decomposed {time.time() - t0:.0f} s', flush=True)

    allc = lambda p, l: {c: l for c in p['classes']}  # noqa: E731
    res = {}
    for nm in PATHS:
        p = paths[nm]
        bl, d0 = p['blocks'], p['d0']
        Nmax = max(b['N'] for b in bl)
        base = H.run(bl, d0, lambda b: H.model_H(b, p['eps0'], {}))
        V3, V3t = base['V3'], base['V3t']
        ser, parts, syl = series(bl, p['eps0'], p['classes'])
        # check: order-1 term = 13h third_order summed over classes
        r3h = sum(sum(H.r4_of(bl, d0, m) for m in H.third_order(bl, p['eps0'], c)) for c in p['classes'])
        r = {k: resid(bl, d0, ser[k], 4) for k in ser}
        chk = nrm(4, r[1] - r3h) / nrm(4, r3h)
        TPm = {id(b): sum(b['T'][c] for c in p['classes'])[np.ix_(b['P'], b['P'])] for b in bl}
        rsyl = resid(bl, d0, {k: syl[k] + TPm[k] for k in syl}, 4)
        rp = {k: resid(bl, d0, v, 4) for k, v in parts.items()}
        r5 = {k: (resid(bl, d0, ser[k], 5) if Nmax >= 5 else None) for k in (1, 2)}
        # step B: exact lambda^2 coefficient
        rl = {l: H.run(bl, d0, lambda b, l=l: H.model_H(b, p['eps0'], allc(p, l))) for l in (0.125, -0.125, 0.25, -0.25,
                                                                                              0.5, -0.5, 1.0)}
        q = {h: (rl[h]['r'] + rl[-h]['r'] - 2 * base['r']) / (2 * h ** 2) for h in (0.125, 0.25, 0.5)}
        qR = (4 * q[0.125] - q[0.25]) / 3
        s1 = (rl[0.125]['r'] - rl[-0.125]['r']) / 0.25
        s1R = (4 * s1 - (rl[0.25]['r'] - rl[-0.25]['r']) / 0.5) / 3
        res[nm] = dict(base=base, r=r, chk=chk, rsyl=rsyl, rp=rp, r5=r5, qR=qR, q=q, s1R=s1R, rl=rl, V3=V3, V3t=V3t)
        print(f'{nm} {time.time() - t0:.0f} s', flush=True)

    a('\n## Step A. Orders in T at O(V^2) (baseline units: |V3|, |V3 t| of the lambda = 0 model)')
    a('| path | r4(T^1 = H3)/|V3| | r4(T^2 = H4)/|V3| | H4 parts Q / P / mixed | r4(T^3) | r4(T^4) | r4(all T, '
      'Sylvester)/|V3| | 5-body |r5(H3)|, |r5(H4)| /|V3| | check T^1 vs 13h |')
    a('|---|---|---|---|---|---|---|---|---|')
    for nm in PATHS:
        x = res[nm]
        V3 = x['V3']
        f5 = ' / '.join('-' if x['r5'][k] is None else f'{nrm(5, x["r5"][k]) / V3:.4f}' for k in (1, 2))
        a(f'| {nm} | {nrm(4, x["r"][1]) / V3:.4f} | {nrm(4, x["r"][2]) / V3:.4f} | ' +
          ' / '.join(f'{nrm(4, x["rp"][k]) / V3:.4f}' for k in ('Q', 'P', 'mix')) +
          f' | {nrm(4, x["r"][3]) / V3:.4f} | {nrm(4, x["r"][4]) / V3:.4f} | {nrm(4, x["rsyl"]) / V3:.4f} | {f5} | '
          f'{x["chk"]:.1e} |')

    a('\n## Step B. lambda^2 coefficient of the exact response (all classes, rho = 0) vs r4(H4)')
    a('q(h) = (r4(h) + r4(-h) - 2 r4(0)) / (2 h^2) (the lambda^2 coefficient); Richardson q = (4 q(0.125) - q(0.25)) / 3. Units |V3| (baseline).')
    a('| path | |q(0.125)| / |q(0.25)| / |q(0.5)| | |q Richardson| | |r4(H4)| | |q - r4(H4)| / |q| | angle | '
      '|slope (Richardson)| vs |r4(H3)|: dev |')
    a('|---|---|---|---|---|---|---|')
    jsB = {}
    for nm in PATHS:
        x = res[nm]
        V3 = x['V3']
        dq = x['qR'] - x['r'][2]
        cos = S.ip(4, x['qR'], x['r'][2]) / (nrm(4, x['qR']) * nrm(4, x['r'][2]))
        ds = x['s1R'] - x['r'][1]
        jsB[nm] = dict(q=[nrm(4, x['q'][h]) / V3 for h in (0.125, 0.25, 0.5)], qR=nrm(4, x['qR']) / V3,
                       H4=nrm(4, x['r'][2]) / V3, dev=nrm(4, dq) / nrm(4, x['qR']),
                       ang=float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), sdev=nrm(4, ds) / nrm(4, x['s1R']))
        a(f'| {nm} | ' + ' / '.join(f'{v:.4f}' for v in jsB[nm]['q']) + f' | {jsB[nm]["qR"]:.4f} | {jsB[nm]["H4"]:.4f} | '
          f'{jsB[nm]["dev"]:.3f} | {jsB[nm]["ang"]:.1f} | {jsB[nm]["sdev"]:.3f} |')
    js['stepB'] = jsB

    a('\n## Step C. Prediction at lambda = 1 (units: exact |V3| at lambda = 1, as in 13h step 4)')
    a('| path | exact r4/|V3| | 13h pred (base+H3+rho) | + H4 | + all T at O(V^2) (base + Sylvester + rho) | '
      'err 13h / +H4 / Sylvester (|exact - pred|/|exact r4|) | exact r4/t | pred r4/t (+H4) | rho-closure part |')
    a('|---|---|---|---|---|---|---|---|---|')
    jsC = {}
    for nm in PATHS:
        p, x = paths[nm], res[nm]
        bl, d0 = p['blocks'], p['d0']
        rr = H.closure_r4(bl, d0, p['eps1'])
        ex = H.run(bl, d0, lambda b: H.model_H(b, p['eps1'], allc(p, 1.0)))
        V3, V3t = ex['V3'], ex['V3t']
        pr = {'13h': x['base']['r'] + x['r'][1] + rr}
        pr['H4'] = pr['13h'] + x['r'][2]
        pr['syl'] = x['base']['r'] + x['rsyl'] + rr
        ne = nrm(4, ex['r'])
        jsC[nm] = dict(ex=ne / V3, ex_t=ex['ratio'], rho=nrm(4, rr) / V3,
                       **{k: nrm(4, v) / V3 for k, v in pr.items()},
                       **{'err_' + k: nrm(4, ex['r'] - v) / ne for k, v in pr.items()})
        jsC[nm]['pred_t'] = nrm(4, pr['H4']) / V3t          # |r4| / |V3 t|, as ratio_t (13c metrics)
        c = jsC[nm]
        a(f'| {nm} | {c["ex"]:.4f} | {c["13h"]:.4f} | {c["H4"]:.4f} | {c["syl"]:.4f} | '
          f'{c["err_13h"]:.3f} / {c["err_H4"]:.3f} / {c["err_syl"]:.3f} | {c["ex_t"]:.4f} | {c["pred_t"]:.4f} | '
          f'{c["rho"]:.4f} |')
    js['stepC'] = jsC
    js['stepA'] = {nm: dict(r={k: nrm(4, v) / x['V3'] for k, v in x['r'].items()},
                            parts={k: nrm(4, v) / x['V3'] for k, v in x['rp'].items()},
                            syl=nrm(4, x['rsyl']) / x['V3'], chk=x['chk'],
                            r5={k: (None if v is None else nrm(5, v) / x['V3']) for k, v in x['r5'].items()})
                   for nm, x in res.items()}
    a(f'\nAnalysis wall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, f'phase13h_add_{tag}.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    open(os.path.join(OUT, f'phase13h_add_{tag}.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
