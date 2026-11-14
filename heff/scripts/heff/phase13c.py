#!/usr/bin/env python3
"""Phase 13c: N dependence in the core minimal space (follow-ups of 13b) and valence d^4/d^5 in s-free spaces.

core      h4 from d^4 alone and the E_av series; Delta(N) = Delta0 + N delta closure fit; (N - 3) scaling of the t-part of a
          four-body operator (algebra) vs the exact-CI t-coefficients of 13b.   -> results/phase13/phase13c_core.txt/.json
orbitals  virtual d / g orbitals of each reduced space with their Phase 11c denominators and path weights.
valence   stage 1: 3d1g d^4/d^5; 5d1g, 6d2g d^4; channels 3d (d only) and 1d1g (3d + 11g, g only), d^0..d^4.
          -> results/phase13/phase13c_valence.txt/.json
"""
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
import phase13b as B  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase13')
EK, TK = B.EK, B.TK


def core_data():
    """Per core run: H (ATOMCI), P/Q indices, seniority map U (13b), N, (L, 2S)."""
    js = json.load(open(os.path.join(OUT, 'phase13b_b.json')))
    out = {}
    for name, N, term, mult in B.core_runs():
        H, lab = B.load_run(name)
        L, twoS = B.term_key(N, term)
        out[name] = dict(N=N, L=L, twoS=twoS, H=H, U=np.array(js['maps'][name]),
                         P=[i for i, l in enumerate(lab) if l.startswith('3s2')],
                         Q=[i for i, l in enumerate(lab) if not l.startswith('3s2')])
    return out, js['Delta']


def core_vecs(data, kind, Delta=None):
    """Component vectors d^0..d^9: kind 'exact' (des Cloizeaux) or 'closure' with Delta a number or a function of N."""
    from atomci_heff import heff
    blocks = {}
    for d in data.values():
        H, Pi, Qi, U, m = d['H'], d['P'], d['Q'], d['U'], len(d['P'])
        if kind == 'exact':
            if Qi:
                e, C, wP, _ = heff.spectrum(H, Pi)
                st, _ = heff.select_states(wP, m)
                A = heff.exact_heff(H, Pi, st)['hermitian']
            else:
                A = H[np.ix_(Pi, Pi)]
        else:
            D = Delta(d['N']) if callable(Delta) else Delta
            A = H[np.ix_(Pi, Pi)] - (H[np.ix_(Pi, Qi)] @ H[np.ix_(Qi, Pi)] / D if Qi else 0)
        blocks.setdefault(d['N'], {})[(d['L'], d['twoS'])] = U.T @ A @ U
    return {N: B.vec_from_blocks(N, blocks[N]) for N in range(10)}


def core_step():
    w = []
    a = w.append
    js = {}
    data, Delta = core_data()
    ex = core_vecs(data, 'exact')
    V = B.extract(ex)
    a('# Phase 13c core: follow-ups of 13b (core minimal space, exact des Cloizeaux H_eff)')

    # 1. h4 from d^4 alone
    V4 = S.from_vec(4, ex[4]) - S.apply_parts(V, 4)
    h = [np.trace(Vk) / comb(10, k) for k, Vk in enumerate(V)] + [np.trace(V4) / comb(10, 4)]
    V5 = S.from_vec(5, ex[5]) - S.apply_parts(V + [V4], 5)
    h5 = np.trace(V5) / comb(10, 5)
    a('\n## 1. Scalar four-body term h4 = tr V4 / C(10,4) from d^4 alone (V4 = d^4 residual after V0..V3)')
    a(f'h4 = {1e3 * h[4]:+.6f} mEh;  h5 (from d^5 after V0..V4) = {1e3 * h5:+.6f} mEh')
    a('| N | E_av residual act - pred(V0..V3) | C(N,4) h4 | remainder | C(N,5) h5 | remainder after h5 |')
    a('|---|---|---|---|---|---|')
    rows = {}
    for N in range(4, 10):
        act = B.eav(N, ex[N])
        pred3 = sum(comb(N, k) * h[k] for k in range(4))
        r = act - pred3
        r4 = r - comb(N, 4) * h[4]
        r5 = r4 - comb(N, 5) * h5
        rows[N] = dict(resid=r, c4=comb(N, 4) * h[4], rem=r4, c5=comb(N, 5) * h5, rem5=r5)
        a(f'| {N} | {1e3 * r:+.4f} | {1e3 * comb(N, 4) * h[4]:+.4f} | {1e3 * r4:+.4f} | {1e3 * comb(N, 5) * h5:+.4f} | '
          f'{1e3 * r5:+.4f} |')
    js['h'] = dict(h0_3=h[:4], h4=h[4], h5=h5, rows=rows)

    # 2. Delta(N) closure
    a('\n## 2. Closure with Delta(N) = Delta0 + N delta, fitted to the exact H_eff(N), N = 1..8 (all components, '
      'metric weights)')

    def misfit(par, Ns=range(1, 9)):
        cl = core_vecs(data, 'closure', lambda N: par[0] + N * par[1])
        return np.concatenate([np.sqrt(S.weights(N)) * (cl[N] - ex[N]) for N in Ns])

    from scipy.optimize import least_squares
    # per-N Delta first (one parameter per N)
    perN = {}
    for N in range(1, 9):
        r = least_squares(lambda p: misfit((p[0], 0.0), [N]), [Delta])
        perN[N] = (float(r.x[0]), float(np.sqrt(np.sum(r.fun ** 2))))
    a('Per-N best Delta (one parameter per N) and the remaining weighted misfit |closure - exact| (mEh):')
    a('| N | ' + ' | '.join(str(N) for N in perN) + ' |')
    a('|---|' + '---|' * len(perN))
    a('| Delta(N) (Eh) | ' + ' | '.join(f'{x[0]:.5f}' for x in perN.values()) + ' |')
    a('| misfit (mEh) | ' + ' | '.join(f'{1e3 * x[1]:.4f}' for x in perN.values()) + ' |')
    r = least_squares(misfit, [Delta, 0.0])
    D0, dl = r.x
    a(f'Linear fit: Delta0 = {D0:.6f} Eh, delta = {dl:+.6f} Eh per electron; total weighted misfit '
      f'{1e3 * np.sqrt(np.sum(r.fun ** 2)):.4f} mEh (fixed Delta = {Delta:.6f}: '
      f'{1e3 * np.sqrt(np.sum(misfit((Delta, 0.0)) ** 2)):.4f} mEh)')
    cl = core_vecs(data, 'closure', lambda N: D0 + N * dl)
    Vc = B.extract(cl)
    a('\nFock expansion of the Delta(N) closure (V0..V3 from its d^0..d^3) vs that of the exact H_eff: residual of the '
      'd^N prediction, weighted rms per state and E_av (mEh)')
    a('| N | exact: rms | exact: E_av | Delta(N) closure: rms | Delta(N) closure: E_av | |closure - exact| rms |')
    a('|---|---|---|---|---|---|')
    jd = {}
    for N in range(4, 10):
        re_ = ex[N] - B.predict(V, N)
        rc = cl[N] - B.predict(Vc, N)
        dd = cl[N] - ex[N]
        f = lambda x: np.sqrt(S.ip(N, x, x) / comb(10, N))
        jd[N] = dict(exact_rms=f(re_), exact_eav=B.eav(N, re_ * 0 + ex[N]) - B.eav(N, B.predict(V, N)),
                     clo_rms=f(rc), clo_eav=B.eav(N, cl[N]) - B.eav(N, B.predict(Vc, N)), diff=f(dd))
        a(f'| {N} | {1e3 * jd[N]["exact_rms"]:.4f} | {1e3 * jd[N]["exact_eav"]:+.4f} | {1e3 * jd[N]["clo_rms"]:.4f} | '
          f'{1e3 * jd[N]["clo_eav"]:+.4f} | {1e3 * jd[N]["diff"]:.4f} |')
    # angle between the residual vectors (d^4, d^5)
    for N in (4, 5):
        re_ = ex[N] - B.predict(V, N)
        rc = cl[N] - B.predict(Vc, N)
        cos = S.ip(N, re_, rc) / np.sqrt(S.ip(N, re_, re_) * S.ip(N, rc, rc))
        a(f'd^{N}: angle between the exact and the Delta(N)-closure four-body residuals {np.degrees(np.arccos(cos)):.1f} deg, '
          f'norm ratio {np.sqrt(S.ip(N, rc, rc) / S.ip(N, re_, re_)):.3f}')
        jd[f'angle{N}'] = float(np.degrees(np.arccos(cos)))
    js['delta_fit'] = dict(perN=perN, D0=D0, delta=dl, rows=jd)

    # 3. (N - 3) scaling of the t-part of a four-body operator
    a('\n## 3. t-part of a pure four-body operator: N dependence (algebra) and the 13b exact-CI t-coefficients')
    ops, _, _ = B.ops_fock()
    tN = {N: B.lifted_t(N) for N in range(3, 8)}
    cells = []
    for k in TK:
        M4 = S.from_vec(4, S.to_vec(4, S.lift(*ops[k], 4)))         # d^4 matrix of t_k, as a pure 4-body operator
        cs = []
        for N in range(4, 8):
            c = B.t_coords(N, S.to_vec(N, S.lift(M4, 4, N)), tN[N])
            other = max(abs(c[j]) for j in TK if j != k)
            cs.append((c[k], other))
        cells.append(cs)
        a(f'- 4-body operator with d^4 matrix t_{k[1]}: coefficient on lifted t_{k[1]}^(N), N = 4..7: '
          + ', '.join(f'{x:.6f}' for x, _ in cs) + f' (expected N - 3 = 1, 2, 3, 4); other t coefficients '
          f'<= {max(o for _, o in cs):.1e}')
    js['t_scaling'] = {k: [x for x, _ in cs] for k, cs in zip(TK, cells)}
    # exact CI: t-coefficients vs V3 + (N - 3) x (t-part of V4)
    T3 = S.jl_table(3)
    G3 = np.array([[S.ip(3, T3[p], T3[q]) for q in TK] for p in TK])
    t3 = B.t_coords(3, ex[3], tN[3])
    t4v = B.t_coords(4, S.to_vec(4, V4), tN[4])
    base = np.array([t3[k] for k in TK])
    a('\nExact CI (13b): t-coefficients on the lifted t_i^(N) vs the prediction V3 + (N - 3) x t-part(V4), ratio to N = 3 '
      '(metric-weighted projection on the d^3 t-vector)')
    a('| N | actual ratio | predicted ratio (V3 + (N-3) V4) | actual - 1 | (N - 3) x (ratio(4) - 1) |')
    a('|---|---|---|---|---|')
    jr = {}
    r4 = None
    for N in range(3, 8):
        c = B.t_coords(N, ex[N], tN[N])
        x = np.array([c[k] for k in TK])
        pred = base + (N - 3) * np.array([t4v[k] for k in TK])
        ra = float(base @ G3 @ x / (base @ G3 @ base))
        rp = float(base @ G3 @ pred / (base @ G3 @ base))
        if N == 4:
            r4 = ra - 1
        jr[N] = dict(actual=ra, pred=rp)
        a(f'| {N} | {ra:.6f} | {rp:.6f} | {ra - 1:+.6f} | {(N - 3) * (r4 if r4 is not None else 0):+.6f} |')
    js['t_ratio'] = jr
    json.dump(js, open(os.path.join(OUT, 'phase13c_core.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, 'phase13c_core.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


# ---------------------------------------------------------------- valence (stage 1)

VSPACES = {'3d1g': (0, 0, 3, 0, 1, 0), '5d1g': (0, 0, 5, 0, 1, 0), '6d2g': (0, 0, 6, 0, 2, 0),
           '3d': (0, 0, 3, 0, 0, 0), '1d1g': (0, 0, 1, 0, 1, 0)}


def orbitals_step():
    """Virtual d / g orbitals per space with the Phase 11c denominators (n = 3) and path weights."""
    txt = open(os.path.join(ROOT, 'results', 'phase11', 'phase11c.txt')).read().split('\n')
    D, Wt = {}, {}
    sec = None
    for l in txt:
        if l.startswith('# Path'):
            sec = l
        if l.startswith('| 1') and sec and l.count('|') == 6:
            c = [x.strip() for x in l.strip('|').split('|')]
            D[c[0]] = float(c[2])
        if l.startswith('| 1') and l.count('|') == 5:
            c = [x.strip() for x in l.strip('|').split('|')]
            Wt[c[0]] = float(c[2])
    out = []
    out.append('| space | virtual d (Delta_i n=3, Eh; share of the 3d->nd closure) | virtual g (Delta_i, share of 3d->ng) |')
    out.append('|---|---|---|')
    for nm, v in {**VSPACES, 'QZP': (7, 6, 6, 4, 3, 2)}.items():
        orbs = sorted(B.space_orbitals(v) - {'3d'})
        dd = [o for o in orbs if o.endswith('d')]
        gg = [o for o in orbs if o.endswith('g')]
        f = lambda o: f'{o} ({D[o]:.2f}; {Wt[o]:.2f})'
        out.append(f'| {nm} | ' + (', '.join(f(o) for o in dd) or '-') + ' | ' + (', '.join(f(o) for o in gg) or '-') +
                   ' |')
    return out, D, Wt


def space_analysis(sp, with_d5=False):
    orbs = B.space_orbitals(VSPACES[sp])
    vecs, info = B.valence_vecs(orbs)
    gaps = {}
    for N in ((4, 5) if with_d5 else (4,)):
        gaps[N] = B.gap_check(sp, N)
        blocks = {}
        for L, twoS, vs in S.terms(N):
            t = f'{twoS + 1}{S.LCH[L]}'
            Bk, inf = B.valence_heff(B.VRUN[N].format(t, sp), N, t, orbs, len(vs))
            blocks[(L, twoS)] = Bk
            info[f'd{N} {t}'] = inf
        vecs[N] = B.vec_from_blocks(N, blocks)
    return vecs, info, gaps


def metrics(vecs, V=None):
    """d^4 metrics: |lift V3| (all, t-part), |r4|, ratio, max-term residual, h4, JL split of r4."""
    V = V or B.extract(vecs)
    L3 = S.to_vec(4, S.lift(V[3], 3, 4))
    x3 = S.to_vec(3, V[3])
    t3, _, c3 = B.tpart(3, x3)
    Lt = S.to_vec(4, S.lift(S.from_vec(3, t3), 3, 4))
    r = vecs[4] - B.predict(V, 4)
    n = lambda x: float(np.sqrt(S.ip(4, x, x)))
    c, nr = B.jl_split(4, r)
    tab = S.jl_table(4)
    er = B.eig_resid(4, vecs[4], B.predict(V, 4))
    return dict(V3=n(L3), V3t=n(Lt), r4=n(r), ratio=n(r) / n(L3), ratio_t=n(r) / n(Lt),
                maxterm=max(abs(x) for v in er.values() for x in v), h4=B.eav(4, vecs[4]) - B.eav(4, B.predict(V, 4)),
                e=float(np.sqrt(sum(nr[k] ** 2 for k in EK))), t=float(np.sqrt(sum(nr[k] ** 2 for k in TK))),
                f=float(np.sqrt(sum(nr[k] ** 2 for k in tab if k.startswith('f')))), jl=nr, r=r, eig=er, V=V)


def valence_step():
    w = []
    a = w.append
    js = {}
    a('# Phase 13c stage 1: valence d^4 (d^5 for 3d1g) in s-free spaces; V0..V3 from d^0..d^3 subsets of the QZP '
      'exports (d^0 = [Ar] core energy); exact des Cloizeaux H_eff, seniority basis via the 13b maps')
    o, D, Wt = orbitals_step()
    a('\n## Orbitals (Phase 11c: Delta_i at n = 3; share = <R_i,B2>/<B2,B2> of that path)')
    w += o
    res = {}
    for sp in ('3d1g', '5d1g', '6d2g', '3d', '1d1g'):
        vecs, info, gaps = space_analysis(sp, with_d5=(sp == '3d1g'))
        m = metrics(vecs)
        res[sp] = dict(vecs=vecs, info=info, gaps=gaps, m=m)
    # core (13b) for comparison
    data, _ = core_data()
    cv = core_vecs(data, 'exact')
    res['core'] = dict(vecs=cv, m=metrics(cv))

    a('\n## Targets (d^4; d^5 for 3d1g)')
    a('| space | N | max CSFs | min P weight | max ambiguity | max cond | roots below a target | min gap (eV, block) |')
    a('|---|---|---|---|---|---|---|---|')
    for sp, r in res.items():
        for N, g in r.get('gaps', {}).items():
            tmin = min(g, key=lambda t: g[t]['gap_eV'])
            a(f'| {sp} | {N} | {max(x["ncsf"] for x in g.values())} | {min(min(x["wP"]) for x in g.values()):.4f} | '
              f'{max(x["amb"] for x in g.values()):.2e} | {max(x["cond"] for x in g.values()):.4f} | '
              f'{sum(x["below"] for x in g.values())} | {g[tmin]["gap_eV"]:.1f} ({tmin}, {g[tmin]["dom"]}) |')

    a('\n## Comparison (d^4, mEh; norms in the JL metric, sum over states)')
    a('| space | |V3 lifted| | |t-part of V3 lifted| | |r4| | r4/|V3| | r4/|V3 t| | max-term resid | h4 | |e| | |t| | |f| |')
    a('|---|---|---|---|---|---|---|---|---|---|---|')
    for sp, r in res.items():
        m = r['m']
        a(f'| {sp} | {1e3 * m["V3"]:.3f} | {1e3 * m["V3t"]:.3f} | {1e3 * m["r4"]:.4f} | {m["ratio"]:.4f} | '
          f'{m["ratio_t"]:.4f} | {1e3 * m["maxterm"]:.4f} | {1e3 * m["h4"]:+.4f} | {1e3 * m["e"]:.4f} | '
          f'{1e3 * m["t"]:.4f} | {1e3 * m["f"]:.4f} |')
    # non-additivity of the channels
    a('\n## Channel additivity (3d1g = 3d channel + g channel?)')
    for key, lab in (('r', 'four-body residual r4 (d^4)'),):
        x, y, z = res['3d1g']['m'][key], res['3d']['m'][key], res['1d1g']['m'][key]
        na = x - y - z
        n = lambda v: float(np.sqrt(S.ip(4, v, v)))
        a(f'- {lab}: |3d1g| {1e3 * n(x):.4f}, |d| {1e3 * n(y):.4f}, |g| {1e3 * n(z):.4f}, |d + g| {1e3 * n(y + z):.4f}, '
          f'|3d1g - (d + g)| {1e3 * n(na):.4f} mEh ({100 * n(na) / n(x):.0f}% of 3d1g)')
        js['nonadd_r4'] = dict(full=n(x), d=n(y), g=n(z), na=n(na))
    for N_, lab in ((3, 'V3 (d^3 matrix, t-part)'),):
        tt = {sp: B.tpart(3, S.to_vec(3, res[sp]['m']['V'][3]))[0] for sp in ('3d1g', '3d', '1d1g')}
        n = lambda v: float(np.sqrt(S.ip(3, v, v)))
        na = tt['3d1g'] - tt['3d'] - tt['1d1g']
        a(f'- {lab}: |3d1g| {1e3 * n(tt["3d1g"]):.4f}, |d| {1e3 * n(tt["3d"]):.4f}, |g| {1e3 * n(tt["1d1g"]):.4f}, '
          f'|3d1g - (d + g)| {1e3 * n(na):.4f} mEh ({100 * n(na) / n(tt["3d1g"]):.0f}%)')
    a('\nJL split of the d^4 four-body residual (normalized, mEh):')
    tab = S.jl_table(4)
    a('| space | ' + ' | '.join(tab) + ' |')
    a('|---|' + '---|' * len(tab))
    for sp, r in res.items():
        a(f'| {sp} | ' + ' | '.join(f'{1e3 * r["m"]["jl"][k]:+.3f}' for k in tab) + ' |')

    # 3d1g: full stage-1a analysis
    r = res['3d1g']
    vecs, V = r['vecs'], r['m']['V']
    a('\n## 3d1g: d^4 and d^5 residuals for the V3 variants (sorted eigenvalues per (L, S) block, mEh)')
    rv = {}
    for var in ('full', 'ut4', 'zero'):
        Vv = B.v3_variant(V, var)
        for N in (4, 5):
            vp = B.predict(Vv, N)
            rv[(var, N)] = dict(eig=B.eig_resid(N, vecs[N], vp), vec=vecs[N] - vp, eav=B.eav(N, vecs[N]) - B.eav(N, vp))
    V4 = V + [S.from_vec(4, vecs[4]) - S.apply_parts(V, 4)]
    vp5 = B.predict(V4, 5)
    e5 = B.eig_resid(5, vecs[5], vp5)
    for N in (4, 5):
        terms = list(rv[('full', N)]['eig'])
        a(f'\n| d^{N} term | V3 full | V3 on span(u,t4) | V3 = 0 |' + (' d0..d4 (V4) |' if N == 5 else ''))
        a('|---|---|---|---|' + ('---|' if N == 5 else ''))
        for t in terms:
            a(f'| {t} | ' + ' | '.join('/'.join(f'{1e3 * x:+.4f}' for x in rv[(v, N)]['eig'][t])
                                     for v in ('full', 'ut4', 'zero')) +
              (' | ' + '/'.join(f'{1e3 * x:+.4f}' for x in e5[t]) if N == 5 else '') + ' |')
        rms = lambda x: 1e3 * np.sqrt(S.ip(N, x, x) / comb(10, N))
        a('| rms per state | ' + ' | '.join(f'{rms(rv[(v, N)]["vec"]):.4f}' for v in ('full', 'ut4', 'zero')) +
          (f' | {rms(vecs[5] - vp5):.4f}' if N == 5 else '') + ' |')
        a('| E_av residual | ' + ' | '.join(f'{1e3 * rv[(v, N)]["eav"]:+.4f}' for v in ('full', 'ut4', 'zero')) +
          (f' | {1e3 * (B.eav(5, vecs[5]) - B.eav(5, vp5)):+.4f}' if N == 5 else '') + ' |')
    single = [t for t in rv[('full', 5)]['eig'] if len(rv[('full', 5)]['eig'][t]) == 1]
    dev = max(abs(rv[('full', 5)]['eig'][t][0] - rv[('ut4', 5)]['eig'][t][0]) for t in single)
    a(f'\nThe 9 singly occurring d^5 terms {single}: |full - ut4| <= {1e3 * dev:.1e} mEh (the t-part of V3 has no '
      f'diagonal d^5 element). Their residuals (V3 full): ' + ', '.join(
          f'{t} {1e3 * rv[("full", 5)]["eig"][t][0]:+.4f}' for t in single) + ' mEh; rms over them '
      f'{1e3 * np.sqrt(np.mean([rv[("full", 5)]["eig"][t][0] ** 2 for t in single])):.4f} mEh; after V4: rms '
      f'{1e3 * np.sqrt(np.mean([e5[t][0] ** 2 for t in single])):.4f} mEh')
    r5 = rv[('full', 5)]['vec']
    c, nr = B.jl_split(5, r5)
    tab5 = S.jl_table(5)
    a(f'd^5 residual (V3 full) JL split: |e| {1e3 * np.sqrt(sum(nr[k] ** 2 for k in EK)):.4f}, |t| '
      f'{1e3 * np.sqrt(sum(nr[k] ** 2 for k in TK)):.4f}, |f| '
      f'{1e3 * np.sqrt(sum(nr[k] ** 2 for k in tab5 if k.startswith("f"))):.4f} mEh')
    js['spaces'] = {sp: dict(m={k: v for k, v in r['m'].items() if k not in ('r', 'V')},
                             info=r.get('info'), gaps=r.get('gaps')) for sp, r in res.items()}
    js['3d1g_variants'] = {f'{v} {N}': dict(eig=x['eig'], eav=x['eav']) for (v, N), x in rv.items()}
    js['3d1g_d5_from_d4'] = e5
    json.dump(js, open(os.path.join(OUT, 'phase13c_valence.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, 'phase13c_valence.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    {'core': core_step, 'valence': valence_step,
     'orbitals': lambda: print('\n'.join(orbitals_step()[0]))}[sys.argv[1]]()
