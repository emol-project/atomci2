#!/usr/bin/env python3
"""Phase 17: our ab initio three- and four-body parameters against Hansen, Raassen, Uylings & Lister, NIM B 31, 134
(1988) (references/Hansen_NuclInstrumMethodsPhysResB_31_134_1988.pdf; values read from the page images). No new runs.
  ident    Table 4 norms against the Judd-Leavitt t_i and our T1..T4; lift properties of f and t
  compare  f (their normalisation, Tr f^2 = 1 in d4) and T1..T4 (their O3'' normalisation) side by side
  fits     Fe III NIST fits: their vector fixed, every pair swap and sign flip of it, all twelve f + T1..T4 free
           (uniform and 2J+1 weights); d3 ions with T3, T4 free
  extra    2J+1-weighted T-only and (1b) fits; level-space distance of the f vectors; swap ranking (needs fits)
  hj       o_t, o_t' against Hansen & Judd, J. Phys. B 18, 2327 (1985): eq. (7), SO(5) labels by the operator Casimir,
           their Table 6 in the 1988 normalisation
  j82      sign of o_t: Table 1 of Judd, Hansen & Raassen, J. Phys. B 15, 1457 (1982) (u, v, w, o3'', o3''') against the
           JL t_i and T1..T4, element by element
  all      all of the above
Output: results/phase17/phase17q20.json"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase17q11 as Q  # noqa: E402
import phase17q12 as Q12  # noqa: E402
from atomci_heff import cfp_l, dshell as S  # noqa: E402

OUT = Q.OUT
JS = os.path.join(OUT, 'phase17q20.json')
FS = [f'f{i}' for i in range(1, 13)]
NORM3 = 162750.0                                          # Table 4: every three-electron operator has this d3 norm^2
MEH_PER_P = np.sqrt(NORM3) / Q.MEH_CM                     # 1 unit of p_i = 1.83813 mEh on a Tr O^2 = 1 operator

# Hansen et al. Table 2, III spectra, operators normalised to 1 in d4 (cm-1): (value, error)
H_FE = dict(zip(FS, [(248, 33), (9, 10), (346, 10), (-208, 10), (107, 7), (37, 7), (-11, 8), (332, 8), (-25, 7), (-32, 7),
                     (0, 8), (-60, 9)]))
H_CR = dict(zip(FS, [(281, 24), (3, 7), (696, 7), (-192, 7), (50, 4), (93, 4), (2, 6), (251, 5), (20, 4), (24, 4),
                     (-4, 6), (-112, 6)]))
# Table 1 (p3'', p3''', p_t, p_t'), each (value, error), normalised on O3'' in d3
H_P = {'v': [(-4.49, .02), (1.05, .02), (.072, .010), (.035, .012)], 'cr': [(-6.29, .02), (.81, .02), (.027, .008), (.014, .008)],
       'mn': [(-7.44, .02), (.71, .02), (-.008, .006), (-.005, .006)], 'fe': [(-4.97, .01), (2.26, .01), (.013, .010), (-.013, .009)]}
LAB = Q12.ION
FINAL = {'v': (8.658, .047, 1.528, .056), 'cr': (12.082, .050, 1.484, .056), 'mn': (14.324, .059, 1.472, .055),
         'fe': (9.482, .065, 3.626, .054)}                 # Final table of 17_report.md (projection)


def jlA():
    return {k: S.jl_norms()[k][4] for k in FS}


def ident():
    """Table 4 squared norms from the JL t_i (JL Table 9) and the orientation of our T1..T4 in d3."""
    T3 = {k: np.array(v, float) for k, v in S.jl_table(3).items() if k in ('t1', 't2', 't3', 't4')}
    n = {k: S.ip(3, v, v) for k, v in T3.items()}
    off = max(abs(S.ip(3, T3[a], T3[b])) for a, b in itertools.combinations(T3, 2))
    u = 3 * T3['t1'] + 2 * T3['t2']
    cand = {"O3'' (Table 4: 162750)": (5 / 18 * (u + 2 * T3['t4']), 162750.0),
            "O3''' (146475/11)": ((44 * u - 5 * T3['t4']) / 132, 146475 / 11),
            'O_t (18900)': (T3['t1'] - T3['t2'], 18900.0), "O_t' (560)": (T3['t3'] / 3, 560.0)}
    ax = dict(zip(['T1', 'T2', 'T3', 'T4'], Q.o_axes3()))
    res = dict(jl_norm2=n, jl_max_offdiag=off, cand={})
    print('JL d3 norms', n, 'max offdiag', off)
    for k, (v, tab) in cand.items():
        nn = S.ip(3, v, v)
        cos = {a: S.ip(3, v, x) / np.sqrt(nn) for a, x in ax.items()}
        res['cand'][k] = dict(norm2=nn, table4=tab, cos=cos)
        print(f'{k:24s} norm^2 {nn:.6f} (table {tab:.6f}) cos with ' + ' '.join(f'{a} {c:+.6f}' for a, c in cos.items()))
    # f, t in d6 against d4, d3: Tr and the conjugation rule the paper's d6 tables rest on
    d4, d6 = Q.ds(4), Q.ds(6)
    rows = {}
    A = jlA()
    for k in FS:
        tr4, tr6 = np.sum(d4.ops[k] ** 2), np.sum(d6.ops[k] ** 2)
        e4 = sorted(np.round(np.linalg.eigvalsh(d4.ops[k]), 6))
        e6 = sorted(np.round(np.linalg.eigvalsh(d6.ops[k]), 6))
        rows[k] = dict(A=A[k], tr4=float(tr4), tr6=float(tr6), same_spectrum=bool(np.allclose(e4, e6, atol=1e-5)))
    res['f_lift'] = rows
    print('f: Tr f^2 in d6 / A:', {k: round(v['tr6'] / v['A'], 8) for k, v in rows.items()})
    print('f: Tr f^2 in d4 / A:', {k: round(v['tr4'] / v['A'], 8) for k, v in rows.items()})
    print('f: d4 and d6 spectra equal:', all(v['same_spectrum'] for v in rows.values()))
    d3, d7 = Q.ds(3), Q.ds(7)
    tt = {}
    for k in ('T1', 'T2', 'T3', 'T4'):
        e3 = np.sort(np.linalg.eigvalsh(d3.ops[k]))
        e7 = np.sort(np.linalg.eigvalsh(d7.ops[k]))
        tt[k] = dict(minus=bool(np.allclose(e3, -e7[::-1], atol=1e-6)), plus=bool(np.allclose(e3, e7, atol=1e-6)),
                     tr6_over_tr3=float(np.sum(d6.ops[k] ** 2) / np.sum(d3.ops[k] ** 2)))
    res['t_lift'] = tt
    print('T: d7 spectrum = -d3 / +d3; Tr d6 / Tr d3:', tt)
    # where each f sits: share of Tr f^2 per matrix element of the d4 table (in particular f1 in the seniority-0 1S)
    res['f_shares'] = {}
    for k in FS:
        x = np.array(S.jl_table(4)[k], float)
        sh = {S.comp_label(4, c): float(wi * xi * xi / A[k]) for (c, wi), xi in zip(S.components(4), x) if abs(xi) > 1e-9}
        res['f_shares'][k] = sh
        print(k, 'Tr shares', {a: round(b, 3) for a, b in sorted(sh.items(), key=lambda t: -t[1])[:5]})
    return res


def ours():
    """Fe2+ sets of f, T3, T4 and the T1..T4 per ion ('corrected' = Final T1, T2 + chain T3, T4)."""
    v = json.load(open(os.path.join(OUT, 'phase17q12_defs.json')))['values']
    q18 = json.load(open(os.path.join(OUT, 'phase17q18.json')))['fe']['X-']
    sets = {'fam': v['fe']['corr'], 'fe_own': v['fe']['fe_own'], 'famiq': {k: q18[k] for k in Q12.EXTRA}}
    T = {}
    for ion, lab in LAB.items():
        a, _, b, _ = FINAL[ion]
        x = v['fe']['fe_own'] if ion == 'fe' else v[ion]['corr']
        T[ion] = dict(corr=[a, b, x['T3'], x['T4']], err=[FINAL[ion][1], FINAL[ion][3]],
                      base=[v[ion]['base'][k] for k in ('T1', 'T2', 'T3', 'T4')])
    return sets, T


def compare():
    sets, T = ours()
    A = jlA()
    res = dict(f={}, T={}, conv=MEH_PER_P)
    print(f'{"op":4s} {"sqrtA":>9s} {"fam":>8s} {"fe_own":>8s} {"famiq":>8s} {"H Fe III":>10s} {"fam/H":>7s} {"own/H":>7s} {"H Cr III":>10s}')
    for k in FS:
        s = np.sqrt(A[k])
        o = {n: sets[n][k] * s for n in sets}
        h, e = H_FE[k]
        res['f'][k] = dict(sqrtA=s, raw={n: sets[n][k] for n in sets}, norm=o, H=h, H_err=e, H_Cr=H_CR[k][0], H_Cr_err=H_CR[k][1])
        rt = (lambda a: f'{a / h:7.2f}') if abs(h) > 50 else (lambda a: '      -')
        print(f"{k:4s} {s:9.2f} {o['fam']:8.1f} {o['fe_own']:8.1f} {o['famiq']:8.1f} {h:6d}({e:2d}) {rt(o['fam'])} {rt(o['fe_own'])} {H_CR[k][0]:6d}({H_CR[k][1]:2d})")
    h = np.array([H_FE[k][0] for k in FS], float)
    res['vec'] = {}
    for n in sets:
        o = np.array([res['f'][k]['norm'][n] for k in FS])
        c = o @ h / np.linalg.norm(o) / np.linalg.norm(h)
        res['vec'][n] = dict(cos=float(c), norm=float(np.linalg.norm(o)), dist=float(np.linalg.norm(o - h)),
                             slope=float(o @ h / (h @ h)))
        print(n, res['vec'][n])
    res['vec']['H'] = dict(norm=float(np.linalg.norm(h)))
    print('T mapping, mEh per unit p:', MEH_PER_P)
    for ion in LAB:
        p = H_P[ion]
        t = T[ion]
        res['T'][ion] = dict(H_p=p, H_mEh=[[x * MEH_PER_P, e * MEH_PER_P] for x, e in p], ours=t)
        print(ion, 'H(mEh)', [round(x * MEH_PER_P, 3) for x, e in p], 'corr', [round(x, 3) for x in t['corr']],
              'base', [round(x, 3) for x in t['base']])
    return res


def wlsq(d, names, fixed, obs, p0, w=None, iters=200):
    """phase15u_crfit.lsq on d.levels with level weights w (sum w r^2); w = None must reproduce lsq."""
    w = np.ones(len(obs)) if w is None else np.asarray(w, float)
    p = np.array(p0, float)
    E = np.array([e for *_, e in obs])
    ref = None
    sw = np.sqrt(w)
    conv = False
    for _ in range(iters):
        H = sum(a * d.ops[n] for a, n in zip(p, names)) + sum(a * d.ops[n] for n, a in fixed.items())
        L = d.levels(H)
        ks = []
        for key, k, e in obs:
            wv, v, idx = L[key]
            ks.append(k if ref is None else int(np.argmax(abs(ref[(key, k)] @ v))))
        assert len(set(zip([o[0] for o in obs], ks))) == len(obs), 'state assignment collision'
        calc = np.array([L[key][0][kk] for (key, k, e), kk in zip(obs, ks)])
        J = np.array([[L[key][1][:, kk] @ d.ops[n][np.ix_(L[key][2], L[key][2])] @ L[key][1][:, kk] for n in names]
                      for (key, k, e), kk in zip(obs, ks)])
        dp = np.linalg.lstsq(J * sw[:, None], (E - calc) * sw, rcond=None)[0]
        p += dp
        ref = {(key, k): L[key][1][:, kk] for (key, k, e), kk in zip(obs, ks)}
        if abs(dp).max() < 1e-8:
            conv = True
            break
    r = E - calc
    dof = len(E) - len(names)
    s2 = float(r @ (w * r) / dof)
    Jw = J * sw[:, None]
    cov = np.linalg.inv(Jw.T @ Jw) * s2
    sv = np.linalg.svd(Jw / np.linalg.norm(Jw, axis=0), compute_uv=False)
    in_order = all(k == kk for (key, k, e), kk in zip(obs, ks)) and Q12._order_ok(dict(resid=r.tolist()), obs)
    return dict(p=dict(zip(names, map(float, p))), err=dict(zip(names, map(float, np.sqrt(np.diag(cov))))),
                mean=float(np.sqrt(r @ r / dof)), wmean=float(np.sqrt(s2)), rms=float(np.sqrt(r @ r / len(E))),
                dof=dof, resid=r.tolist(), cond=float(sv[0] / sv[-1]), converged=conv, order_ok=bool(in_order),
                chi2=float(r @ r))


def tr_fit(d, names, fixed, obs, x0, w, ref):
    """Trust-region least squares on the J-ordered levels (k-th eigenvalue of J to the k-th NIST level of J),
    analytic Jacobian; ref = levels for the eigenvector overlap check."""
    from scipy.optimize import least_squares
    w = np.ones(len(obs)) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    E = np.array([e for *_, e in obs])
    Hf = sum((a * d.ops[n] for n, a in fixed.items()), np.zeros_like(d.ops['A']))
    last = {}

    def lev(x):
        key = tuple(x)
        if last.get('k') != key:
            last.update(k=key, L=d.levels(Hf + sum(a * d.ops[n] for a, n in zip(x, names))))
        return last['L']

    def res(x):
        L = lev(x)
        return sw * (np.array([L[j][0][k] for j, k, _ in obs]) - E)

    def jac(x):
        L = lev(x)
        return sw[:, None] * np.array([[L[j][1][:, k] @ d.ops[n][np.ix_(L[j][2], L[j][2])] @ L[j][1][:, k] for n in names]
                                       for j, k, _ in obs])
    s = least_squares(res, [x0[n] for n in names], jac=jac, method='trf', x_scale='jac', xtol=1e-12, ftol=1e-12, gtol=1e-10,
                      max_nfev=2000)
    L = lev(s.x)
    r = E - np.array([L[j][0][k] for j, k, _ in obs])
    dof = len(E) - len(names)
    J = jac(s.x)
    cn = np.linalg.norm(J, axis=0)
    sv = np.linalg.svd(J / cn, compute_uv=False)
    s2 = float(r @ (w * r) / dof)
    cov = np.linalg.pinv(J.T @ J, rcond=1e-14) * s2
    mo = 1.0
    for j, k, _ in obs:
        mo = min(mo, float(np.max(np.abs(ref[j][1].T @ L[j][1][:, k]))))
    return dict(p=dict(zip(names, map(float, s.x))), err=dict(zip(names, map(float, np.sqrt(np.diag(cov))))),
                mean=float(np.sqrt(r @ r / dof)), wmean=float(np.sqrt(s2)), rms=float(np.sqrt(r @ r / len(E))), dof=dof,
                resid=r.tolist(), sv=sv.tolist(), cond=float(sv[0] / sv[-1]), status=int(s.status), nfev=int(s.nfev),
                min_overlap=mo, chi2=float(r @ r))


def show(tag, f, A, keys=FS):
    print(f"{tag}: mean {f['mean']:.2f} rms {f['rms']:.2f} dof {f['dof']} cond {f['cond']:.0f} status {f['status']} nfev {f['nfev']} "
          f"min overlap {f['min_overlap']:.2f} " + ' '.join(f"{q} {f['p'][q]:.3f}({f['err'][q]:.3f})" for q in ('T1', 'T2', 'T3', 'T4') if q in f['p']))
    print('   f(norm) ' + ' '.join(f"{k} {f['p'][k] * np.sqrt(A[k]):+.0f}({f['err'][k] * np.sqrt(A[k]):.0f})" for k in keys))
    if len(f['sv']) > 10:
        print('   sv ' + ' '.join(f'{x:.2g}' for x in f['sv']))


def term_effect(d, sets, XH):
    """Spin-free term energies: H(ours) - H(theirs) with the f difference only, then the A, B, C, alpha, beta, T1, T2
    part removed by least squares over all 16 + term energies (unit weights)."""
    out = {}
    for key in ('fam', 'fe_own'):
        Dl = sum((sets[key][k] - XH[k]) * d.ops[k] for k in FS)
        base = sum(p * d.ops[k] for k, p in dict(A=0, B=1050.0, C=3800.0, alpha=90.0, beta=-130.0, T1=9.2, T2=4.1).items())
        t0, t1 = d.terms(base), d.terms(base + 1e-3 * Dl)
        rows = {}
        for tk, (w, v, idx) in t0.items():
            g = 1e3 * (t1[tk][0] - w)
            rows[f'{tk[0]}{"SPDFGHIKL"[tk[1]]}'] = [float(x) for x in g]
        out[key] = rows
        print('f difference on the terms (first order, cm-1):', key, {a: [round(x) for x in b] for a, b in rows.items()})
    return out


def fits():
    from scipy.stats import f as fdist
    sets, T = ours()
    A = jlA()
    obs, N = Q.nist_obs('fe')
    d = Q.ds(N)
    p0 = Q.to_bc(Q.bfit_free('fe')['p'])
    x0 = [p0[n] for n in Q.NM1B]
    res = {}
    f0 = wlsq(d, Q.NM1B, {}, obs, x0)
    res['free_1b'] = f0
    print(f"(1b) mean {f0['mean']:.2f} T1 {f0['p']['T1']:.3f} +- {f0['err']['T1']:.3f} T2 {f0['p']['T2']:.3f} +- {f0['err']['T2']:.3f}")
    # T only (Trees): T1 free, T2 = 0
    nm1a = [n for n in Q.NM1B if n != 'T2']
    fa = wlsq(d, nm1a, {}, obs, [p0[n] for n in nm1a])
    res['T_only'] = fa
    print(f"T only: mean {fa['mean']:.2f} T1 {fa['p']['T1']:.3f} +- {fa['err']['T1']:.3f} = T {fa['p']['T1'] / MEH_PER_P:.3f} +- {fa['err']['T1'] / MEH_PER_P:.3f}")
    res['T_table'] = {'fe': dict(Tonly=[fa['p']['T1'] / MEH_PER_P, fa['err']['T1'] / MEH_PER_P, fa['mean']],
                                 b1=[f0['p']['T1'] / MEH_PER_P, f0['err']['T1'] / MEH_PER_P, f0['mean'], f0['p']['T2'], f0['err']['T2']])}
    for ion in ('v', 'cr', 'mn'):
        ob3, N3 = Q.nist_obs(ion)
        d3 = Q.ds(N3)
        p3 = Q.to_bc(Q.bfit_free(ion)['p'])
        fb = wlsq(d3, Q.NM1B, {}, ob3, [p3[n] for n in Q.NM1B])
        fo = wlsq(d3, nm1a, {}, ob3, [p3[n] for n in nm1a])
        res['T_table'][ion] = dict(Tonly=[fo['p']['T1'] / MEH_PER_P, fo['err']['T1'] / MEH_PER_P, fo['mean']],
                                   b1=[fb['p']['T1'] / MEH_PER_P, fb['err']['T1'] / MEH_PER_P, fb['mean'], fb['p']['T2'], fb['err']['T2']])
    print('T table', res['T_table'])

    def Xh(h):
        return {k: h[k] / np.sqrt(A[k]) for k in FS}

    def score(X):
        f = wlsq(d, Q.NM1B, X, obs, x0)
        return f

    hv = {k: float(H_FE[k][0]) for k in FS}
    fH = score(Xh(hv))
    res['H_fixed'] = fH
    print(f"their f fixed: mean {fH['mean']:.2f} (order {fH['order_ok']}) T1 {fH['p']['T1']:.3f} T2 {fH['p']['T2']:.3f}")
    fam = score({k: sets['fam'][k] for k in FS})
    res['fam_f_fixed'] = fam
    own = score({k: sets['fe_own'][k] for k in FS})
    res['own_f_fixed'] = own
    print(f"our f (fam) fixed: mean {fam['mean']:.2f} T1 {fam['p']['T1']:.3f} T2 {fam['p']['T2']:.3f}; fe_own {own['mean']:.2f}")
    # with their T3, T4 too (both sign hypotheses): p_t <-> T4, p_t' <-> T3
    pt, ptp = H_P['fe'][2][0] * MEH_PER_P, H_P['fe'][3][0] * MEH_PER_P
    res['H_fixed_T34'] = {}
    for sa, sb in itertools.product((1, -1), repeat=2):
        f = score(dict(Xh(hv), T4=sa * pt, T3=sb * ptp))
        res['H_fixed_T34'][f'{sa:+d}{sb:+d}'] = f['mean']
    print('their f + p_t, p_t\' (signs):', res['H_fixed_T34'])
    # lambda on their vector (9 free), against their vector fixed
    d.ops['Xab'] = sum(c * d.ops[k] for k, c in Xh(hv).items())
    fl = wlsq(d, Q.NM1B + ['Xab'], {}, obs, x0 + [1.0])
    res['H_lam'] = dict(lam=fl['p']['Xab'], err=fl['err']['Xab'], mean=fl['mean'])
    del d.ops['Xab']
    d.ops['Xab'] = sum(sets['fam'][k] * d.ops[k] for k in FS)
    fl2 = wlsq(d, Q.NM1B + ['Xab'], {}, obs, x0 + [1.0])
    res['fam_lam'] = dict(lam=fl2['p']['Xab'], err=fl2['err']['Xab'], mean=fl2['mean'])
    del d.ops['Xab']
    print('lam: H', res['H_lam'], 'fam f', res['fam_lam'])
    # both directions: is the data's preference for theirs over ours decided by one operator? ours with one replaced
    res['fam_replace_one'] = {}
    for k in FS:
        X = {q: sets['fam'][q] for q in FS}
        X[k] = hv[k] / np.sqrt(A[k])
        res['fam_replace_one'][k] = score(X)['mean']
    print('ours with one f replaced by theirs:', {k: round(v, 2) for k, v in res['fam_replace_one'].items()})
    # ordering: every pair swap and every sign flip of their (normalised) values
    sw = {}
    for a, b in itertools.combinations(FS, 2):
        h = dict(hv)
        h[a], h[b] = h[b], h[a]
        try:
            f = score(Xh(h))
            sw[f'{a}<->{b}'] = dict(mean=f['mean'], order_ok=f['order_ok'])
        except (AssertionError, np.linalg.LinAlgError) as ex:
            sw[f'{a}<->{b}'] = dict(mean=float('nan'), order_ok=False, err=str(ex))
    fl_ = {}
    for a in FS:
        h = dict(hv)
        h[a] = -h[a]
        try:
            f = score(Xh(h))
            fl_[a] = dict(mean=f['mean'], order_ok=f['order_ok'])
        except (AssertionError, np.linalg.LinAlgError) as ex:
            fl_[a] = dict(mean=float('nan'), order_ok=False, err=str(ex))
    res['swaps'], res['flips'] = sw, fl_
    top = sorted(((v['mean'], k, v['order_ok']) for k, v in sw.items() if np.isfinite(v['mean'])))[:8]
    print('best swaps', [(round(m, 2), k, o) for m, k, o in top])
    print('flips', {k: round(v['mean'], 2) for k, v in fl_.items()})
    # all free: trust region on the J-ordered levels (Gauss-Newton diverges, see report), several starts
    nm = Q.NM1B + ['T3', 'T4'] + FS
    W = np.array([j2 + 1 for j2, k, e in obs], float)
    pF, pH = fam['p'], fH['p']
    start = {'ours': {**pF, **{k: sets['fam'][k] for k in FS}, 'T3': sets['fam']['T3'], 'T4': sets['fam']['T4']},
             'theirs': {**pH, **Xh(hv), 'T4': pt, 'T3': ptp}}
    ref = d.levels(sum(v * d.ops[k] for k, v in start['theirs'].items()))
    res['allfree'] = {}
    for wn, ww in (('uniform', None), ('2J+1', W)):
        for st, xs in start.items():
            f = tr_fit(d, nm, {}, obs, xs, ww, ref)
            res['allfree'][f'{wn}|{st}'] = f
            show(f'all free {wn} start {st}', f, A)
    rng = np.random.default_rng(20261105)
    res['allfree_multistart'] = []
    for i in range(8):
        xs = {k: start['ours'][k] for k in nm}
        for k in FS:                                   # scatter the f by 100 cm-1 in their normalisation
            xs[k] = start['ours'][k] + rng.normal(0, 100) / np.sqrt(A[k])
        f = tr_fit(d, nm, {}, obs, xs, None, ref)
        res['allfree_multistart'].append(dict(mean=f['mean'], fnorm={k: f['p'][k] * np.sqrt(A[k]) for k in FS},
                                              T=[f['p'][k] for k in ('T1', 'T2', 'T3', 'T4')], minov=f['min_overlap']))
        print(f"  start {i}: mean {f['mean']:.3f} min overlap {f['min_overlap']:.2f} f(norm) " + ' '.join(f"{f['p'][k] * np.sqrt(A[k]):+.0f}" for k in FS))
    # batches in the spirit of the paper: only the large f free, the others held
    LARGE = ['f1', 'f3', 'f4', 'f5', 'f8', 'f12']
    NOT1 = ['f3', 'f4', 'f5', 'f8', 'f12']
    res['batch'] = {}
    for key, free, held, x0b in (
            ('large free, rest 0', LARGE, {}, start['theirs']),
            ('large free, rest ours (fam)', LARGE, {k: sets['fam'][k] for k in Q12.EXTRA if k not in LARGE}, start['ours']),
            ('large free, rest theirs', LARGE, {k: v for k, v in {**Xh(hv), 'T4': pt, 'T3': ptp}.items() if k not in LARGE}, start['theirs']),
            ('no f1: large free, rest + f1 theirs', NOT1, {k: v for k, v in {**Xh(hv), 'T4': pt, 'T3': ptp}.items() if k not in NOT1}, start['theirs'])):
        names = Q.NM1B + free
        f = tr_fit(d, names, held, obs, {k: x0b[k] for k in names}, None, ref)
        res['batch'][key] = f
        show(key, f, A, free)
    # the difference ours - theirs seen on the terms: spin-free term energies with each f vector, two-body part refitted
    res['term_effect'] = term_effect(d, sets, Xh(hv))
    # the unobserved level
    res['b1S'] = {}
    for key, ff in (('their f', fH), ('fam f', fam), ('fe_own f', own)):
        X = {'their f': Xh(hv), 'fam f': {k: sets['fam'][k] for k in FS}, 'fe_own f': {k: sets['fe_own'][k] for k in FS}}[key]
        L = d.levels(sum(v * d.ops[k] for k, v in {**ff['p'], **X}.items()))
        w0 = L[0][0]
        res['b1S'][key] = [float(x - L[8][0][0]) for x in w0]
        print('J=0 levels above 5D4', key, [round(x, 0) for x in res['b1S'][key]])
    dl = None
    fa2 = res['allfree'].get('uniform|ours')
    tests = {}
    dl, cl = fa2['dof'], fa2['chi2']
    for key, fx in (('(1b)', f0), ('their f fixed, T3 = T4 = 0', fH), ('fam f fixed, T3 = T4 = 0', fam), ('fe_own f fixed, T3 = T4 = 0', own)):
        F = (fx['chi2'] - cl) / 14 / (cl / dl)
        tests[key] = dict(F=float(F), p=float(fdist.sf(F, 14, dl)))
    res['F_vs_allfree'] = tests
    print('F tests vs all free', tests)
    # d3 ions: T3, T4 free (10 free, 19 levels)
    res['d3_free'] = {}
    for ion in ('v', 'cr', 'mn'):
        ob3, N3 = Q.nist_obs(ion)
        d3 = Q.ds(N3)
        p3 = Q.to_bc(Q.bfit_free(ion)['p'])
        nm3 = Q.NM1B + ['T3', 'T4']
        for wn, ww in (('uniform', None), ('2J+1', np.array([j2 + 1 for j2, k, e in ob3], float))):
            f = wlsq(d3, nm3, {}, ob3, [p3.get(n, 0.0) for n in nm3], w=ww)
            res['d3_free'][f'{ion}|{wn}'] = f
            print(f"{ion} {wn}: T3,T4 free: mean {f['mean']:.2f} T1 {f['p']['T1']:.3f}({f['err']['T1']:.3f}) T2 {f['p']['T2']:.3f}({f['err']['T2']:.3f}) "
                  f"T3 {f['p']['T3']:.3f}({f['err']['T3']:.3f}) T4 {f['p']['T4']:.3f}({f['err']['T4']:.3f}) order {f['order_ok']}")
    return res


def extra():
    """2J+1-weighted T-only and (1b) (their metric); level-space distance of the two f vectors; swap ranking."""
    sets, T = ours()
    A = jlA()
    res = dict(w2J={})
    nm1a = [n for n in Q.NM1B if n != 'T2']
    for ion in LAB:
        ob, N = Q.nist_obs(ion)
        d = Q.ds(N)
        p = Q.to_bc(Q.bfit_free(ion)['p'])
        W = np.array([j2 + 1 for j2, k, e in ob], float)
        fo = wlsq(d, nm1a, {}, ob, [p[n] for n in nm1a], w=W)
        fb = wlsq(d, Q.NM1B, {}, ob, [p[n] for n in Q.NM1B], w=W)
        res['w2J'][ion] = dict(Tonly=fo['p']['T1'] / MEH_PER_P, Tonly_err=fo['err']['T1'] / MEH_PER_P,
                               b1=fb['p']['T1'] / MEH_PER_P, b1_err=fb['err']['T1'] / MEH_PER_P,
                               b1T2=fb['p']['T2'], b1T2_err=fb['err']['T2'], rms=fb['rms'], Tonly_rms=fo['rms'])
        print(ion, {k: round(v, 3) for k, v in res['w2J'][ion].items()})
    ob, N = Q.nist_obs('fe')
    d = Q.ds(N)
    p = Q.to_bc(Q.bfit_free('fe')['p'])
    x0 = [p[n] for n in Q.NM1B]
    W = np.array([j2 + 1 for j2, k, e in ob], float)
    XH = {k: H_FE[k][0] / np.sqrt(A[k]) for k in FS}
    fH = wlsq(d, Q.NM1B, XH, ob, x0, w=W)
    fO = wlsq(d, Q.NM1B, {k: sets['fam'][k] for k in FS}, ob, x0, w=W)
    res['w2J']['fe_H_fixed'] = dict(T1=fH['p']['T1'], T1_err=fH['err']['T1'], T2=fH['p']['T2'], T2_err=fH['err']['T2'], rms=fH['rms'], mean=fH['mean'])
    res['w2J']['fe_fam_fixed'] = dict(T1=fO['p']['T1'], T1_err=fO['err']['T1'], T2=fO['p']['T2'], T2_err=fO['err']['T2'], rms=fO['rms'], mean=fO['mean'])
    print('2J+1: their f', res['w2J']['fe_H_fixed'], 'fam f', res['w2J']['fe_fam_fixed'])
    u = json.load(open(JS))['fits']
    rH, rO = np.array(u['H_fixed']['resid']), np.array(u['fam_f_fixed']['resid'])
    rF = np.array(u['free_1b']['resid'])
    res['level_dist'] = dict(rms_diff_H_fam=float(np.sqrt(np.mean((rH - rO) ** 2))), max_diff=float(abs(rH - rO).max()),
                             rms_H=float(np.sqrt(np.mean(rH ** 2))), rms_fam=float(np.sqrt(np.mean(rO ** 2))),
                             corr=float(rH @ rO / np.linalg.norm(rH) / np.linalg.norm(rO)),
                             rms_free=float(np.sqrt(np.mean(rF ** 2))))
    print('levels', res['level_dist'])
    for (j2, k, e), a, b in zip(ob, rH, rO):
        if abs(a - b) > 8:
            print('  level 2J', j2, 'k', k, 'E', e, 'r_H %.1f r_fam %.1f' % (a, b))
    sw = sorted(u['swaps'].items(), key=lambda x: x[1]['mean'])
    print('swaps', [(k, round(v['mean'], 2)) for k, v in sw])
    big = {'f1', 'f3', 'f4', 'f5', 'f8', 'f12'}
    print('swaps with a large one, except f3<->f8:', [(k, round(v['mean'], 2)) for k, v in sw if set(k.split('<->')) & big and k != 'f3<->f8'][:10])
    return res


def so5_generators(N):
    """The 10 SO(5) generators on the d^N determinants: real antisymmetric rotations of the five real d orbitals, spin-free."""
    i0 = lambda m: m + 2                                      # noqa: E731
    U = np.zeros((5, 5), complex)                             # columns: real d orbitals in the m basis (Condon-Shortley)
    U[i0(0), 0] = 1
    for n, m in enumerate((1, 2)):
        U[i0(-m), 1 + 2 * n], U[i0(m), 1 + 2 * n] = 1 / np.sqrt(2), (-1) ** m / np.sqrt(2)
        U[i0(-m), 2 + 2 * n], U[i0(m), 2 + 2 * n] = 1j / np.sqrt(2), -1j * (-1) ** m / np.sqrt(2)
    SO = cfp_l.spin_orbitals(2)
    G = []
    for a, b in itertools.combinations(range(5), 2):
        A = np.zeros((5, 5))
        A[a, b], A[b, a] = 1, -1
        g5 = U @ A @ U.conj().T
        g = np.array([[g5[i0(m), i0(m2)] if s == s2 else 0 for m2, s2 in SO] for m, s in SO])
        G.append(S._one_body(N, g.real) + 1j * S._one_body(N, g.imag))
    return G


def hj():
    """Definitions of o_t, o_t' in Hansen & Judd 1985 against T3, T4. Eigenvalue of -sum [G, [G, O]] for an operator in
    (w1 w2) of SO(5) is w1(w1 + 3) + w2(w2 + 1): (00) 0, (22) 16, (30) 18, (42) 34."""
    ax = dict(zip(['T1', 'T2', 'T3', 'T4'], Q.o_axes3()))
    jl = {k: np.array(v, float) for k, v in S.jl_table(3).items() if k in ('t1', 't2', 't3', 't4')}
    lab = [S.comp_label(3, c) for c, _ in S.components(3)]
    res = {}
    # eq. (7): o_t'(2P) = o_t'(2H) = 2, o_t'(2D, v = 3) = 4, o_t'(2F) = o_t'(2G) = -3, zero everywhere else in d3
    e7 = {'3:2P': 2, '3:2H': 2, '3:2D': 4, '3:2F': -3, '3:2G': -3}
    v7 = np.array([e7.get(k, 0) for k in lab], float)
    n7 = S.ip(3, v7, v7)
    res['eq7'] = dict(norm2=n7, table4_1988=560.0, max_diff_t3_over_3=float(abs(v7 - jl['t3'] / 3).max()),
                      cos_T3=S.ip(3, v7, ax['T3']) / np.sqrt(n7), max_diff_T3=float(abs(v7 / np.sqrt(n7) - ax['T3']).max()))
    print('eq. (7)', res['eq7'])
    # SO(5) labels
    d3 = Q.ds(3)
    G = so5_generators(3)
    C = -sum(g @ g for g in G)
    res['state_casimir'] = sorted({float(x) for x in np.round(np.linalg.eigvalsh(0.5 * (C + C.conj().T)), 8)})
    print('state Casimir (expect (10) 4, (11) 6, (21) 12):', res['state_casimir'])

    def casimir(O):
        O = O - np.trace(O) / len(O) * np.eye(len(O))
        X = -sum(g @ (g @ O - O @ g) - (g @ O - O @ g) @ g for g in G)
        n = np.sum(abs(O) ** 2)
        lam = float((np.sum(O.conj() * X) / n).real)
        return dict(eig=lam, residual=float(np.sqrt(np.sum(abs(X - lam * O) ** 2) / n)))
    up = lambda v: S.lift(S.from_vec(3, v), 3, 3)              # noqa: E731
    ops = {k: d3.ops[k] for k in ('C', 'beta', 'T1', 'T2', 'T3', 'T4')}
    ops.update({'jl ' + k: up(v) for k, v in jl.items()})
    ops['jl t1-t2'] = up(jl['t1'] - jl['t2'])
    # control: B without its (00) part (identity, C, beta) is Racah's (22), the F2 - 5 F4 of Hansen & Judd section 6
    Vb = [S.to_vec(3, np.eye(len(C))), S.to_vec(3, d3.ops['C']), S.to_vec(3, d3.ops['beta'])]
    Gm = np.array([[S.ip(3, a, b) for b in Vb] for a in Vb])
    vB = S.to_vec(3, d3.ops['B'])
    cB = np.linalg.solve(Gm, [S.ip(3, a, vB) for a in Vb])
    ops['B minus (00)'] = S.from_vec(3, vB - sum(c * a for c, a in zip(cB, Vb)))
    res['op_casimir'] = {k: casimir(O) for k, O in ops.items()}
    for k, v in res['op_casimir'].items():
        print(f"{k:14s} eig {v['eig']:9.5f} residual {v['residual']:.1e}")
    # the (22) part of the pure three-body space is span{t1, t2}; the line in it orthogonal to u = 3t1 + 2t2 is o_t
    u, ot = 3 * jl['t1'] + 2 * jl['t2'], jl['t1'] - jl['t2']
    res['o_t'] = dict(ip_u=S.ip(3, u, ot), norm2=S.ip(3, ot, ot), table4_1988=18900.0,
                      cos_T4=S.ip(3, ot, ax['T4']) / np.sqrt(S.ip(3, ot, ot)))
    print('o_t = t1 - t2', res['o_t'])
    # Table 6, Cr IV 3d3 (unnormalised 1985 operators), in the 1988 normalisation and in mEh on our axes
    nrm = {"p3''": 162750.0, "p3'''": 146475 / 11, 'p_t': 18900.0}
    t6 = {"p3''": (-6.291, .018), "p3'''": (2.923, .084), 'p_t': (.120, .048)}
    res['table6_cr'] = {}
    for k, (x, e) in t6.items():
        sc = np.sqrt(nrm[k] / NORM3)
        res['table6_cr'][k] = dict(p1985=x, err1985=e, p_norm=x * sc, err_norm=e * sc, mEh=x * sc * MEH_PER_P, err_mEh=e * sc * MEH_PER_P)
        print(k, res['table6_cr'][k])
    return res


def j82():
    """o_t = U - V (Hansen & Judd 1985) from the matrix elements of Judd, Hansen & Raassen 1982, Table 1. The diagonal
    elements do not depend on the 2D phase; the off-diagonal ones test that the phase convention is the same."""
    r = np.sqrt(189.0)
    #        4P  4F     2P     1:2D 3:2D 1:2D|3:2D  2F     2G       2H
    rows = ['3:4P', '3:4F', '3:2P', '1:2D', '3:2D', '1:2D|3:2D', '3:2F', '3:2G', '3:2H']
    tab = {'u': [7, -3, -11, 0, 0, -r, 9, -7, 3], 'v': [21, -9, 3, 0, -6, 0, -12, 8, 3], 'w': [0, 0, -198, 0, 132, 0, -33, 55, -30],
           "o3''": [5 / 6 * x for x in (21, -9, -141, 0, 84, -r, -21, 35, -15)],             # printed as 6 o3''/5
           "o3'''": [21, -9, -3 / 2, 0, -9, -r, 9 / 4, -15 / 4, 135 / 22]}
    lab = [S.comp_label(3, c) for c, _ in S.components(3)]
    vec = {k: np.array([dict(zip(rows, v))[q] for q in lab], float) for k, v in tab.items()}
    jl = {k: np.array(v, float) for k, v in S.jl_table(3).items() if k in ('t1', 't2', 't3', 't4')}
    ax = dict(zip(['T1', 'T2', 'T3', 'T4'], Q.o_axes3()))
    u, v, w = vec['u'], vec['v'], vec['w']
    res = dict(max_diff=dict(u_t1=float(abs(u - jl['t1']).max()), v_t2=float(abs(v - jl['t2']).max()), w_t4=float(abs(w - jl['t4']).max()),
                             o3pp_resolution=float(abs(vec["o3''"] - (5 / 6 * u + 5 / 9 * (v + w))).max()),
                             o3ppp_resolution=float(abs(vec["o3'''"] - (u + 2 * v / 3 - 5 * w / 132)).max())))
    ot = u - v
    cand = {'o_t = u - v': (ot, 'T4'), "o3''": (vec["o3''"], 'T1'), "o3'''": (vec["o3'''"], 'T2')}
    res['ops'] = {}
    for k, (x, a) in cand.items():
        n2 = S.ip(3, x, x)
        res['ops'][k] = dict(norm2=n2, axis=a, cos=S.ip(3, x, ax[a]) / np.sqrt(n2), elements=dict(zip(lab, map(float, x))))
    res['ops']['o_t = u - v']['diag_max_diff_t1_t2'] = float(abs(ot - (jl['t1'] - jl['t2'])).max())
    print(json.dumps(res, indent=1, default=float))
    return res


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    res = json.load(open(JS)) if os.path.exists(JS) else {}
    for k, fn in (('ident', ident), ('compare', compare), ('fits', fits), ('extra', extra), ('hj', hj), ('j82', j82)):
        if cmd in (k, 'all'):
            res[k] = fn()
            json.dump(res, open(JS, 'w'), indent=1, default=float)      # extra reads the stored fits


if __name__ == '__main__':
    main()
