#!/usr/bin/env python3
"""Phase 17 item 7 (1 Nov brief): the operators outside the (1b) model, fixed at our ab initio values, in the NIST fit.

Operators (phase17q11.DS, one determinant basis): inner product <X, Y> = Tr(X Y) over all states of d^N (dshell.ip).
  T1, T2   Uylings' (1984) O2', O2'' (phase15c, built algebraically), pure three-body, Tr O^2 = 1 in d3, coefficient mEh
  T3, T4   d3 only: the rest of the pure three-body space, Gram-Schmidt of the Judd-Leavitt t3, then t1 against T1, T2
           (and T3), Tr O^2 = 1 in d3, coefficient mEh (own completion: only span{T3, T4} is unique)
  f1..f12  Judd & Leavitt (1986) four-body operators, pure four-body; the unnormalised Table 7 d4 matrices (Tr f^2 in
           d4 = A(f_i) of their Table 9), coefficient cm-1
  in d6 all of these are the d3 / d4 operators lifted as pure n-body operators (so T* keep their d3 meaning).
Values: d3: proj chain (phase17q11.chain('proj') extended to T3, T4: base + hole + carried items + DK3, absolute);
        Fe2+: largest-space projection (fend fam_d5f2g2h1); variant 'fe_own': + the Fe own increments (n = 3, all s,
        hole, i, DK3; raw).
Fits on the NIST level set of each ion (same J, same order as (1b)); mean error = sqrt(sum r^2 / (n - free)):
  free     the (1b) fit (A B C zeta alpha beta T1 T2 free)
  fixX     + the extra operators fixed at our values
  lam      + lambda * (sum of the extra operators at our values), lambda free (the nesting model for the F tests)
  fixX12   fixX with T1, T2 also fixed at our proj Final values (the complete spin-free set fixed; A B C zeta alpha
           beta free)
  split    (item 8, Fe2+) only T3, T4 / only f1..f12 / both fixed at the fam values
  cheap    (item 8, Fe2+) the extras from the cheap spaces (k1fh, k1fh_n3, k1fh_sall, u3c1) and their combinations
           with the item-2 increments (+ i, DK3)
Commands: defs, fit, split, cheap"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase17q11 as Q  # noqa: E402
from atomci_heff import dshell as S  # noqa: E402

OUT = Q.OUT
ION = {'v': 'V²⁺', 'cr': 'Cr³⁺', 'mn': 'Mn⁴⁺', 'fe': 'Fe²⁺'}
EXTRA = ['T3', 'T4'] + [f'f{i}' for i in range(1, 13)]
FE_OWN = [(('d6', 'fend', 'k1fh_n3', 'fe'), ('d6', 'fend', 'k1fh', 'fe')),
          (('d6', 'fend', 'k1fh_sall', 'fe'), ('d6', 'fend', 'k1fh', 'fe')),
          (('d6', 'fend', 'u3c1', 'fe'), ('d6', 'fend', 'k1fh_n3', 'fe')),
          (('d6', 'feti5', 'k1fhi', 'fe'), ('d6', 'fend', 'k1fh', 'fe')),
          (('d6', 'fedk', 'rasK1r', 'fe'), ('d6', 'feop', 'rasK1r', 'fe'))]


def values():
    """{ion: dict(proj base, corrected, T1/T2 Final (proj))} for all operators of the complete set."""
    Q.Q = ('T1', 'T2', 'B', 'C', 'beta', 'T3', 'T4')          # the proj chain carries T3, T4 like T2 (absolute)
    ch = Q.chain('proj')
    out = {}
    for ion, lab in ION.items():
        b = Q.rep(Q.SRC['base'][lab])['proj']
        r = ch['ions'][lab]
        ex = [k for k in EXTRA if k in b]
        v = dict(base={k: b[k] for k in ['T1', 'T2'] + ex}, final=dict(T1=r['T1'], T2=r['T2']))
        if ion == 'fe':
            v['corr'] = {k: b[k] for k in ex}
            v['fe_own'] = {k: b[k] + sum(Q.rep(a)['proj'][k] - Q.rep(c)['proj'][k] for a, c in FE_OWN) for k in ex}
        else:
            v['corr'] = {k: r[k] for k in ex}
        out[ion] = v
    return out


def defs():
    """Norms and mutual overlaps of the operators as used (d3: T1..T4; d4/d6: f), and the values per ion."""
    res = {}
    for N in (3, 4, 6):
        d = Q.ds(N) if N != 4 else None
        if N == 4:
            V = {k: np.array(v, float) for k, v in S.jl_table(4).items() if k.startswith('f')}
            nm = {k: S.ip(4, v, v) for k, v in V.items()}
            res['d4_f_norm2'] = nm
            res['d4_f_norm2_vs_JL_table9'] = {k: nm[k] / S.jl_norms()[k][4] for k in nm}
            continue
        names = Q.MODEL + d.extra
        vec = {k: d.vec(d.ops[k]) for k in names}
        nr = {k: np.sqrt(S.ip(N, v, v)) for k, v in vec.items()}
        G = {a: {b: S.ip(N, vec[a], vec[b]) / nr[a] / nr[b] for b in names} for a in names}
        res[f'd{N}'] = dict(norm={k: float(nr[k]) for k in names},
                            max_cos_T_vs_model=float(max(abs(G[a][b]) for a in d.extra for b in Q.MODEL)),
                            max_cos_within_T=float(max(abs(G[a][b]) for a in ('T1', 'T2', 'T3', 'T4')
                                                       for b in ('T1', 'T2', 'T3', 'T4') if a < b)))
    vals = values()
    res['values'] = vals
    json.dump(res, open(os.path.join(OUT, 'phase17q12_defs.json'), 'w'), indent=1, default=float)
    print('d3 norms (cm-1 per unit; T in mEh -> x219.47)', {k: round(v, 3) for k, v in res['d3']['norm'].items()})
    print('d3 max |cos| T vs model', res['d3']['max_cos_T_vs_model'], 'within T', res['d3']['max_cos_within_T'])
    print('d4 Tr f^2 / A(f) (JL Table 9)', {k: round(v, 6) for k, v in res['d4_f_norm2_vs_JL_table9'].items()})
    print('d6 norms', {k: round(v, 2) for k, v in res['d6']['norm'].items()})
    for ion, v in vals.items():
        print(ion, 'base', {k: round(x, 4) for k, x in v['base'].items()})
        print('   corr', {k: round(x, 4) for k, x in v['corr'].items()}, 'final', v['final'])
        if 'fe_own' in v:
            print('   fe_own', {k: round(x, 4) for k, x in v['fe_own'].items()})


def _order_ok(f, obs):
    calc = {}
    for (j2, k, e), r in zip(obs, f['resid']):
        calc.setdefault(j2, []).append((k, e - r))
    return all([k for k, _ in sorted(v, key=lambda x: x[1])] == sorted(k for k, _ in v) for v in calc.values())


def fit():
    from scipy.stats import f as fdist
    vals = values()
    res = {}
    for ion, lab in ION.items():
        obs, N = Q.nist_obs(ion)
        d = Q.ds(N)
        p0 = Q.to_bc(Q.bfit_free(ion)['p'])
        n = len(obs)
        out = {}

        def run(key, names, fixed):
            f = Q.fit(d, obs, d.levels, names, fixed, [p0.get(q, 1.0) for q in names])
            chi = float(np.sum(np.square(f['resid'])))
            out[key] = dict(p={q: float(x) for q, x in f['p'].items()}, err={q: float(x) for q, x in f['err'].items()},
                            mean_error=f['rms'], rms=float(np.sqrt(chi / n)), chi2=chi, free=len(names),
                            order_ok=_order_ok(f, obs))
            return out[key]

        f0 = run('free', Q.NM1B, {})
        sets = {'corr': vals[ion]['corr'], 'base': {k: x for k, x in vals[ion]['base'].items() if k in EXTRA}}
        if ion == 'fe':
            sets['fe_own'] = vals[ion]['fe_own']
        for sk, X in sets.items():
            d.ops['Xab'] = sum(c * d.ops[k] for k, c in X.items())
            f1 = run(f'fixX:{sk}', Q.NM1B, dict(X))
            fl = run(f'lam:{sk}', Q.NM1B + ['Xab'], {})
            f2 = run(f'fixX12:{sk}', [q for q in Q.NM1B if q not in ('T1', 'T2')], dict(X, **vals[ion]['final']))
            f12 = run(f'fix12:{sk}', [q for q in Q.NM1B if q not in ('T1', 'T2')], dict(vals[ion]['final']))
            cl, dl = fl['chi2'], n - fl['free']
            F = lambda c, q: (c - cl) / q / (cl / dl)                    # noqa: E731
            t = dict(lam=fl['p']['Xab'], lam_err=fl['err']['Xab'],
                     F_lam0=F(f0['chi2'], 1), p_lam0=float(fdist.sf(F(f0['chi2'], 1), 1, dl)),
                     F_lam1=F(f1['chi2'], 1), p_lam1=float(fdist.sf(F(f1['chi2'], 1), 1, dl)),
                     F_full=F(f2['chi2'], 3), p_full=float(fdist.sf(F(f2['chi2'], 3), 3, dl)))
            c1, d1 = f1['chi2'], n - f1['free']
            t['F_12_given_X'] = (f2['chi2'] - c1) / 2 / (c1 / d1)
            t['p_12_given_X'] = float(fdist.sf(t['F_12_given_X'], 2, d1))
            c0, d0 = f0['chi2'], n - f0['free']
            t['F_12_noX'] = (f12['chi2'] - c0) / 2 / (c0 / d0)
            t['p_12_noX'] = float(fdist.sf(t['F_12_noX'], 2, d0))
            t['move'] = {q: f1['p'][q] - f0['p'][q] for q in Q.NM1B if q != 'A'}
            t['toward_proj'] = {q: abs(f1['p'][q] - vals[ion]['final'][q]) < abs(f0['p'][q] - vals[ion]['final'][q])
                                for q in ('T1', 'T2')}
            out[f'test:{sk}'] = t
            del d.ops['Xab']
            print(f"{lab} [{sk}] n {n}: mean error free {f0['mean_error']:.2f} -> fixX {f1['mean_error']:.2f} "
                  f"({100 * (f1['mean_error'] / f0['mean_error'] - 1):+.1f} %), lam {t['lam']:.2f} ± {t['lam_err']:.2f} "
                  f"(p lam=0 {t['p_lam0']:.3g}, lam=1 {t['p_lam1']:.3g}); T1 {f0['p']['T1']:.3f} -> {f1['p']['T1']:.3f}, "
                  f"T2 {f0['p']['T2']:.3f} -> {f1['p']['T2']:.3f} (proj {vals[ion]['final']['T1']:.3f} "
                  f"{vals[ion]['final']['T2']:.3f}); fixX12 mean error {f2['mean_error']:.2f} p_full {t['p_full']:.3g} "
                  f"p(T1T2|X) {t['p_12_given_X']:.3g} [no X: {f12['mean_error']:.2f} p {t['p_12_noX']:.3g}] "
                  f"order {f1['order_ok']} {f2['order_ok']}", flush=True)
        res[lab] = out
    json.dump(res, open(os.path.join(OUT, 'phase17q12_fit.json'), 'w'), indent=1)


def fe_tests(sets):
    """Fe2+ NIST fit with each extra-operator set fixed (fixX) and scaled (lam); {key: summary}."""
    from scipy.stats import f as fdist
    obs, N = Q.nist_obs('fe')
    d = Q.ds(N)
    p0 = Q.to_bc(Q.bfit_free('fe')['p'])
    n = len(obs)

    def run(names, fixed):
        f = Q.fit(d, obs, d.levels, names, fixed, [p0.get(q, 1.0) for q in names])
        return f, float(np.sum(np.square(f['resid'])))

    f0, c0 = run(Q.NM1B, {})
    out = {'free': dict(mean_error=f0['rms'], T1=f0['p']['T1'], T2=f0['p']['T2'])}
    for key, X in sets.items():
        d.ops['Xab'] = sum(c * d.ops[k] for k, c in X.items())
        f1, c1 = run(Q.NM1B, dict(X))
        fl, cl = run(Q.NM1B + ['Xab'], {})
        del d.ops['Xab']
        dl = n - len(Q.NM1B) - 1
        F1 = (c1 - cl) / (cl / dl)
        out[key] = dict(values=X, mean_error=f1['rms'], T1=f1['p']['T1'], T1_err=f1['err']['T1'], T2=f1['p']['T2'],
                        T2_err=f1['err']['T2'], lam=fl['p']['Xab'], lam_err=fl['err']['Xab'],
                        lam_mean_error=fl['rms'], p_lam1=float(fdist.sf(F1, 1, dl)),
                        order_ok=_order_ok(f1, obs), gain=(f0['rms'] - f1['rms']) / f0['rms'])
        r = out[key]
        print(f"{key:16s} mean error {f0['rms']:.2f} -> {r['mean_error']:.2f} ({-100 * r['gain']:+.1f} %), lam "
              f"{r['lam']:.3f} ± {r['lam_err']:.3f} (p lam=1 {r['p_lam1']:.3g}), T1 {r['T1']:.3f} ± {r['T1_err']:.3f}, "
              f"T2 {r['T2']:.3f} ± {r['T2_err']:.3f} order {r['order_ok']}", flush=True)
    return out


def split():
    X = values()['fe']['corr']
    T = {k: v for k, v in X.items() if k.startswith('T')}
    f = {k: v for k, v in X.items() if k.startswith('f')}
    sets = {'T3T4': T, 'f1-f12': f, 'both': X}
    for k in f:                                     # each f alone (which of the twelve)
        sets[f'only {k}'] = {k: f[k]}
    json.dump(fe_tests(sets), open(os.path.join(OUT, 'phase17q12_split.json'), 'w'), indent=1)


def cheap():
    sp = {k: Q.rep(('d6', 'fend', k, 'fe'))['proj'] for k in ('fam_d5f2g2h1', 'k1fh', 'k1fh_n3', 'k1fh_sall', 'u3c1')}
    ii = Q.rep(('d6', 'feti5', 'k1fhi', 'fe'))['proj']
    op, dk = Q.rep(('d6', 'feop', 'rasK1r', 'fe'))['proj'], Q.rep(('d6', 'fedk', 'rasK1r', 'fe'))['proj']
    sets = {}
    for k, v in sp.items():
        sets[k] = {q: v[q] for q in EXTRA}
    # k1fh + the item-2 increments: n = 3, all s, hole (u3c1 - k1fh_n3) = u3c1 + k1fh_sall - k1fh
    sets['k1fh+item2'] = {q: sp['u3c1'][q] + sp['k1fh_sall'][q] - sp['k1fh'][q] for q in EXTRA}
    sets['k1fh+item2+i+DK3'] = {q: sets['k1fh+item2'][q] + ii[q] - sp['k1fh'][q] + dk[q] - op[q] for q in EXTRA}
    res = fe_tests(sets)
    res['T1T2'] = {k: dict(T1=v['T1'], T2=v['T2']) for k, v in sp.items()}
    json.dump(res, open(os.path.join(OUT, 'phase17q12_cheap.json'), 'w'), indent=1)
    for k, v in sets.items():
        print(f'{k:18s}', ' '.join(f'{q} {x:+.3f}' for q, x in v.items()))


if __name__ == '__main__':
    {'defs': defs, 'fit': fit, 'split': split, 'cheap': cheap}[sys.argv[1]]()
