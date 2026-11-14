#!/usr/bin/env python3
"""Phase 15u B: measured-side J-level fits of the 3d^N ground configurations with the 15i operators, NIST ASD levels
(data/levels/NIST_ASD_levels_<ion>.txt; only the 3d^N rows). -> results/phase15/phase15u_bfit.md (+ .json)

Operators (determinant basis of d^N, results/phase15/15i_trees): E_av (identity), F2, F4 (coefficients = Slater F^2,
F^4), zeta, alpha L(L+1), beta Q (Racah, Q(d3, v=1) = 4), Trees' T = the 15i O_gamma (3s->3d closure; its pure
three-body part has d3 norm 403.42286 and is the e1 direction), T1, T2 = coefficients of the pure three-body e1, e2
(d3 Frobenius norm 1) in mEh. The two-body set {E_av, F2, F4, alpha, beta} is complete in d^N, so T, T1, T2 do not
depend on the two-body part of the three-body operator; F2, alpha, beta do (fit 1a is run with both).
Fits (all on J levels; levels assigned by J and energy order, then followed by eigenvector overlap):
  1a  E_av F2 F4 zeta alpha beta T           (JHR 1982 appendix 2, fit E)
  1b  E_av F2 F4 zeta alpha beta T1 T2       (Uylings' three-body pair)
  2   1b with T1, T2 fixed (Cr IV: best space + all h 12.211 / 1.693; hole-corrected estimate 12.857 / 1.675)
Mean error = sqrt(sum r^2 / (n - p)), p counting every free parameter including E_av (JHR's definition is assumed to
be the same; the reproduction of fit E checks it). Spin-orbit: only zeta (no two-body magnetic operators in the
tools).
usage: phase15u_bfit.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
_C.setup_path()
import phase15u_crfit as CF  # noqa: E402

ROOT, TOOLS, H2E = CF.ROOT, CF.TOOLS, CF.H2E
OUT = os.path.join(ROOT, 'results', 'phase15')
NORM3 = 403.42286                      # d3 norm of the pure three-body part of O_gamma (T -> T1: x NORM3 / H2E)
IONS = [('cr', 'Cr IV', 'Cr_IV', 3), ('v', 'V III', 'V_III', 3), ('mn', 'Mn V', 'Mn_V', 3), ('fe', 'Fe III', 'Fe_III', 6)]
JHR_E = dict(T=-5.985, T_err=0.182, F2=74225.9, F4=45757.3, zeta=277.4, alpha=30.5, beta=-474.8, mean_error=25.2)
FIX = {'best space + all h': (12.211, 1.693), 'hole-corrected estimate': (12.857, 1.675)}
LET = CF.LET


def nist(fn, N):
    """3d^N rows of a NIST ASD level table -> [(label, leading %, 2J, E)]."""
    cfg = term = None
    out = []
    for line in open(os.path.join(_C.REFDATA, 'levels', f'NIST_ASD_levels_{fn}.txt')):
        c = [x.strip() for x in line.split('|')]
        if len(c) < 8 or c[0].startswith('Config'):
            continue
        cfg, term = c[0] or cfg, c[1] or term
        if cfg == f'3d{N}' and c[4]:
            j = c[2].split('/')
            out.append((term, c[6], int(j[0]) * (1 if len(j) == 2 else 2), float(c[4])))
    return out


class D(CF.DN):
    def __init__(self, N):
        super().__init__(N)
        o = self.o
        self.ops.update(F2=o['F2'], F4=o['F4'])
        if N == 3:
            self.ops['T'] = np.load(os.path.join(TOOLS, 'Ogamma_d3.npy'))
            self.ops['Tp'] = np.load(os.path.join(TOOLS, 'Ogamma3_d3.npy'))   # pure three-body part only
        # (S, L, seniority) classes per M_J = J block, for the eigenvector composition
        self.cls = {}
        for j2, idx in self.blocks.items():
            K = self.S2[np.ix_(idx, idx)] + np.sqrt(2) * self.L2[np.ix_(idx, idx)] + np.sqrt(3) * self.Q[np.ix_(idx, idx)]
            w, u = np.linalg.eigh(K)
            lab = []
            for a in range(len(w)):
                x = u[:, a]
                s = x @ self.S2[np.ix_(idx, idx)] @ x
                ll = x @ self.L2[np.ix_(idx, idx)] @ x
                q = x @ self.Q[np.ix_(idx, idx)] @ x
                S2p1 = int(round(np.sqrt(1 + 4 * s)))
                L = int(round((-1 + np.sqrt(1 + 4 * ll)) / 2))
                v = min(range(N % 2, N + 1, 2), key=lambda v: abs((N - v) * (12 - N - v) / 4 - q))
                lab.append((S2p1, L, v))
            self.cls[j2] = (u, lab)

    def comp(self, j2, x):
        u, lab = self.cls[j2]
        w = {}
        for a, k in enumerate(lab):
            w[k] = w.get(k, 0.0) + float((u[:, a] @ x) ** 2)
        multi = {(m, L) for (m, L, v) in w if sum(1 for k in w if k[:2] == (m, L)) > 1}
        name = {k: f'{k[0]}{LET[k[1]]}' + (f'(v={k[2]})' if k[:2] in multi else '') for k in w}
        return sorted(((name[k], p) for k, p in w.items() if p > 0.005), key=lambda t: -t[1])


def fit(d, lev, names, fixed, p0):
    byj = {}
    for x in sorted(lev, key=lambda x: x[3]):
        byj.setdefault(x[2], []).append(x)
    obs = [(j2, k, x[3]) for j2, xs in sorted(byj.items()) for k, x in enumerate(xs)]
    f = CF.lsq(d, names, fixed, obs, d.levels, p0, iters=200)
    # final states, eigenvector composition, order check
    H = sum(f['p'][n] * d.ops[n] for n in names) + sum(a * d.ops[n] for n, a in fixed.items())
    S = d.levels(H)
    rows, order_ok = [], True
    ref = {}
    for (j2, k, e), r in zip(obs, f['resid']):
        w, v, idx = S[j2]
        kk = int(np.argmin(abs(w - (e - r))))
        order_ok &= kk == k
        x = byj[j2][k]
        rows.append(dict(J2=j2, E=e, calc=e - r, resid=r, nist=x[0], nist_pct=x[1], comp=d.comp(j2, v[:, kk])))
    f['levels'], f['order_ok'], f['fixed'] = rows, order_ok, fixed
    f['rms_n'] = float(np.sqrt(np.mean(np.square(f['resid']))))
    return f


def t1t2(f):
    """T1, T2 in mEh from a fit (T: Trees' T in cm-1 -> T1 = T * NORM3 / H2E, T2 = 0)."""
    p, e = f['p'], f['err']
    if 'T1' in p:
        return p['T1'], e['T1'], p['T2'], e['T2']
    if 'T1' in f['fixed']:
        return f['fixed']['T1'], 0.0, f['fixed']['T2'], 0.0
    t = p.get('T', p.get('Tp'))
    te = e.get('T', e.get('Tp'))
    return t * NORM3 / H2E, te * NORM3 / H2E, 0.0, 0.0


def run():
    res = {}
    for ion, lab, fn, N in IONS:
        d = D(N)
        lev = nist(fn, N)
        e0 = np.mean([x[3] for x in lev])
        base = dict(A=e0, F2=74000.0, F4=46000.0, zeta=300.0, alpha=50.0, beta=-300.0)
        R = {}
        if N == 3:
            for key, tn in (('1a', 'T'), ('1a_pure', 'Tp')):
                nm = ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', tn]
                R[key] = fit(d, lev, nm, {}, [*base.values(), 0.0])
        nm = ['A', 'F2', 'F4', 'zeta', 'alpha', 'beta', 'T1', 'T2']
        p0 = [R['1a']['p'][n] for n in nm[:6]] + [t1t2(R['1a'])[0], 0.0] if N == 3 else [*base.values(), 9.0, 3.0]
        R['1b'] = fit(d, lev, nm, {}, p0)
        if ion == 'cr':
            for k, (a, b) in FIX.items():
                R[f'2: {k}'] = fit(d, lev, nm[:6], dict(T1=a, T2=b), [R['1b']['p'][n] for n in nm[:6]])
        res[ion] = dict(label=lab, N=N, n_levels=len(lev), fits=R)
    return res


def fmt(res):
    L = ['B: J-level fits of the 3d^N ground configurations to NIST ASD (cm-1; T1, T2 in mEh; mean error = '
         'sqrt(sum r^2 / (n - p)), p including E_av). 1a_pure = 1a with the pure three-body part of Trees\' operator.', '',
         '| ion | fit | n | p | mean error | rms (/n) | F2 | F4 | ζ | α | β | T (cm⁻¹) | T1 (mEh) | T2 (mEh) | B | C | '
         'order ok |', '|' + '---|' * 17]
    for ion, r in res.items():
        for k, f in r['fits'].items():
            p, e = f['p'], f['err']
            t1, t1e, t2, t2e = t1t2(f)
            T = next((n for n in ('T', 'Tp') if n in p), None)
            tt = f'{p[T]:+.3f} ± {e[T]:.3f}' if T else ''
            nfree = len(p)

            def pe(n, fm='.1f'):
                return f'{p[n]:{fm}} ± {e[n]:{fm}}' if n in p else ''
            B = p['F2'] / 49 - 5 * p['F4'] / 441
            C = 35 * p['F4'] / 441
            fx = '' if 'T1' in p or T else ' (fixed)'
            L.append(f'| {r["label"]} | {k} | {f["n"]} | {nfree} | {f["rms"]:.1f} | {f["rms_n"]:.1f} | {pe("F2")} | '
                     f'{pe("F4")} | {pe("zeta")} | {pe("alpha", ".2f")} | {pe("beta")} | {tt} | '
                     f'{t1:.3f}' + (f' ± {t1e:.3f}' if t1e else fx) + f' | {t2:+.3f}' + (f' ± {t2e:.3f}' if t2e else
                                                                                     ('' if T else fx)) +
                     f' | {B:.1f} | {C:.1f} | {"yes" if f["order_ok"] else "NO"} |')
    for ion, r in res.items():
        f = r['fits']['1b']
        L += ['', f'{r["label"]} 3d{r["N"]}, fit 1b: levels (NIST label and leading %, our eigenvector composition)', '',
              '| J | E obs | E calc | obs − calc | NIST | ours |', '|---|---|---|---|---|---|']
        for x in sorted(f['levels'], key=lambda x: x['E']):
            j = f'{x["J2"]}/2' if x['J2'] % 2 else str(x['J2'] // 2)
            c = ', '.join(f'{n} {100 * w:.0f}' for n, w in x['comp'][:3])
            L.append(f'| {j} | {x["E"]:.1f} | {x["calc"]:.1f} | {x["resid"]:+.1f} | {x["nist"]} {x["nist_pct"]} | {c} |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    res = run()
    txt = fmt(res)
    open(os.path.join(OUT, 'phase15u_bfit.md'), 'w').write(txt)
    json.dump(res, open(os.path.join(OUT, 'phase15u_bfit.json'), 'w'), indent=1, default=float)
    print(txt)
