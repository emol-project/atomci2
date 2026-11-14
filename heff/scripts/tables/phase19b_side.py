#!/usr/bin/env python3
"""Phase 19b side items (seconds).
  hansen  Fe III, Co IV (1b) fits with Hansen et al. 1988 f1..f12 and three-body p_t, p_t' fixed (T4 = p_t x 1.83813,
          T3 = p_t' x 1.83813, the phase17q20 Fe III mapping), mean deviation and lambda on that vector
  sall    T1, T2 increments of the chain item 'all virtual s' (phase18_fe2.json / phase18_co.json final budgets)
-> results/phase19/phase19b_side.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
OUT = os.path.join(ROOT, 'results', 'phase19')
HP = {'fe': (0.013, 0.010, -0.013, 0.009), 'co': (0.027, 0.010, -0.014, 0.009)}   # p_t, err, p_t', err (Table 1)


def hansen():
    import phase17q11 as Q
    import phase17q20 as Q20
    import phase18_nist  # noqa: F401
    FS = [f'f{i}' for i in range(1, 13)]
    A = Q20.jlA()
    co = json.load(open(os.path.join(ROOT, 'results', 'phase18', 'phase18_co.json')))
    hfv = {'fe': {k: float(Q20.H_FE[k][0]) for k in FS}, 'co': {k: float(v[0]) for k, v in co['hansen_f'].items()}}
    out = {}
    for ion in ('fe', 'co'):
        obs, N = Q.nist_obs(ion)
        d = Q.ds(N)
        p0 = Q.to_bc(Q.bfit_free(ion)['p'])
        x0 = [p0[n] for n in Q.NM1B]
        Xf = {k: hfv[ion][k] / np.sqrt(A[k]) for k in FS}
        pt, _, ptp, _ = HP[ion]
        X34 = dict(Xf, T4=pt * Q20.MEH_PER_P, T3=ptp * Q20.MEH_PER_P)
        r = {}
        for nm, X in (('f only (T3 = T4 = 0)', Xf), ('f + p_t, p_t\'', X34)):
            f = Q20.wlsq(d, Q.NM1B, X, obs, x0)
            d.ops['Xab'] = sum(c * d.ops[k] for k, c in X.items())
            fl = Q20.wlsq(d, Q.NM1B + ['Xab'], {}, obs, x0 + [1.0])
            del d.ops['Xab']
            r[nm] = dict(mean=f['mean'], order_ok=f['order_ok'], T1=f['p']['T1'], T2=f['p']['T2'], T1_err=f['err']['T1'],
                         T2_err=f['err']['T2'], lam=fl['p']['Xab'], lam_err=fl['err']['Xab'], lam_mean=fl['mean'],
                         T3=X.get('T3', 0.0), T4=X.get('T4', 0.0))
            print(f"{ion} {nm:22s} mean {f['mean']:.2f} (order {f['order_ok']}) T1 {f['p']['T1']:.3f} T2 {f['p']['T2']:.3f} "
                  f"lam {fl['p']['Xab']:.3f} +- {fl['err']['Xab']:.3f}  [T3 {r[nm]['T3']:+.4f} T4 {r[nm]['T4']:+.4f} mEh]")
        out[ion] = r
    return out


def sall():
    out = {}
    for ion, fn in (('Fe2+', 'phase18_fe2.json'), ('Co3+', 'phase18_co.json')):
        d = json.load(open(os.path.join(ROOT, 'results', 'phase18', fn)))
        out[ion] = {}
        for R in ('proj', 'levels'):
            b = next(x for x in d['final'][R]['budget'] if x['item'] == 'all virtual s')
            out[ion][R] = dict(dT1pct=b['dT1pct'], dT2=b['dT2'], u1=b['u1'], u2=b['u2'],
                               base_T1=d['final'][R]['base']['T1'], base_T2=d['final'][R]['base']['T2'])
            print(f"{ion} {R}: all virtual s dT1 {b['dT1pct']:+.3f} % dT2 {b['dT2']:+.4f} mEh (u {b['u1']:.3f} / {b['u2']:.4f})")
    return out


if __name__ == '__main__':
    r = dict(hansen=hansen(), sall=sall())
    json.dump(r, open(os.path.join(OUT, 'phase19b_side.json'), 'w'), indent=1, default=float)
