#!/usr/bin/env python3
"""Phase 19b (12 Nov): level-shift test of the 3d^(N-1) ns channel on the stored H (no ATOMCI runs). Writes only
results/phase19/.

Spaces (stored H, results/<run>.npz):
  v, cr  Step 8 (besth_sall; vnd3, crnd3), d3: exact H_eff placed on the vcm HF P block as phase15k.analyze3,
         projection phase15k.decompose3 (the Step 8 'exact' of paper_tables section 2)
  fe, co RAS k = 1 + h with all virtual s (k1fh_sall; fend6, cond6; the space of the chain item 'all virtual s'),
         d6: own-orbital runs, H_eff itself (phase17q11 dump convention), projection phase17q11.DS(6).decompose
         (as phase18_co.x_of / three6 'proj')
H_eff: atomci_heff.iterative.heff_block (block Davidson, the m largest-P-weight states, des Cloizeaux) - the paper's rule.
Shift: Delta (cm-1) added to the CSF diagonal of every 3s2 3p6 3d^(N-1) v_s CSF with v_s the 4s-like virtual s
(the s orbital with the largest weight in the first 3d^(N-1) ns-dominated root of the calibration blocks).
Commands (one ion per process; env set inside):
  est  <ion>                     sizes of the blocks
  cal  <ion>                     calibration roots (job-5 style) in the calibration blocks, unshifted and shifted
  heff <ion> <variant> ...       variants: none | half | full | own  -> phase19b_<ion>.json"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
OUT = os.path.join(ROOT, 'results', 'phase19')
EH_CM = 219474.6313705
ION = {'v': dict(N=3, env='vnd', fmt='v15u_besth_sall_d3_{t}e_vnd3', nist='V_III', cal=['4F', '2D'], gs='4F'),
       'cr': dict(N=3, env='crnd', fmt='v15u_besth_sall_d3_{t}e_crnd3', nist='Cr_IV', cal=['4F', '2D'], gs='4F'),
       'fe': dict(N=6, env='fend', fmt='v15u_k1fh_sall_d6_{t}e_fend6', nist='Fe_III', cal=['5D', '3G', '3H'], gs='5D'),
       'co': dict(N=6, env='cond', fmt='v15u_k1fh_sall_d6_{t}e_cond6', nist='Co_IV', cal=['5D', '3G', '3H'], gs='5D')}


def setup(ion):
    c = ION[ion]
    os.environ['ION'] = c['env']
    os.environ['HEFF_SOLVER'] = 'iter'
    _C.setup_path()
    return c


def jpath(ion):
    return os.path.join(OUT, f'phase19b_{ion}.json')


def jload(ion):
    p = jpath(ion)
    return json.load(open(p)) if os.path.exists(p) else {}


def jsave(ion, r):
    json.dump(r, open(jpath(ion), 'w'), indent=1, default=float)


def classify(labels, N):
    import phase19 as P
    return P.classify(labels, N)


def nist_cal(c):
    """(2J+1) centroids above the ground-term centroid: the lowest 3d^(N-1)4s term of each calibration (L,S)."""
    import phase19 as P
    N = c['N']
    rows = P.nist_all(c['nist'])
    cen = lambda rr: sum((r[2] + 1) * r[3] for r in rr) / sum(r[2] + 1 for r in rr)   # noqa: E731
    g = [r for r in rows if r[0] == f'3d{N}' and r[1].split()[-1] == c['gs'] and (r[1] in (c['gs'], 'a ' + c['gs']))]
    g0 = cen(g)
    out = {}
    terms = {}
    for r in rows:
        if P.is_dn1s(r[0], N, 4):
            terms.setdefault((r[0], r[1]), []).append(r)
    for t in c['cal']:
        cand = [(cen(rr) - g0, cf, tt) for (cf, tt), rr in terms.items() if tt.split()[-1] == t]
        if cand:
            e, cf, tt = min(cand)
            out[t] = dict(E=e, cfg=cf, term=tt)
    return out, g0


def load(c, t):
    from atomci_heff.sparse import SymLower
    name = c['fmt'].format(t=t)
    z = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
    op = SymLower.from_npz(z)
    lab = [str(x) for x in z['labels']]
    del z
    return name, op, lab


def terms(N):
    import phase13d as D
    return D.term_list(N)


def est(ion):
    c = setup(ion)
    import zipfile
    tot = 0
    for L, twoS, m, t in terms(c['N']):
        name = c['fmt'].format(t=t)
        zf = zipfile.ZipFile(os.path.join(ROOT, 'results', name + '.npz'))
        n = int(np.load(os.path.join(ROOT, 'results', name + '.npz'))['N'])
        sz = sum(i.file_size for i in zf.infolist())
        tot += sz
        print(f'{name:40s} m {m} N {n:7d} npz {sz / 1e9:.2f} GB')
    print('total', tot / 1e9, 'GB')


def roots_until_ns(op, cls, P, nsorb, kmax=64, tol=1e-8):
    """block Davidson on op - c until the first root with wP < 0.5 and dominant 3d^(N-1) ns weight."""
    from atomci_heff.sparse import SymLower, davidson
    c = float(op.d[P].mean())
    s = SymLower.__new__(SymLower)
    s.L, s.LT, s.d, s.n = op.L, op.LT, op.d - c, op.n
    k, X0 = 8, None
    while True:
        q = list(np.argsort(s.d)[:k])
        X = np.zeros((op.n, k))
        X[q, np.arange(k)] = 1.0
        if X0 is not None:
            X[:, :X0.shape[1]] = X0
        E, C, rn, it, nmv = davidson(s, k, X0=X, tol=tol, maxit=4000, max_sub=max(6 * k, 60))
        assert rn.max() < tol, rn.max()
        wP = (C[P] ** 2).sum(axis=0)
        w = {o: (C[cls == o] ** 2).sum(axis=0) for o in nsorb}
        wns = sum(w.values())
        found = [a for a in range(k) if wP[a] < 0.5 and wns[a] > max(wP[a], 1 - wP[a] - wns[a])]
        if found:
            return E + c, wP, w, found[0]
        X0, k = C, 2 * k
        assert k <= kmax, f'no ns root among {kmax // 2} roots'


def cal(ion, shift=None):
    """calibration blocks (+ the ground-term block for E_gs): first 3d^(N-1) ns root; shift (cm-1, on the 4s-like s)
    optional (to report the shifted positions)."""
    import shutil
    c = setup(ion)
    assert shutil.disk_usage('/data').free / 2 ** 30 > 150
    N = c['N']
    nc, g0 = nist_cal(c)
    r = jload(ion)
    out =dict(nist=nc, nist_ground_centroid=g0, blocks={})
    for t in [c['gs']] + [x for x in c['cal'] if x != c['gs']]:
        t0 = time.time()
        name, op, lab = load(c, t)
        cls = classify(lab, N)
        P = np.where(cls == 'P')[0]
        nsorb = sorted({x for x in cls if x.startswith('ns:')}, key=lambda x: int(x[3:-1]))
        if shift is not None:
            v = r['cal']['vs']
            sh = shift[t] if isinstance(shift, dict) else shift
            op.d[cls == f'ns:{v}'] += sh / EH_CM
        E, wP, w, f = roots_until_ns(op, cls, P, nsorb)
        out['blocks'][t] = dict(run=name, N=int(op.n), nP=int(len(P)), n_ns={o: int((cls == o).sum()) for o in nsorb},
                                E_abs=[float(x) for x in E], wP=[float(x) for x in wP],
                                w_ns={o: [float(x) for x in w[o]] for o in nsorb}, found=int(f), wall_s=time.time() - t0)
        print(f'{ion} {t}: N {op.n} found root {f} wP {wP[f]:.3f} ' + ' '.join(f'{o} {w[o][f]:.3f}' for o in nsorb) +
              f' ({time.time() - t0:.0f} s)', flush=True)
        del op
    Eg = out['blocks'][c['gs']]['E_abs'][0]
    for t, b in out['blocks'].items():
        b['E_found_cm'] = (b['E_abs'][b['found']] - Eg) * EH_CM
        f = b['found']
        b['vs'] = max(b['w_ns'], key=lambda o: b['w_ns'][o][f])[3:]
        if t in nc:
            b['shift'] = nc[t]['E'] - b['E_found_cm']
    if shift is None:
        vs = {b['vs'] for t, b in out['blocks'].items() if t in c['cal']}
        assert len(vs) == 1, vs
        out['vs'] = vs.pop()
        own = {t: out['blocks'][t]['shift'] for t in c['cal'] if 'shift' in out['blocks'][t]}
        out['shift_own'] = own
        out['shift_mean'] = float(np.mean(list(own.values())))
    r['cal' if shift is None else ('cal_shift_own' if isinstance(shift, dict) else 'cal_shift_mean')] = out
    jsave(ion, r)
    for t, b in out['blocks'].items():
        print(f"  {t}: first 3d^(N-1)ns root {b['E_found_cm']:.0f} cm-1 above E_gs, dominant {b['vs']}"
              + (f", NIST {nc[t]['E']:.0f} ({nc[t]['cfg']} {nc[t]['term']}), shift {b['shift']:+.0f}" if 'shift' in b else ''))


def decompose(c, by):
    """{(L, twoS): m x m (Eh, canonical basis)} -> proj dict (T in mEh, two-body in cm-1)."""
    import phase13b as B
    if c['N'] == 3:
        import phase15n  # noqa: F401  (ion setup)
        import phase15k as K
        return K.decompose3(B.vec_from_blocks(3, by))
    import phase17q11 as Q
    from atomci_heff import dshell as SH
    X = SH.from_vec(6, B.vec_from_blocks(6, by) * Q.EH_CM)
    p, res, rank = Q.ds(6).decompose(X)
    return dict(p, resid=res)


def hf_blocks(c):
    """d3: the vcm HF P blocks (phase17a.blocks); d6: None (own-orbital runs, H_eff itself)."""
    if c['N'] != 3:
        return None
    import phase15n  # noqa: F401
    import phase15k as K
    tl = terms(3)
    hfn = [f'vcm_d3_{t}e_{K.TAG3}' for L, twoS, m, t in tl]
    return {b['t']: b['U'].T @ b['H'][:b['m'], :b['m']] @ b['U'] for b in K.H.blocks(3, hfn)}


def heff(ion, variants):
    import shutil
    c = setup(ion)
    from atomci_heff import iterative as IT
    N = c['N']
    r = jload(ion)
    vs, own, mean = r['cal']['vs'], r['cal']['shift_own'], r['cal']['shift_mean']
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hf = hf_blocks(c)
    for var in variants:
        assert shutil.disk_usage('/data').free / 2 ** 30 > 150
        t0 = time.time()
        by, info = {}, {}
        for L, twoS, m, t in terms(N):
            name, op, lab = load(c, t)
            cls = classify(lab, N)
            P = list(range(m))
            assert all(cls[i] == 'P' for i in P) and (cls == 'P').sum() == m, name
            sel = cls == f'ns:{vs}'
            sh = {'none': 0.0, 'half': 0.5 * mean, 'full': mean, 'own': own.get(t, mean),
                  'full_x1S': 0.0 if t == '1S' else mean}[var]       # full_x1S: diagnostic, intruder block unshifted
            op.d[sel] += sh / EH_CM
            U = np.array(maps[f'vcm_d{N}_{t}e'])
            b = dict(P=P, Q=list(range(m, op.n)), m=m, U=U)
            ex, inf = IT.heff_block(b, op)
            Hpp = U.T @ op.sub(np.asarray(P)).dense() @ U
            by[(L, twoS)] = ex if hf is None else hf[t] + ex - Hpp
            info[t] = dict(run=name, N=int(op.n), n_shifted=int(sel.sum()), shift_cm=float(sh), **{k: v for k, v in inf.items()})
            del op
        p = decompose(c, by)
        r.setdefault('heff', {})[var] = dict(proj=p, info=info, min_wP=min(i['wP'] for i in info.values()), wall_s=time.time() - t0)
        jsave(ion, r)
        print(f"{ion} {var}: T1 {p['T1']:.4f} T2 {p['T2']:.4f} T3 {p['T3']:+.4f} T4 {p['T4']:+.4f} B {p['B']:.1f} C {p['C']:.1f} "
              f"min wP {r['heff'][var]['min_wP']:.4f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == '__main__':
    cmd, ion = sys.argv[1], sys.argv[2]
    if cmd == 'est':
        est(ion)
    elif cmd == 'cal':
        cal(ion)
    elif cmd == 'calshift':
        setup(ion)
        r = jload(ion)
        cal(ion, r['cal']['shift_mean'] if sys.argv[3] == 'mean' else r['cal']['shift_own'])
    elif cmd == 'heff':
        heff(ion, sys.argv[3:])
