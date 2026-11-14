#!/usr/bin/env python3
"""Phase 14c: within-sector test of "own orbitals + V^2 with dressed denominators" (no lifts).

For N = 2, 3, 4, own orbitals (14b tags), spaces s3d, s6d, 3d, core, g; one sector at a time:
  H = H0 + V + T (13h decomposition from the sector's own tape labels: eps = exact configuration averages,
  T = term-dependent class parts incl. Q-Q between configurations, V = H_PQ)
  (a) H_PP;  (b) H_PP - V diag(1/Delta_i) V+  (closure, configuration-average Delta_i = eps_i - E_av(P));
  (c) H_PP + 1/2 (Y V+ + V Y+), (E_P + T_PP) Y - Y (eps_Q + T_QQ) = V (Sylvester, all orders in T at O(V^2)),
      and its T^1 / T^2 terms added to (b).
Compared with the exact des Cloizeaux H_eff via the 14b within-sector split (e0, e' = e1..e4, t, f).
Step 2: e_k^(1)(N) from H_PP, Delta e_k = e_k - e_k^(1), t (t^(1) = 0); own 3d F^2, F^4, <r>.
-> results/phase14/phase14c.txt/.json
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
import phase13b as B  # noqa: E402
import phase13d as D  # noqa: E402
import phase13f as F  # noqa: E402
import phase13g as G  # noqa: E402
import phase13h as H  # noqa: E402
import phase13h_add as HA  # noqa: E402
import phase14b as P14  # noqa: E402
import phase14a_orb as O  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase14')
SPACES = ['s3d', 's6d', '3d', 'core', 'g']
VORB = {'s3d': {'3d', '11d', '12d'}, 's6d': {'3d', '11d', '12d', '13d', '14d', '15d'}, '3d': {'3d', '11d', '12d'},
        'core': {'3s', '3d'}, 'g': {'3d', '11g'}}
NS = (2, 3, 4)


def run_name(sp, N, t, tag):
    if sp == 'core':
        return f'vcm_d{N}_{t}e_{tag}'
    return f'vq_{D.ION[N]}_{t}e_{tag}_{ {"3d": "r3d", "g": "r1d1g"}.get(sp, sp)}'


def to_vec(N, bl, mats):
    by = {(b['L'], b['twoS']): b['U'].T @ mats[id(b)] @ b['U'] for b in bl}
    return B.vec_from_blocks(N, by)


def parts(N, x):
    """e0, e' (e1..e4), t, f part vectors of x (14b split)."""
    names, vecs = P14.basis_ops(N)
    G_ = np.array([[S.ip(N, p, q) for q in vecs] for p in vecs])
    c = np.linalg.solve(G_, [S.ip(N, p, x) for p in vecs])
    grp = {'e0': ['e0'], 'e': B.EK[1:], 't': B.TK, 'f': [n for n in names if n.startswith('f')]}
    out = {g: sum((ci * v for ci, v, nm in zip(c, vecs, names) if nm in ks), np.zeros_like(x)) for g, ks in grp.items()}
    return out, dict(zip(names, c.tolist()))


def nrm(N, x):
    return float(np.sqrt(max(S.ip(N, x, x), 0.0)))


def sector(sp, N):
    tag = P14.OWN[N]
    bl = P14.sector_blocks(sp, N, tag)
    for b in bl:
        b['W'] = G.tape_W(run_name(sp, N, b['t'], tag))
        b['H0'] = b['H']
    lf = G.fit_labels(bl)
    const, chk = H.decompose(bl, VORB[sp], lf)
    classes = [c for c in H.CLASSES if max(np.abs(b['T'][c]).max() for b in bl) > 1e-12]
    eps = {id(b): b['eps_full'] for b in bl}
    chkH = max(np.abs(H.model_H(b, eps, {c: 1.0 for c in classes}) - b['H']).max() for b in bl)
    used = [nm for b in bl for nm in b['W'] if H.label_class(nm, VORB[sp]) not in (None,)]
    nullmax = max((G.lval(nm, lf)[1] for nm in set(used)), default=0.0)
    # exact
    ex = {}
    info = []
    for b in bl:
        Bk, inf = F.heff_block(b, b['H'])
        ex[(b['L'], b['twoS'])] = Bk
        info.append(inf)
    xe = B.vec_from_blocks(N, ex)
    # predictions (CSF P basis)
    HPP = {id(b): b['H'][np.ix_(b['P'], b['P'])] for b in bl}
    clo = {}
    for b in bl:
        P, Q = b['P'], b['Q']
        e = eps[id(b)]
        dl = e[Q] - e[P[0]]
        V = b['H'][np.ix_(P, Q)]
        clo[id(b)] = -(V / dl) @ V.T
    ser, _, syl = HA.series(bl, eps, classes, kmax=3)
    # Sylvester: T_PP is inside H_PP already (H_PP = E_P + T_PP)
    preds = {'a': HPP,
             'b': {k: HPP[k] + clo[k] for k in HPP},
             'b+T1': {k: HPP[k] + ser[0][k] + ser[1][k] for k in HPP},
             'b+T1+T2': {k: HPP[k] + ser[0][k] + ser[1][k] + ser[2][k] for k in HPP},
             'c': {k: HPP[k] + syl[k] for k in HPP}}
    clo_chk = max(np.abs(ser[0][k] - clo[k]).max() for k in clo)
    pe, ce = parts(N, xe)
    res = dict(ex={g: nrm(N, v) for g, v in pe.items()}, cex=ce, wP=min(i['wP'] for i in info),
               below=sum(i['below'] for i in info), chk=dict(spread=chk['spread'], offw=chk['offw'], H=chkH,
                                                             null=nullmax, lf=lf['dev'], clo=clo_chk), pred={})
    for k, m in preds.items():
        x = to_vec(N, bl, m)
        pp, cp = parts(N, x)
        r = {}
        for g in ('e0', 'e', 't', 'f'):
            d = pp[g] - pe[g]
            r[g] = dict(n=nrm(N, pp[g]), dev=nrm(N, d), rel=nrm(N, d) / max(res['ex'][g], 1e-300))
            if g in ('t', 'f') and res['ex'][g] > 0 and nrm(N, pp[g]) > 0:
                cos = S.ip(N, pp[g], pe[g]) / (nrm(N, pp[g]) * res['ex'][g])
                r[g]['ratio'] = nrm(N, pp[g]) / res['ex'][g]
                r[g]['ang'] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        res['pred'][k] = dict(parts=r, c=cp)
    return res


def own_slater():
    bs = O.Basis()
    z = np.load(os.path.join(OUT, 'orbsets_14b.npz'))
    out = {}
    for N in NS:
        c = z[f'{N}|2'][:, 0]
        P = bs.P(2, c)
        out[N] = dict(F2=bs.R(P, P, P, P, 2), F4=bs.R(P, P, P, P, 4), F0=bs.R(P, P, P, P, 0),
                      r=float(np.sum(P * P * bs.r * bs.w)))
    return out


def diffs(v, Ns):
    v = np.array(v)
    p = np.polyfit(Ns, v, 1)
    return dict(v=v.tolist(), d1=np.diff(v).tolist(), d2=np.diff(v, 2).tolist(), slope=float(p[0]),
                lin=float(np.sqrt(np.mean((np.polyval(p, Ns) - v) ** 2))))


def main():
    t0 = time.time()
    js = {}
    w = []
    a = w.append
    a('# Phase 14c: within-sector test of own orbitals + V^2 with dressed denominators')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip())
    R = {}
    for sp in SPACES:
        for N in NS:
            R[(sp, N)] = sector(sp, N)
        print(f'{sp} {time.time() - t0:.0f} s', flush=True)
    js['sectors'] = {f'{sp} {N}': {k: v for k, v in x.items() if k != 'cex'} for (sp, N), x in R.items()}

    a('\n## Checks')
    a('| space | N | P weight min | states below | label fit (Eh) | max null comp | H rebuild (Eh) | closure = T^0 '
      'term (Eh) |')
    a('|---|---|---|---|---|---|---|---|')
    for (sp, N), x in R.items():
        c = x['chk']
        a(f'| {sp} | {N} | {x["wP"]:.3f} | {x["below"]} | {c["lf"]:.1e} | {c["null"]:.1e} | {c["H"]:.1e} | '
          f'{c["clo"]:.1e} |')

    a('\n## Step 1. Predictions vs exact, by part (mEh; rel = |pred - exact| / |exact part|)')
    a('Exact |e\'|, |t|, |f| per sector; for each prediction: rel dev of e\', t (ratio, angle), f (ratio, angle); '
      'e0 dev in mEh.')
    for (sp, N), x in R.items():
        ex = x['ex']
        a(f'\n### {sp}, N = {N}: exact |e\'| {1e3 * ex["e"]:.2f}, |t| {1e3 * ex["t"]:.4f}, |f| {1e3 * ex["f"]:.4f} mEh')
        a('| pred | e0 dev (mEh) | e\' rel | t rel | t ratio, angle | f rel | f ratio, angle | abs(f_pred) (mEh) |')
        a('|---|---|---|---|---|---|---|---|')
        for k, p in x['pred'].items():
            r = p['parts']
            tt = f'{r["t"].get("ratio", 0):.3f}, {r["t"].get("ang", 0):.1f}' if ex['t'] > 0 else '-'
            ff = f'{r["f"].get("ratio", 0):.3f}, {r["f"].get("ang", 0):.1f}' if ex['f'] > 0 else '-'
            a(f'| {k} | {1e3 * r["e0"]["dev"]:.3f} | {r["e"]["rel"]:.4f} | ' +
              (f'{r["t"]["rel"]:.3f}' if ex['t'] > 0 else '-') + f' | {tt} | ' +
              (f'{r["f"]["rel"]:.3f}' if ex['f'] > 0 else '-') + f' | {ff} | {1e3 * r["f"]["n"]:.2e} |')

    # step 2
    a('\n## Step 2. HF (first-order) part vs correlation part of the parameters, own orbitals, N = 2, 3, 4')
    sl = own_slater()
    a('Own 3d: ' + '; '.join(f'N={N}: <r> {sl[N]["r"]:.4f}, F^0 {sl[N]["F0"]:.5f}, F^2 {sl[N]["F2"]:.5f}, '
                              f'F^4 {sl[N]["F4"]:.5f} Eh' for N in NS))
    js['slater'] = sl
    a('e in mEh (e0 in Eh). e^(1) from H_PP; De = e - e^(1). D1, D2; lin = rms residual of a linear fit / |slope|.')
    s2 = {}
    for sp in SPACES:
        a(f'\n### {sp}')
        a('| coef | e^(1) N=2/3/4 | D1 | D2 | lin/abs(slope) | De N=2/3/4 | D1 | D2 | lin/abs(slope) | D2(De)/D2(e) |')
        a('|---|---|---|---|---|---|---|---|---|---|')
        rows = {}
        for k in B.EK + B.TK:
            Ns = [N for N in NS if k in R[(sp, N)]['cex']]
            e1 = [R[(sp, N)]['pred']['a']['c'].get(k, 0.0) for N in Ns]
            ee = [R[(sp, N)]['cex'][k] for N in Ns]
            de = [x - y for x, y in zip(ee, e1)]
            f = 1.0 if k == 'e0' else 1e3
            if len(Ns) < 2:
                continue
            A_, Dd, Ee = diffs(e1, Ns), diffs(de, Ns), diffs(ee, Ns)
            rows[k] = dict(e1=A_, de=Dd, e=Ee)
            fmt = lambda d: (' / '.join(f'{f * v:.4f}' for v in d['v']), ' / '.join(f'{f * v:.4f}' for v in d['d1']),  # noqa: E731
                             ' / '.join(f'{f * v:.4f}' for v in d['d2']), f'{d["lin"] / max(abs(d["slope"]), 1e-300):.3f}')
            r2 = (Dd['d2'][0] / Ee['d2'][0]) if Ee['d2'] and abs(Ee['d2'][0]) > 1e-12 else float('nan')
            a(f'| {k} | ' + ' | '.join(fmt(A_)) + ' | ' + ' | '.join(fmt(Dd)) + f' | {r2:.2f} |')
        s2[sp] = rows
    js['step2'] = s2
    a(f'\nWall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase14c.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    open(os.path.join(OUT, 'phase14c.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
