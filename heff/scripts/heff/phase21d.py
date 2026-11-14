#!/usr/bin/env python3
"""Phase 21 job D (stored H only): T1 of Cr3+ in the Step 8 space (besth_sall, crnd3) by excitation class.

Classes (CSF configuration relative to 3s2 3p6 3d3; V = virtual orbitals 11s.., nv = electrons in V):
  3s→3d        3s 3p6 3d4 (no virtual, no inner hole)
  2s→3d        2s 3s2 3p6 3d4 (one 2s hole, no virtual)
  3d→v single  3s2 3p6 3d2 v
  3p²→3d²      3s2 3p4 3d5 (no virtual)
  3p→f         a 3p hole and an f virtual (no inner hole; the phase17a rule)
  3s3p double  3s 3p5 ... (one 3s and one 3p hole, no f virtual, no inner hole)
  others       everything else (3d² → v v', 3s → v, 3p² → v v', 3s² → 3d², other inner holes, ...), split further
               by the phase17a classes for information
Closure (C2) and S2-diag: additive over Q, from the stored per-configuration increments (results/phase17/
a_crnd_besth_sall.npz, phase17a.incr), HF P block + class sums.
Exact: not additive. Two partitions from principal sub-matrices of the stored Step 8 H (same CSFs, phase21a method):
  cumulative  P + classes added in the order above; increments sum to exact(Step 8) - HF by construction
  remove-one  exact(Step 8) - exact(Step 8 without the class)
H_eff: iterative.heff_block, placed on the vcm HF P block, phase15k.decompose3 (as phase19b / the Step 8 'exact').
Commands: classes (counts, closure / S2-diag; seconds) / exact (stored H; ~1.5-2 h) -> phase21d.json"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
OUT = os.path.join(ROOT, 'results', 'phase21')
JS = os.path.join(OUT, 'phase21d.json')
ORDER = ['3s→3d', '2s→3d', '3d→v single', '3p²→3d²', '3p→f', '3s3p double', 'others']
FMT = 'v15u_besth_sall_d3_{t}e_crnd3'


def setup():
    os.environ['ION'] = 'crnd'
    os.environ['HEFF_SOLVER'] = 'iter'
    _C.setup_path()


def occ(label):
    d = {}
    for tok in label.split():
        m = re.fullmatch(r'(\d+)([spdfghik])(\d+)', tok)
        d[m.group(1) + m.group(2)] = int(m.group(3))
    for o in ('1s', '2s', '2p', '3s', '3p', '3d'):
        d.setdefault(o, 0)
    return d


def cls7(label, N=3):
    d = occ(label)
    vir = {o: q for o, q in d.items() if int(o[:-1]) >= 11 and q}
    nv = sum(vir.values())
    h1s, h2s, h2p = 2 - d['1s'], 2 - d['2s'], 6 - d['2p']
    inner = h1s + h2s + h2p
    hs, hp = 2 - d['3s'], 6 - d['3p']
    if inner == 0 and nv == 0 and (hs, hp) == (0, 0):
        return 'P'
    if inner == 0 and nv == 0 and (hs, hp) == (1, 0):
        return '3s→3d'
    if (h1s, h2s, h2p) == (0, 1, 0) and nv == 0 and (hs, hp) == (0, 0):
        return '2s→3d'
    if inner == 0 and (hs, hp) == (0, 0) and nv == 1 and d['3d'] == N - 1:
        return '3d→v single'
    if inner == 0 and nv == 0 and (hs, hp) == (0, 2):
        return '3p²→3d²'
    if inner == 0 and hp >= 1 and any(o[-1] == 'f' for o in vir):
        return '3p→f'
    if inner == 0 and (hs, hp) == (1, 1):
        return '3s3p double'
    return 'others'


def classes():
    setup()
    import phase17a as A
    a = A.load('crnd', 'besth_sall')
    cf = [str(c) for c in a['cfgs']]
    c7 = np.array([cls7(c) for c in cf])
    c17 = np.array([A.classify(c) for c in cf])
    res = dict(n_cfg={k: int((c7 == k).sum()) for k in ORDER}, second={}, others_split={})
    hf = A.params({t: a[f'hf|{t}'] for t in A.terms()})
    res['hf'] = dict(T1=hf['T1'], T2=hf['T2'])
    for m in ('C2', 'S2-diag'):
        tot = A.total(a, m)
        rows = {k: A.incr(a, m, c7 == k) for k in ORDER}
        res['second'][m] = dict(total=dict(T1=tot['T1'], T2=tot['T2']),
                                rows={k: dict(T1=v['T1'], T2=v['T2']) for k, v in rows.items()},
                                sum_check=hf['T1'] + sum(v['T1'] for v in rows.values()))
        print(m, 'total', round(tot['T1'], 4), 'HF', round(hf['T1'], 4), ' '.join(f"{k} {v['T1']:+.3f}" for k, v in rows.items()),
              'sum', round(res['second'][m]['sum_check'], 4))
        oth = {}
        for k in sorted(set(c17[c7 == 'others'])):
            msk = (c7 == 'others') & (c17 == k)
            v = A.incr(a, m, msk)
            oth[k] = dict(T1=v['T1'], T2=v['T2'], n=int(msk.sum()))
        res['others_split'][m] = oth
    d = json.load(open(JS)) if os.path.exists(JS) else {}
    d['classes'] = res
    json.dump(d, open(JS, 'w'), indent=1, default=float)
    print('configurations per class', res['n_cfg'])


def exact():
    import shutil
    setup()
    import phase19b as P19
    from atomci_heff import iterative as IT
    from atomci_heff.sparse import SymLower
    pc = dict(N=3)
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hf = P19.hf_blocks(pc)
    variants = {'cum:P': ['P']}
    for i, k in enumerate(ORDER[:-1]):
        variants[f'cum:{k}'] = ['P'] + ORDER[:i + 1]
    variants['full'] = ['P'] + ORDER
    for k in ORDER:
        variants[f'rm:{k}'] = ['P'] + [x for x in ORDER if x != k]
    by = {v: {} for v in variants}
    info = {v: {} for v in variants}
    t0 = time.time()
    ncsf = {}
    for L, twoS, m, t in P19.terms(3):
        assert shutil.disk_usage('/data').free / 2 ** 30 > 150
        z = np.load(os.path.join(ROOT, 'results', FMT.format(t=t) + '.npz'), allow_pickle=True)
        lab = [str(x) for x in z['labels']]
        op = SymLower.from_npz(z)
        del z
        c = {l: cls7(l) for l in set(lab)}
        cl = np.array([c[l] for l in lab])
        ncsf[t] = {k: int((cl == k).sum()) for k in ['P'] + ORDER}
        U = np.array(maps[f'vcm_d3_{t}e'])
        for v, ks in variants.items():
            idx = np.where(np.isin(cl, ks))[0]
            assert list(idx[:m]) == list(range(m))
            sub = op if len(idx) == op.n else op.sub(idx)
            if sub.n == m:
                ex = U.T @ sub.dense() @ U          # canonical basis, as heff_block (fixed 13 Nov; cum:P re-done by fixd)
                inf = dict(wP=1.0)
            else:
                b = dict(P=list(range(m)), Q=list(range(m, sub.n)), m=m, U=U)
                ex, inf = IT.heff_block(b, sub)
            Hpp = U.T @ sub.sub(np.arange(m)).dense() @ U
            by[v][(L, twoS)] = hf[t] + ex - Hpp
            info[v][t] = dict(n=int(sub.n), wP=float(inf.get('wP', 1.0)))
            del sub
        print(f'{t}: {ncsf[t]} ({time.time() - t0:.0f} s)', flush=True)
        del op
    res = dict(ncsf=ncsf, variants={})
    for v in variants:
        p = P19.decompose(pc, by[v])
        res['variants'][v] = dict(T1=p['T1'], T2=p['T2'], T3=p['T3'], T4=p['T4'], B=p['B'], C=p['C'],
                                  min_wP=min(i['wP'] for i in info[v].values()), n=info[v])
        print(f"{v:18s} T1 {p['T1']:8.4f} T2 {p['T2']:8.4f}", flush=True)
    V = res['variants']
    prev = V['cum:P']['T1']
    cum = {}
    for i, k in enumerate(ORDER):
        key = f'cum:{k}' if i < len(ORDER) - 1 else 'full'
        cum[k] = V[key]['T1'] - prev
        prev = V[key]['T1']
    res['cumulative'] = cum
    res['remove_one'] = {k: V['full']['T1'] - V[f'rm:{k}']['T1'] for k in ORDER}
    res['wall_s'] = time.time() - t0
    d = json.load(open(JS)) if os.path.exists(JS) else {}
    d['exact'] = res
    json.dump(d, open(JS, 'w'), indent=1, default=float)
    print('cumulative', {k: round(v, 3) for k, v in cum.items()}, 'sum', round(sum(cum.values()), 4))
    print('remove-one', {k: round(v, 3) for k, v in res['remove_one'].items()})


def fixd():
    """cum:P of the 13 Nov exact run was placed in the CSF basis (P-only branch, now fixed): replace it by the HF P
    block (hf + U^T PHP U - U^T PHP U = hf) and recompute the cumulative partition. The other variants are unaffected."""
    setup()
    import phase19b as P19
    pc = dict(N=3)
    hf = P19.hf_blocks(pc)
    p = P19.decompose(pc, {(L, twoS): hf[t] for L, twoS, m, t in P19.terms(3)})
    d = json.load(open(JS))
    res = d['exact']
    V = res['variants']
    V['cum:P'].update(T1=p['T1'], T2=p['T2'], T3=p['T3'], T4=p['T4'], B=p['B'], C=p['C'], fixed='HF P block (13 Nov)')
    prev = V['cum:P']['T1']
    cum = {}
    for i, k in enumerate(ORDER):
        key = f'cum:{k}' if i < len(ORDER) - 1 else 'full'
        cum[k] = V[key]['T1'] - prev
        prev = V[key]['T1']
    res['cumulative'] = cum
    json.dump(d, open(JS, 'w'), indent=1, default=float)
    print('cum:P', round(p['T1'], 6), round(p['T2'], 6), 'cumulative', {k: round(v, 3) for k, v in cum.items()},
          'sum', round(sum(cum.values()), 4), 'full', round(V['full']['T1'], 4))


if __name__ == '__main__':
    {'classes': classes, 'exact': exact, 'fixd': fixd}[sys.argv[1]]()
