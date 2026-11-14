#!/usr/bin/env python3
"""Phase 21 job A (stored H only, no ATOMCI runs): core-pair balance in the 3s-hole intermediate states, RAS k = 1 + h.

Every space here is a configuration subset of the stored u3c1 run of the ion (v15u_u3c1_d<N>_<t>e_<tag>: k1fh_h3 + every
3- and 4-hole configuration with <= 1 virtual electron, same orbitals and virtual set as k1fh). A CSF list is generated
per configuration, so the CI matrix of a configuration subset is the principal sub-matrix of the stored H on the CSFs
of those configurations (checked: the k1fh subset reproduces the stored k1fh H_eff, and for Cr3+ the k1fh_h3 subset
the stored k1fh_h3 one).
Spaces (V = the k1fh virtual set, nv = electrons in V, level = excitation level relative to R2 = 3s 3p6 3d^(N+1)):
  k1fh     the configurations of the stored k1fh run (n_hole <= 2)
  n3a      k1fh + 3s 3p4 3d^(N+3) (the 15u u1a configuration)
  Av2      k1fh + every 3s 3p4 3d^(N+1) v v'  (no 2s / 2p / 1s hole)
  A        k1fh + every configuration with 3s1 3p4 (three 3s3p holes, one in 3s), nv <= 2, level <= 2 from R2
           (= 3s 3p4 3d^(N+3), 3s 3p4 3d^(N+2) v, 3s 3p4 3d^(N+1) v v')
  h3       k1fh + every 3-hole configuration with nv <= 2 and level <= 2 from R2 (= stored k1fh_h3: A + 3s0 3p5 ...)
  hole     the stored u3c1 itself, and n3all = k1fh + the three pure n = 3 configurations (= stored k1fh_n3): the
           Table S1 'hole remainder' is u3c1 - k1fh_n3
H_eff: atomci_heff.iterative.heff_block (the paper's rule); d3: placed on the vcm HF P block and projected by
phase15k.decompose3; d6: H_eff itself, phase17q11.DS(6).decompose (as phase19b).
Commands: cfg <ion> (configuration lists and CSF counts) / heff <ion> -> phase21a_<ion>.json"""
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
ION = {'v': dict(N=3, env='vnd', tag='vnd3'), 'cr': dict(N=3, env='crnd', tag='crnd3'),
       'mn': dict(N=3, env='mnnd', tag='mnnd3'), 'fe5': dict(N=3, env='fe5nd', tag='fe5nd3'),
       'fe': dict(N=6, env='fend', tag='fend6'), 'co': dict(N=6, env='cond', tag='cond6')}
SPACES = ['k1fh', 'n3a', 'Av2', 'A', 'h3', 'n3all', 'u3c1']


def setup(ion):
    c = ION[ion]
    os.environ['ION'] = c['env']
    os.environ['HEFF_SOLVER'] = 'iter'
    _C.setup_path()
    return c


def occ(label):
    d = {}
    for tok in label.split():
        m = re.fullmatch(r'(\d+)([spdfghik])(\d+)', tok)
        d[m.group(1) + m.group(2)] = int(m.group(3))
    for o in ('1s', '2s', '2p', '3s', '3p', '3d'):
        d.setdefault(o, 0)
    return d


def isvirt(o):
    return int(o[:-1]) >= 11


def nv(d):
    return sum(q for o, q in d.items() if isvirt(o))


def level(d, ref):
    return sum(max(0, q - d.get(o, 0)) for o, q in ref.items())


def select(c, labs_u, cfg_k1fh):
    """{space: sorted CSF indices into the u3c1 block}."""
    N = c['N']
    R2 = {'1s': 2, '2s': 2, '2p': 6, '3s': 1, '3p': 6, '3d': N + 1}
    occs = {l: occ(l) for l in set(labs_u)}
    core = lambda d: d['1s'] == 2 and d['2s'] == 2 and d['2p'] == 6      # noqa: E731
    holes = lambda d: (2 - d['3s']) + (6 - d['3p'])                         # noqa: E731
    n3 = [{'3s': 1, '3p': 4, '3d': N + 3}, {'3s': 2, '3p': 2, '3d': N + 4}, {'3s': 0, '3p': 4, '3d': N + 4}]
    is_n3 = lambda d, ks: core(d) and nv(d) == 0 and any(all(d[o] == x[o] for o in x) for x in ks)  # noqa: E731
    rule = {
        'k1fh': lambda l, d: l in cfg_k1fh,
        'n3a': lambda l, d: l in cfg_k1fh or is_n3(d, n3[:1]),
        'Av2': lambda l, d: l in cfg_k1fh or (core(d) and d['3s'] == 1 and d['3p'] == 4 and d['3d'] == N + 1 and nv(d) == 2),
        'A': lambda l, d: l in cfg_k1fh or (d['3s'] == 1 and d['3p'] == 4 and nv(d) <= 2 and level(d, R2) <= 2),
        'h3': lambda l, d: l in cfg_k1fh or (holes(d) == 3 and nv(d) <= 2 and level(d, R2) <= 2),
        'n3all': lambda l, d: l in cfg_k1fh or is_n3(d, n3),
        'u3c1': lambda l, d: True}
    out, cfgs = {}, {}
    for s, f in rule.items():
        keep = {l for l in occs if f(l, occs[l])}
        cfgs[s] = sorted(keep - cfg_k1fh)
        out[s] = np.array([i for i, l in enumerate(labs_u) if l in keep])
    return out, cfgs


def runs(c, t):
    return f"v15t_k1fh_d{c['N']}_{t}e_{c['tag']}", f"v15u_u3c1_d{c['N']}_{t}e_{c['tag']}"


def labels(name):
    z = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
    return [str(x) for x in z['labels']]


def short(l):
    d = occ(l)
    s = []
    if d['2s'] < 2:
        s.append(f"2s{d['2s']}")
    if d['2p'] < 6:
        s.append(f"2p{d['2p']}")
    s += [f"3s{d['3s']}", f"3p{d['3p']}", f"3d{d['3d']}"]
    s += [f'{o}{q}' if q > 1 else o for o, q in sorted(d.items(), key=lambda x: (int(x[0][:-1]), x[0][-1])) if isvirt(o) and q]
    return ' '.join(s)


def cfg(ion):
    c = setup(ion)
    import phase19b as P19
    res = dict(blocks={}, configs=None)
    for L, twoS, m, t in P19.terms(c['N']):
        k, u = runs(c, t)
        lk, lu = labels(k), labels(u)
        assert set(lk) <= set(lu), (t, len(set(lk) - set(lu)))
        sel, cf = select(c, lu, set(lk))
        assert len(sel['k1fh']) == len(lk), (t, len(sel['k1fh']), len(lk))
        res['blocks'][t] = {s: int(len(v)) for s, v in sel.items()}
        if res['configs'] is None:
            res['configs'] = {}
        for s, v in cf.items():
            res['configs'].setdefault(s, set()).update(v)
        print(t, res['blocks'][t], flush=True)
    res['configs'] = {s: sorted(short(x) for x in v) for s, v in res['configs'].items()}
    res['n_configs_added'] = {s: len(v) for s, v in res['configs'].items()}
    print('configurations added to k1fh:', res['n_configs_added'])
    for s in ('n3a', 'Av2', 'A', 'n3all'):
        print(s, '; '.join(res['configs'][s]))
    hx = sorted(set(res['configs']['h3']) - set(res['configs']['A']))
    print('h3 - A:', len(hx), '; '.join(hx))
    ux = sorted(set(res['configs']['u3c1']) - set(res['configs']['n3all']))
    print('u3c1 - k1fh_n3 (Table S1 hole remainder):', len(ux))
    r = jload(ion)
    r['cfg'] = res
    jsave(ion, r)


def jpath(ion):
    return os.path.join(OUT, f'phase21a_{ion}.json')


def jload(ion):
    return json.load(open(jpath(ion))) if os.path.exists(jpath(ion)) else {}


def jsave(ion, r):
    json.dump(r, open(jpath(ion), 'w'), indent=1, default=float)


def heff(ion, spaces):
    import shutil
    c = setup(ion)
    import phase19b as P19
    from atomci_heff import iterative as IT
    from atomci_heff.sparse import SymLower
    N = c['N']
    pc = dict(N=N)
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hf = P19.hf_blocks(pc)
    by = {s: {} for s in spaces}
    info = {s: {} for s in spaces}
    t0 = time.time()
    for L, twoS, m, t in P19.terms(N):
        assert shutil.disk_usage('/data').free / 2 ** 30 > 150
        k, u = runs(c, t)
        lk = labels(k)
        z = np.load(os.path.join(ROOT, 'results', u + '.npz'), allow_pickle=True)
        lu = [str(x) for x in z['labels']]
        op = SymLower.from_npz(z)
        del z
        sel, _ = select(c, lu, set(lk))
        U = np.array(maps[f'vcm_d{N}_{t}e'])
        for s in spaces:
            idx = sel[s]
            assert list(idx[:m]) == list(range(m)) and all(lu[i] == lu[0] for i in range(m)), (t, s)
            sub = op if len(idx) == op.n else op.sub(idx)
            b = dict(P=list(range(m)), Q=list(range(m, sub.n)), m=m, U=U)
            ex, inf = IT.heff_block(b, sub)
            Hpp = U.T @ sub.sub(np.arange(m)).dense() @ U
            by[s][(L, twoS)] = ex if hf is None else hf[t] + ex - Hpp
            info[s][t] = dict(n=int(sub.n), **inf)
            print(f'{ion} {t} {s}: n {sub.n} wP {inf.get("wP", float("nan")):.4f} ({time.time() - t0:.0f} s)', flush=True)
            del sub
        del op
    r = jload(ion)
    for s in spaces:
        p = P19.decompose(pc, by[s])
        r.setdefault('heff', {})[s] = dict(proj=p, info=info[s], min_wP=min(i['wP'] for i in info[s].values()))
        print(f"{ion} {s}: T1 {p['T1']:.4f} T2 {p['T2']:.4f} T3 {p['T3']:+.4f} T4 {p['T4']:+.4f} B {p['B']:.2f} C {p['C']:.2f}",
              flush=True)
    r['wall_s'] = time.time() - t0
    jsave(ion, r)


if __name__ == '__main__':
    cmd, ion = sys.argv[1], sys.argv[2]
    if cmd == 'cfg':
        cfg(ion)
    elif cmd == 'heff':
        heff(ion, sys.argv[3:] or SPACES)
