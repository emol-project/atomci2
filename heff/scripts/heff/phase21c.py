#!/usr/bin/env python3
"""Phase 21 job C: Fe5+ (and Cr3+ as reference) basis tests of T1, T2 in a T2-capable reduced space.

Ions (results/phase21/shadow/phase15m.py: the tracked registry plus these entries; scripts/ unchanged):
  plain  fe5nd (fe.without-diffuse.bas, fe5nd3)        crnd (cr.without-diffuse.bas, crnd3)
  (i)    fe5dc fe_qzp2012_open_d.bas (Phase 18 file)    crdc cr_qzp2012_open_d.bas (same recipe, new)
  (ii)   fe5td plain + D 382.87 (2.5 x 153.15)          crtd plain + D 309.02 (2.5 x 123.61)
  (iii)  fe5qd fe.bas (Sapporo-QZP-2012-diffuse)        crqd cr.bas (orbitals = cr15n3: the recipe reproduces it)
Orbitals: the 15a recipe (phase15f.own_set, d3), as crnd3 / fe5nd3; P-block HF runs vcm_d3_<t>e_<tag> (phase18.vcm_runs).
Space (k = 1 spaces carry no T2: RAS k = 1 + h gives T2 = -0.165 (Cr3+), -0.284 (Fe5+) against 1.714 / 1.592 in
Step 8): famnsph = every virtual d, the two most compact f, both g, no virtual s, p, h (the Phase 18 B famdc_nsph rule,
d3). Run names v21_<space>_d3_<t>e_<tag>.
Commands: orb <ion> / vcm <ion> / dims <ion> <spaces> / prep <ion> <spaces> / deckcheck (fe5nd k1fh decks regenerated
through the shadow path == the stored ones) / run <ion> <space> (ATOMCI, one block per process, timeout) /
heff <ion> <space> -> phase21c.json"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
OUT = os.path.join(ROOT, 'results', 'phase21')
SHADOW = os.path.join(_C.SCRIPTS, 'heff', 'shadow21')
JS = os.path.join(OUT, 'phase21c.json')
MY = ('famnsph', 'famall', 'fnf8')
DROP8 = ('19d', '20d')   # fallback: famnsphf without the two tightest virtual d (dc basis; dc full > 3 h)


def setup(ion):
    os.environ['ION'] = ion
    os.environ['HEFF_SOLVER'] = 'iter'
    _C.setup_path()
    sys.path.insert(0, SHADOW)
    import phase15m as M
    assert M.__file__.startswith(SHADOW), M.__file__
    sys.path.insert(0, SHADOW)          # the 15m import chain puts scripts/ in front again
    import phase15t as T
    assert T.__file__.startswith(SHADOW), T.__file__
    import phase17  # noqa: F401  (the QSCRIPT chain: 15n / 15s / 15t / 15u / 17 patches)
    import phase15n as N15
    import phase15u as U
    import phase15s as S
    Q = S.Q
    base_sp, base_rn = N15.spaces, N15.rname

    def spaces():
        sp = base_sp()
        v = U.T._virt(d='all', f=2, g=2, h=1)
        sp['famall'] = ('allcore', S.ras_cfgs_x(v))
        for nm, drop in SCAN.items():
            if nm != 'fam':
                sp[nm] = ('allcore', S.ras_cfgs_x([o for o in v if o[-1] not in drop]))
        sp['fnf8'] = ('allcore', S.ras_cfgs_x([o for o in v if o[-1] not in 'sphf' and o not in DROP8]))
        return sp

    def rname(space, t):
        if space in MY or space in SCAN:
            return f'v21_{space}_d3_{t}e_{M.TAG_}'
        return base_rn(space, t)
    N15.spaces = spaces
    Q.spaces = spaces
    N15.rname = rname
    Q.K.rname_k = rname
    return M, N15, spaces, rname


def jload():
    return json.load(open(JS)) if os.path.exists(JS) else {}


def jsave(d):
    json.dump(d, open(JS, 'w'), indent=1, default=float)


NEW = {'fe5nd': (26, 'fe.without-diffuse.bas', 'fe5nd3'), 'crnd': (24, 'cr.without-diffuse.bas', 'crnd3'),
       'crqd': (24, 'cr.bas', 'cr15n3'), 'fe5dc': (26, 'fe_qzp2012_open_d.bas', 'fe5dc3'),
       'fe5td': (26, 'fe_qzp2012_add_d382.87.bas', 'fe5td3'), 'fe5qd': (26, 'fe.bas', 'fe5qd3'),
       'crdc': (24, 'cr_qzp2012_open_d.bas', 'crdc3'), 'crtd': (24, 'cr_qzp2012_add_d309.02.bas', 'crtd3')}


def orb(ion):
    """own orbitals before any ion-dependent import (the 15t / 15n setup reads vq_orb_<tag>/be.NO at import)."""
    import types
    _C.setup_path()
    Z, bas, tag = NEW[ion]
    M = types.SimpleNamespace(Z_=Z, BAS_=bas, TAG_=tag)
    import phase15f as F
    from atomci_heff.reader import read_vectors
    bs = F.Bas(M.BAS_, M.Z_)
    C, inf = F.own_set(bs, 3)
    rv = {}
    for l in range(len(bs.nbf)):
        first = {0: 3, 1: 2, 2: 1}.get(l, 0)
        for k in range(first, bs.nbf[l]):
            P = bs.P(l, C[l][:, k])
            rv[f'{11 + k - first}{"spdfghik"[l]}'] = float(np.sum(P * P * bs.r * bs.w))
    inf.update(nbf=bs.nbf, r_virt=rv, basis=M.BAS_, tag=M.TAG_)
    p = os.path.join(ROOT, 'runs', f'vq_orb_{M.TAG_}', 'be.NO')
    if os.path.exists(p):           # crqd -> cr15n3, fe5nd / crnd: recipe check only
        R = read_vectors(p, bs.nbf)[0]
        inf['max_dev_vs_existing'] = max(float(np.abs(C[l] * np.sign(np.sum(C[l] * R[l], 0)) - R[l]).max()) for l in C)
    else:
        F.write_be_no(C, M.TAG_, bs.nbf)
    d = jload()
    d.setdefault('orb', {})[ion] = inf
    jsave(d)
    print(ion, M.TAG_, bs.nbf, f"E {inf['E']:.8f}", 'dev' if 'max_dev_vs_existing' in inf else 'new',
          inf.get('max_dev_vs_existing'), {o: round(r, 3) for o, r in rv.items() if o[-1] == 'd'}, flush=True)


def vcm(ion):
    M, *_ = setup(ion)
    import phase18 as P18
    if all(os.path.exists(os.path.join(ROOT, 'results', f'vcm_d3_{t}e_{M.TAG_}.npz'))
           for t in ('2P', '4P', '2D', '2F', '4F', '2G', '2H')):
        print(ion, 'vcm runs exist for', M.TAG_)
        return
    P18.vcm_runs(M.BAS_, M.TAG_, M.Z_)


def dims(ion, names):
    M, N15, spaces, rname = setup(ion)
    import counts as CN
    import phase15e as E15
    import phase13d as D
    sp = spaces()
    d = jload()
    for name in names:
        kind, cf = sp[name]
        per = {t: CN.count(E15.LMAP, cf, t)[0] for L, twoS, m, t in D.term_list(3)}
        d.setdefault('dims', {})[f'{ion}:{name}'] = dict(ncfg=len(cf), per=per, max=max(per.values()), sum=sum(per.values()))
        print(f'{ion}:{name:10s} cfg {len(cf):5d} largest {max(per.values()):7d} sum {sum(per.values()):8d} {per}', flush=True)
    jsave(d)


def prep(ion, names, out_root=None):
    M, N15, spaces, rname = setup(ion)
    import phase15k as K
    import phase13d as D
    sp = spaces()
    for name in names:
        kind, cf = sp[name]
        for L, twoS, m, t in D.term_list(3):
            K.write_deck3(rname(name, t), kind, cf, t, **({'out_root': out_root} if out_root else {}))
    print('decks written:', ion, names)


def deckcheck():
    """fe5nd k1fh decks through the shadow path into a temp dir, compared with inputs/v15t_k1fh_d3_<t>e_fe5nd3."""
    import filecmp
    import inspect
    import tempfile
    M, N15, spaces, rname = setup('fe5nd')
    import phase15k as K
    import phase13d as D
    sig = inspect.signature(K.write_deck3)
    tmp = tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))
    kind, cf = spaces()['k1fh']
    bad, n = [], 0
    for L, twoS, m, t in D.term_list(3):
        name = rname('k1fh', t)
        if 'out_root' in sig.parameters:
            K.write_deck3(name, kind, cf, t, out_root=tmp)
            src = os.path.join(tmp, name)
        else:       # writer has no out_root: write to inputs/ under a temporary run name, compare, remove
            tn = name.replace('v15t_', 'v21chk_')
            K.write_deck3(tn, kind, cf, t)
            src = os.path.join(ROOT, 'inputs', tn)
        for f in ('be.deck', 'be.csf'):
            n += 1
            a, b = open(os.path.join(src, f)).read(), open(os.path.join(ROOT, 'inputs', name, f)).read()
            if f == 'be.deck':
                a = a.replace(os.path.basename(src), name)
            if a != b:
                bad.append(f'{name}/{f}')
        if src.startswith(os.path.join(ROOT, 'inputs')):
            shutil.rmtree(src)
    shutil.rmtree(tmp)
    print('deckcheck:', n, 'files compared,', len(bad), 'differ', bad)
    assert not bad


def run(ion, space, timeout_h=3.0):
    M, N15, spaces, rname = setup(ion)
    import phase13d as D
    tag = M.TAG_
    for L, twoS, m, t in D.term_list(3):
        name = rname(space, t)
        if os.path.exists(os.path.join(ROOT, 'results', name + '.npz')):
            continue
        assert shutil.disk_usage('/data').free / 2 ** 30 > 150, 'disk < 150 GB'
        t0 = time.time()
        r = subprocess.run(['timeout', str(int(timeout_h * 3600)), sys.executable, _C.script('run_atomci.py'),
                            name, '--vectors-from', f'vq_orb_{tag}'], stdout=subprocess.DEVNULL,
                           env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        print(f'{name} rc {r.returncode} {time.time() - t0:.0f} s', flush=True)


def heff(ion, space):
    M, N15, spaces, rname = setup(ion)
    import phase13d as D
    import phase19b as P19
    import phase15k as K
    from atomci_heff import iterative as IT
    from atomci_heff.sparse import SymLower
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hfn = [f'vcm_d3_{t}e_{K.TAG3}' for L, twoS, m, t in D.term_list(3)]
    hf = {b['t']: b['U'].T @ b['H'][:b['m'], :b['m']] @ b['U'] for b in K.H.blocks(3, hfn)}
    by, info = {}, {}
    for L, twoS, m, t in D.term_list(3):
        name = rname(space, t)
        z = np.load(os.path.join(ROOT, 'results', name + '.npz'), allow_pickle=True)
        lab = [str(x) for x in z['labels']]
        op = SymLower.from_npz(z)
        del z
        assert all(la == lab[0] for la in lab[:m]) and all(la != lab[0] for la in lab[m:]), name
        U = np.array(maps[f'vcm_d3_{t}e'])
        b = dict(P=list(range(m)), Q=list(range(m, op.n)), m=m, U=U)
        ex, inf = IT.heff_block(b, op)
        Hpp = U.T @ op.sub(np.arange(m)).dense() @ U
        by[(L, twoS)] = hf[t] + ex - Hpp
        info[t] = dict(n=int(op.n), wP=float(inf['wP']))
        del op
    p = P19.decompose(dict(N=3), by)
    d = jload()
    d.setdefault('heff', {})[f'{ion}:{space}'] = dict(proj=p, info=info, min_wP=min(i['wP'] for i in info.values()))
    jsave(d)
    print(f"{ion} {space}: T1 {p['T1']:.4f} T2 {p['T2']:.4f} T3 {p['T3']:+.4f} T4 {p['T4']:+.4f} B {p['B']:.2f} C {p['C']:.2f} "
          f"min wP {min(i['wP'] for i in info.values()):.4f}", flush=True)


SCAN = {'fam': '', 'famnsph': 'sph', 'famnsphg': 'sphg', 'famnsphf': 'sphf', 'dpath': 'sphfg'}


def scan():
    """Space choice: T2 of sub-spaces of the stored Cr3+ fam_d5f2g2h1 H (crnd3; principal sub-matrices on the CSFs of
    the configurations without any virtual of the dropped l), against Step 8 (1.714)."""
    _C.setup_path()
    os.environ['ION'] = 'crnd'
    os.environ['HEFF_SOLVER'] = 'iter'
    import phase19b as P19
    import phase21a as A21
    from atomci_heff import iterative as IT
    from atomci_heff.sparse import SymLower
    maps = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))['maps']
    hf = P19.hf_blocks(dict(N=3))
    by = {s: {} for s in SCAN}
    n = {s: {} for s in SCAN}
    t0 = time.time()
    for L, twoS, m, t in P19.terms(3):
        z = np.load(os.path.join(ROOT, 'results', f'v15u_fam_d5f2g2h1_d3_{t}e_crnd3.npz'), allow_pickle=True)
        lab = [str(x) for x in z['labels']]
        op = SymLower.from_npz(z)
        del z
        ls = {l: {o[-1] for o, q in A21.occ(l).items() if A21.isvirt(o) and q} for l in set(lab)}
        U = np.array(maps[f'vcm_d3_{t}e'])
        for s, drop in SCAN.items():
            idx = np.array([i for i, l in enumerate(lab) if not (ls[l] & set(drop))])
            sub = op if len(idx) == op.n else op.sub(idx)
            b = dict(P=list(range(m)), Q=list(range(m, sub.n)), m=m, U=U)
            ex, inf = IT.heff_block(b, sub)
            Hpp = U.T @ sub.sub(np.arange(m)).dense() @ U
            by[s][(L, twoS)] = hf[t] + ex - Hpp
            n[s][t] = int(sub.n)
            del sub
        print(t, {s: n[s][t] for s in SCAN}, f'{time.time() - t0:.0f} s', flush=True)
        del op
    d = jload()
    d['scan'] = {}
    for s in SCAN:
        p = P19.decompose(dict(N=3), by[s])
        d['scan'][s] = dict(T1=p['T1'], T2=p['T2'], drop=SCAN[s], n=n[s], max=max(n[s].values()), T2_vs_step8=p['T2'] / 1.7141 - 1)
        print(f"{s:10s} drop '{SCAN[s]}' max {max(n[s].values()):6d} T1 {p['T1']:.4f} T2 {p['T2']:.4f} ({100 * (p['T2'] / 1.7141 - 1):+.1f} % vs Step 8)", flush=True)
    jsave(d)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'scan':
        scan()
    if c == 'orb':
        orb(sys.argv[2])
    elif c == 'vcm':
        vcm(sys.argv[2])
    elif c == 'dims':
        dims(sys.argv[2], sys.argv[3:])
    elif c == 'prep':
        prep(sys.argv[2], sys.argv[3:])
    elif c == 'deckcheck':
        deckcheck()
    elif c == 'run':
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 3.0)
    elif c == 'heff':
        heff(sys.argv[2], sys.argv[3])
