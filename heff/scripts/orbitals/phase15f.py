#!/usr/bin/env python3
"""Phase 15f: does decontracting the Sapporo d shell converge T2 of the d path? (V2+ d3, Cr3+ d3, Cr2+ d4)

Basis: v_qzp_dunc.bas / cr_dunc.bas = the -diffuse sets with the d shell replaced by its 12 primitives (s p f g h
unchanged). Orbitals: configuration-average HF of 3d^N (core re-optimized, 15a recipe), virtual d = eigenvectors of
the own generalized Fock (14a/15a, A14.virtuals) in energy order, l >= 3 = eigenvectors of h. d path as 15e (b):
[Ne] frozen, 3s2 3p6 x full CI of N electrons in {3d} + the first k virtual d.
Decks: the 15e construction (13b vcm deck + new CSF list) with the basis size fields (nob, nbf, mpob, ind) and the
outer-orbital header rows changed for 12 d; for the contracted basis the generator reproduces the 15e decks.
usage: phase15f.py basis | orb | dims | prep | run <tag> <space> ... | analyze
-> results/phase15/phase15f_orb.json, phase15f.txt/.json, orbsets_15f.npz
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import gto  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402
from atomci_heff.reader import read_vectors, records  # noqa: E402
import counts as CN  # noqa: E402
import gen_v  # noqa: E402
import phase13b as B  # noqa: E402
import phase13d as D  # noqa: E402
import phase14a as A14  # noqa: E402
import phase14a_orb as O  # noqa: E402
import phase14b as P14  # noqa: E402
import phase15a as F15  # noqa: E402
import phase15e as E15  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase15')
EL = {'V': 23, 'Cr': 24}
BAS = {('V', 'c'): 'v_qzp.bas', ('V', 'u'): 'v_qzp_dunc.bas', ('Cr', 'c'): 'cr.bas', ('Cr', 'u'): 'cr_dunc.bas'}
IONS = [(3, 'V'), (3, 'Cr'), (4, 'Cr')]
QP = {3: (3, 3), 4: (4, 6)}
TAG15E = {(3, 'V'): 'o14n3', (3, 'Cr'): 'cr15n3', (4, 'Cr'): 'cr15n4'}
KS = {'c': (2, 3, 5), 'u': (2, 3, 5, 7, 9, 11)}
HF86_V2P_AV = -942.09524533
DV = [f'{k}d' for k in range(11, 22)]
LMAP = dict(E15.LMAP, **{o: 2 for o in DV})
CAP = {o: 2 * (2 * l + 1) for o, l in LMAP.items()}
V_DUP = 0.336462          # V core-contraction exponent 8 % below the valence 0.363876


def tag(el, N, b):
    return TAG15E[(N, el)] if b == 'c' else f'{el.lower()}15fu{N}'


def nbf_of(bas):
    ls = [l for l, _, _ in gto.read_bas(os.path.join(_C.BASIS, bas))]
    return [ls.count(l) for l in range(max(ls) + 1)]


class Bas(O.Basis):
    def __init__(self, bas, Z, fns=None):
        self.fns = fns or gto.read_bas(os.path.join(_C.BASIS, bas))
        ls = [l for l, _, _ in self.fns]
        self.nbf = [ls.count(l) for l in range(max(ls) + 1)]
        self.mat = {l: gto.matrices(self.fns, l, Z) for l in range(len(self.nbf))}
        self.h = {l: self.mat[l]['T'] + self.mat[l]['V'] for l in self.mat}
        self.S = {l: self.mat[l]['S'] for l in self.mat}
        self.x, self.r = I.radial_grid()
        n = len(self.x)
        dx = self.x[1] - self.x[0]
        w = np.full(n, 2.0)
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        self.w = w * dx / 3 * self.r
        self.chi = I.radial_functions(self.fns, {l: np.eye(self.nbf[l]) for l in range(len(self.nbf))}, self.r)


# ---------------------------------------------------------------- step 1: basis

def dexps(bas):
    return sorted({e for l, ee, c in gto.read_bas(os.path.join(_C.BASIS, bas)) if l == 2 for e in ee}, reverse=True)


def ovl_info(S):
    s = np.linalg.eigvalsh(S)
    return float(s[0]), float(s[-1] / s[0])


def basis_step():
    res = {}
    for el, Z in EL.items():
        r = {}
        for b in 'cu':
            bs = gto.read_bas(os.path.join(_C.BASIS, BAS[(el, b)]))
            S = gto.matrices(bs, 2, Z)['S']
            r[b] = dict(nbf=nbf_of(BAS[(el, b)]), d_min_eig=ovl_info(S)[0], d_cond=ovl_info(S)[1])
            if b == 'u':
                r['u']['d_exps'] = dexps(BAS[(el, b)])
                # leave-one-out: smallest eigenvalue after removing each primitive
                ex = np.array(r['u']['d_exps'])
                Sp = (2 * np.sqrt(np.outer(ex, ex)) / (ex[:, None] + ex[None, :])) ** 3.5
                r['u']['loo'] = {f'{x:.7g}': float(np.linalg.eigvalsh(np.delete(np.delete(Sp, i, 0), i, 1))[0])
                                 for i, x in enumerate(ex)}
        res[el] = r
    return res


# ---------------------------------------------------------------- step 2: orbitals

def own_set(bs, N):
    q, pair = QP[N]
    r = O.scf(bs, q, pair)
    Cv, e = A14.virtuals(bs, r['C'], A14.own_coef(q, pair))
    C = {0: Cv[0], 1: Cv[1], 2: Cv[2]}
    for l in range(3, len(bs.nbf)):
        C[l] = O.gen_eig(bs.h[l], bs.S[l])[1]
    dev = max(float(np.abs(C[l].T @ bs.S[l] @ C[l] - np.eye(bs.nbf[l])).max()) for l in C)
    P = bs.P(2, C[2][:, 0])
    return C, dict(E=float(r['E']), it=r['it'], orth_dev=dev, eps_virt_d=[float(x) for x in e],
                   r=float(np.sum(P * P * bs.r * bs.w)), F2=bs.R(P, P, P, P, 2), F4=bs.R(P, P, P, P, 4))


def orb():
    t0 = time.time()
    js, sets = {'basis': basis_step()}, {}
    for N, el in IONS:
        for b in 'cu':
            bs = Bas(BAS[(el, b)], EL[el])
            C, inf = own_set(bs, N)
            js[f'{el} {N} {b}'] = inf
            if b == 'u':
                sets[(el, N)] = C
            print(f'{el} N={N} {b} E {inf["E"]:.8f} <r> {inf["r"]:.6f} ({inf["it"]} it) {time.time() - t0:.0f} s',
                  flush=True)
    # V2+: 11 primitives (near-duplicate removed) as a stability check of the 12-primitive result
    fns = [f for f in gto.read_bas(os.path.join(_C.BASIS, BAS[('V', 'u')]))
           if not (f[0] == 2 and abs(f[1][0] - V_DUP) < 1e-6)]
    bs = Bas(None, 23, fns)
    _, inf = own_set(bs, 3)
    inf['d_min_eig'], inf['d_cond'] = ovl_info(bs.S[2])
    js['V 3 u11'] = inf
    print(f'V N=3 u11 E {inf["E"]:.8f}', flush=True)
    np.savez(os.path.join(OUT, 'orbsets_15f.npz'), **{f'{el}|{N}|{l}': C[l] for (el, N), C in sets.items() for l in C})
    json.dump(js, open(os.path.join(OUT, 'phase15f_orb.json'), 'w'), indent=1)


def write_be_no(C, tg, nbf):
    """be.NO (two identical sets) for an arbitrary per-l basis size; header from runs/vq_orb/be.NO patched."""
    rec = records(os.path.join(ROOT, 'runs', 'vq_orb', 'be.NO'))
    nl = len(nbf)
    hdrs = []
    for s in (0, 1):
        hdr = rec[s * (1 + nl)]
        iv = list(struct.unpack(f'<{(len(hdr) - 72) // 4}i', hdr[72:]))
        assert iv[3] == nl and iv[4:4 + nl] == list(O.NBF) and iv[24:24 + nl] == list(O.NBF) and iv[44] == nl
        for l in range(nl):
            iv[4 + l] = iv[24 + l] = nbf[l]
            assert iv[46 + 4 * l:48 + 4 * l] == [l, 1] and iv[48 + 4 * l] == O.NBF[l]
            iv[48 + 4 * l] = nbf[l]
        hdrs.append(hdr[:72] + struct.pack(f'<{len(iv)}i', *iv))
    work = os.path.join(ROOT, 'runs', f'vq_orb_{tg}')
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, 'be.NO'), 'wb') as fo:
        for s in (0, 1):
            hdr = hdrs[s]
            fo.write(struct.pack('<i', len(hdr)) + hdr + struct.pack('<i', len(hdr)))
            for l in range(nl):
                r = struct.pack('<i', l) + C[l].T.astype('<f8').tobytes()
                fo.write(struct.pack('<i', len(r)) + r + struct.pack('<i', len(r)))
    chk = read_vectors(os.path.join(work, 'be.NO'), nbf)
    assert all(np.array_equal(chk[k][l], C[l]) for k in (0, 1) for l in range(nl))


# ---------------------------------------------------------------- step 3: decks, runs

def fci(N, orbs):
    out = []
    import itertools
    for occ in itertools.product(*[range(min(CAP[o], N) + 1) for o in orbs]):
        if sum(occ) == N:
            out.append({o: q for o, q in zip(orbs, occ) if q})
    return out


def configs(N, k):
    P0 = E15.full_cfg('vcm', {'3d': N})
    cfs = [E15.full_cfg('vcm', d) for d in fci(N, ['3d'] + DV[:k])]
    return [P0] + [c for c in cfs if c != P0]


def spaces(N, b):
    ks = KS[b] if N == 3 else ((2, 3, 5) if b == 'c' else KS4)
    return [f'd{k}' for k in ks]


KS4 = (2, 3, 5, 7)


def rname(space, N, t, tg):
    return f'v15f_{space}_d{N}_{t}e_{tg}' if not tg.startswith(('o14', 'cr15n')) else E15.rname(space, N, t, tg)


def outer_header(bas, Z):
    """Outer-orbital header rows of be.csf from atinputs for the given basis (rows before the active row)."""
    nbf = nbf_of(bas)
    outer = ', '.join(f'{n - gen_v.CORE[l] - (1 if l == 2 else 0)}{"SPDFGHIK"[l]}' for l, n in enumerate(nbf))
    ind = '\n'.join(f'      - record: 2, l: {l}, first: 1, last: {n}' for l, n in enumerate(nbf))
    txt = f"""title:  " hdr "
deck:   be.deck
csf:    be.csf
atom:
  z:     {Z}
  ang:   F
  spin:  4
files:
  cfptable:  ./CFPTBL
  basis:     {bas}
  format:    atomci
  mcp:       mcp.dat
  open:
    - unit: 27, file: be.NO, format: binary, status: old
orbital:
  frozen_core:  1S, 2S, 2P
  active:       3S, 3P, 3D
  outer:        {outer}
reference:
  - scf: true, occ: 3S*2 3P*6 3D*3
ctrl:
  ai:  type: 2, idprnt: 0
  wf:  rule: 1, npflg: 0
  tr:
    record: 1
    ind:
{ind}
  ci:  ips: 10, maxit: 400, nsdim: 100, nstate: 1
  rs:  print: -0.001
"""
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, 'x.inp'), 'w').write(txt)
    shutil.copy(os.path.join(_C.BASIS, bas), tmp)
    env = dict(os.environ, PYTHONPATH=_C.ATOMCI_SCRIPTS, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', '-m', 'atinputs', 'x.inp'], cwd=tmp, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    csf = open(os.path.join(tmp, 'be.csf')).read()
    shutil.rmtree(tmp)
    return csf.split('  3S   3P   3D\n')[0]


def make_deck(N, t, el, b, space, tg):
    src = os.path.join(ROOT, 'inputs', f'vcm_d{N}_{t}e')
    deck = open(os.path.join(src, 'be.deck')).read()
    csf = open(os.path.join(src, 'be.csf')).read()
    bas = BAS[(el, b)]
    assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1
    deck = deck.replace('z=23,', f'z={EL[el]},').replace('FILE=v_qzp.bas', f'FILE={bas}')
    header = csf.split(' ))\n', 1)[0] + ' ))\n'
    nbf = nbf_of(bas)
    if nbf != list(O.NBF):
        old, rest = header.split('  3S   3P   3D\n')
        header = outer_header(bas, EL[el]) + '  3S   3P   3D\n' + rest
        old_nbf = ','.join(map(str, O.NBF))
        deck = gen_v.sub_all(deck, 'nob', ','.join(map(str, nbf)), expect=old_nbf)
        deck = gen_v.sub_all(deck, 'nbf', ','.join(map(str, nbf)), expect=old_nbf)
        mp = [int(x) for x in re.search(r'^mpob=([\d,]+),$', deck, flags=re.M).group(1).split(',')]
        cut = np.cumsum([0] + list(O.NBF))
        parts = [mp[cut[l]:cut[l + 1]] for l in range(len(O.NBF))]
        assert parts[2] == list(range(O.NBF[2], 0, -1))
        parts[2] = list(range(nbf[2], 0, -1))
        deck = gen_v.sub_first(deck, 'mpob', ','.join(str(x) for p in parts for x in p))
        deck = deck.replace(f'\n2,2,1,{O.NBF[2]},\n', f'\n2,2,1,{nbf[2]},\n', 1)
        assert f'\n2,2,1,{nbf[2]},\n' in deck
    seq = [x.lower() for x in re.findall(r'\d+[SPDFGH]', header.split(')')[0])]
    cf = configs(N, int(space[1:]))
    tot, ncfg, per = CN.count(LMAP, cf, t)
    assert per[cf[0]] == [m for L, tS, m, tt in D.term_list(N) if tt == t][0]
    order = [c for c in cf if c in per]
    lines = ['  ' + '  )  '.join(f'{o[:-1]}{o[-1].upper()}* {dict(c)[o]}' for o in seq if o in dict(c)) + '  ))'
             for c in order]
    deck = gen_v.sub_all(deck, 'nsdim', tot + 10)
    dst = rname(space, N, t, tg)
    deck = re.sub(r"^title='.*',$", f"title=' {dst} ',", deck, count=1, flags=re.M)
    return dst, deck, header + '\n'.join(lines) + '\n )\n'


def prep():
    z = np.load(os.path.join(OUT, 'orbsets_15f.npz'))
    n, same = 0, 0
    for N, el in IONS:
        bas = BAS[(el, 'u')]
        write_be_no({l: z[f'{el}|{N}|{l}'] for l in range(6)}, tag(el, N, 'u'), nbf_of(bas))
        for b in 'cu':
            tg = tag(el, N, b)
            for space in spaces(N, b):
                for L, twoS, m, t in D.term_list(N):
                    dst, deck, csf = make_deck(N, t, el, b, space, tg)
                    if b == 'c':
                        # contracted basis: must reproduce the 15e decks
                        d0 = os.path.join(ROOT, 'inputs', dst)
                        assert open(os.path.join(d0, 'be.deck')).read() == deck, dst
                        assert open(os.path.join(d0, 'be.csf')).read() == csf, dst
                        same += 1
                        continue
                    out = os.path.join(ROOT, 'inputs', dst)
                    os.makedirs(out, exist_ok=True)
                    open(os.path.join(out, 'be.deck'), 'w').write(deck)
                    open(os.path.join(out, 'be.csf'), 'w').write(csf)
                    n += 1
    print(f'{n} new decks; {same} contracted decks identical to 15e')


def dims():
    w = ['| ion | basis | space | largest block CSFs | sum over terms | configurations |', '|---|---|---|---|---|---|']
    for N in (3, 4):
        for k in KS['u']:
            cf = configs(N, k)
            mx = tot = 0
            for L, twoS, m, t in D.term_list(N):
                a = CN.count(LMAP, cf, t)
                mx, tot = max(mx, a[0]), tot + a[0]
            w.append(f'| d{N} | any | d{k} | {mx} | {tot} | {len(cf)} |')
    print('\n'.join(w))
    return w


def run_list(names):
    """Run ATOMCI per (run, vectors) in this process, sequentially; one thread."""
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for dst, tg in names:
        if os.path.isdir(os.path.join(ROOT, 'runs', dst)) and os.path.exists(os.path.join(ROOT, 'results', dst + '.npz')):
            continue
        t0 = time.time()
        subprocess.run([sys.executable, _C.script('run_atomci.py'), dst,
                        '--vectors-from', f'vq_orb_{tg}'], check=True, stdout=subprocess.DEVNULL, env=env)
        print(f'{dst} {time.time() - t0:.1f} s', flush=True)


def run(el, N, spaces_):
    tg = tag(el, N, 'u')
    run_list([(rname(sp, N, t, tg), tg) for sp in spaces_ for L, twoS, m, t in D.term_list(N)])


# ---------------------------------------------------------------- analysis

def blocks(space, N, tg):
    js = json.load(open(os.path.join(ROOT, 'results', 'phase13', 'phase13b_b.json')))
    out = []
    for L, twoS, m, t in D.term_list(N):
        Hm, lab = B.load_run(rname(space, N, t, tg))
        P = [i for i, l in enumerate(lab) if l == lab[0]]
        assert len(P) == m and P == list(range(m)), (space, t, P)
        out.append(dict(N=N, t=t, L=L, twoS=twoS, H=Hm, lab=lab, U=np.array(js['maps'][f'vcm_d{N}_{t}e']), m=m, P=P,
                        Q=[i for i in range(len(lab)) if i not in P]))
    return out


def jl_sign_guard():
    """Require the lifted e4, t1..t3 to reproduce JL Table 7 before any d4 analysis. (Before fix/stretched-tiebreak,
    cfp_l.stretched resolved tied largest components by np.argmax, i.e. by BLAS rounding, and the d4 signs flipped
    with OMP_NUM_THREADS=1 or 2; the tie now goes to the lowest index, and the guard is kept as a check.)"""
    from atomci_heff import dshell as S
    ops, _, _ = B.ops_fock()
    J = S.jl_table(4)
    dev = max(float(np.abs(S.to_vec(4, S.lift(*ops[k], 4)) - J[k]).max()) for k in ('e4', 't1', 't2', 't3'))
    assert dev < 1e-10, f'd4 seniority signs inconsistent with JL Table 7 (max dev {dev:.3g})'
    return dev


def analyze():
    import phase13f as F
    t0 = time.time()
    print(f'JL d4 sign guard: max dev {jl_sign_guard():.1e}', flush=True)
    c15 = json.load(open(os.path.join(OUT, 'phase15c.json')))
    O1 = E15.d3(dict(zip(B.TK, c15['O1'])))
    O2 = E15.d3(dict(zip(B.TK, c15['O2'])))
    O1, O2 = O1 / E15.nrm3(O1), O2 / E15.nrm3(O2)
    e15 = json.load(open(os.path.join(OUT, 'phase15e.json')))
    orbj = json.load(open(os.path.join(OUT, 'phase15f_orb.json')))
    res = {}
    for N, el in IONS:
        for b in 'cu':
            tg = tag(el, N, b)
            r = {}
            ref = None
            for space in spaces(N, b):
                if not all(os.path.exists(os.path.join(ROOT, 'results', rname(space, N, t, tg) + '.npz'))
                           for L, twoS, m, t in D.term_list(N)):
                    continue
                bl = blocks(space, N, tg)
                hpp = {bb['t']: bb['U'].T @ bb['H'][np.ix_(bb['P'], bb['P'])] @ bb['U'] for bb in bl}
                ref = ref or hpp
                dev = max(float(np.abs(hpp[t] - ref[t]).max()) for t in hpp)
                by, info = {}, []
                for bb in bl:
                    Bk, inf = F.heff_block(bb, bb['H'])
                    by[(bb['L'], bb['twoS'])] = Bk
                    info.append(inf)
                x = B.vec_from_blocks(N, by)
                dec = P14.decompose(N, x)
                tt = E15.d3(dec['c'])
                T1, T2 = S_ip(tt, O1), S_ip(tt, O2)
                r[space] = dict(T1=T1, T2=T2, t=E15.nrm3(tt), out=E15.nrm3(tt - T1 * O1 - T2 * O2) / max(E15.nrm3(tt), 1e-300),
                                ang=float(np.degrees(np.arctan2(T2, T1))), wP=min(i['wP'] for i in info),
                                below=sum(i['below'] for i in info), hpp_dev=dev,
                                E_P=[float(np.linalg.eigvalsh(h)[0]) for h in hpp.values()])
                if b == 'c':
                    e = e15[f'{N} {el}'][space]
                    r[space]['d15e'] = max(abs(r[space]['T1'] - e['T1']), abs(r[space]['T2'] - e['T2']))
            res[f'{el} {N} {b}'] = r
            print(f'{el} d{N} {b} {time.time() - t0:.0f} s', flush=True)
    json.dump(res, open(os.path.join(OUT, 'phase15f.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    report(res, orbj, time.time() - t0)


def S_ip(a, b):
    from atomci_heff import dshell as S
    return S.ip(3, a, b)


def ion(el, N):
    return {('V', 3): 'V2+ d3', ('Cr', 3): 'Cr3+ d3', ('Cr', 4): 'Cr2+ d4'}[(el, N)]


def report(res, orbj, wall):
    w = ['# Phase 15f: d path with the Sapporo d shell decontracted', __doc__.split('usage:')[0].strip(), '']
    bj = orbj['basis']
    w.append('## Step 1: basis (d overlap, smallest eigenvalue / condition number)')
    w.append('| element | contracted (6 d) | decontracted (12 d) | d exponents (decontracted) |')
    w.append('|---|---|---|---|')
    for el in EL:
        c, u = bj[el]['c'], bj[el]['u']
        w.append(f'| {el} | {c["d_min_eig"]:.3e} / {c["d_cond"]:.3e} | {u["d_min_eig"]:.3e} / {u["d_cond"]:.3e} | '
                 + ' '.join(f'{x:.7g}' for x in u['d_exps']) + ' |')
    for el in EL:
        lo = bj[el]['u']['loo']
        best = max(lo, key=lo.get)
        w.append(f'- {el}: removing one primitive raises the smallest eigenvalue to at most {lo[best]:.2e} (removing {best}).')
    w.append('\n## Step 2: configuration-average HF (Eh; differences in mEh)')
    w.append('| ion | E contracted | E decontracted | decontracted - contracted | 3d <r> contracted | 3d <r> decontracted | '
             'rel. change of <r> | F2 rel. change | F4 rel. change | max |C^T S C - 1| |')
    w.append('|---|---|---|---|---|---|---|---|---|---|')
    for N, el in IONS:
        c, u = orbj[f'{el} {N} c'], orbj[f'{el} {N} u']
        w.append(f'| {ion(el, N)} | {c["E"]:.8f} | {u["E"]:.8f} | {1e3 * (u["E"] - c["E"]):+.4f} | {c["r"]:.6f} | '
                 f'{u["r"]:.6f} | {(u["r"] - c["r"]) / c["r"]:+.2e} | {(u["F2"] - c["F2"]) / c["F2"]:+.2e} | '
                 f'{(u["F4"] - c["F4"]) / c["F4"]:+.2e} | {u["orth_dev"]:.1e} |')
    c, u, u11 = orbj['V 3 c'], orbj['V 3 u'], orbj['V 3 u11']
    w.append(f'- V2+ vs HF86 (d3 average, h = 1/64, {HF86_V2P_AV:.8f}): contracted {1e3 * (c["E"] - HF86_V2P_AV):+.3f}, '
             f'decontracted {1e3 * (u["E"] - HF86_V2P_AV):+.3f} mEh.')
    w.append(f'- V2+ with 11 d primitives ({V_DUP} removed; smallest eigenvalue {u11["d_min_eig"]:.2e}, cond '
             f'{u11["d_cond"]:.2e}): E {u11["E"]:.8f}, {1e3 * (u11["E"] - u["E"]):+.4f} mEh vs 12; 3d <r> {u11["r"]:.6f}.')
    for N, el in IONS:
        e = orbj[f'{el} {N} u']['eps_virt_d']
        w.append(f'- {ion(el, N)} virtual-d Fock eigenvalues (decontracted, Eh): ' + ' '.join(f'{x:.3f}' for x in e))
    w.append('\n## Step 3: d path T1, T2 vs k (mEh, d3-normalized O2\', O2\'\'); angle; out of plane; min P weight; '
             'states below; max |H_PP - H_PP(k=2)| (Eh)')
    w.append('| ion | basis | k | T1 | T2 | angle | out | min P | below | H_PP check | vs 15e |')
    w.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for N, el in IONS:
        for b in 'cu':
            for sp, x in res[f'{el} {N} {b}'].items():
                w.append(f'| {ion(el, N)} | {"contracted" if b == "c" else "decontracted"} | {sp[1:]} | {1e3 * x["T1"]:+.4f} | '
                         f'{1e3 * x["T2"]:+.4f} | {x["ang"]:+.2f} | {x["out"]:.4f} | {x["wP"]:.3f} | {x["below"]} | '
                         f'{x["hpp_dev"]:.1e} | {"%.1e" % x["d15e"] if "d15e" in x else "-"} |')
    w.append('\n## T2 vs k (mEh)')
    ks = KS['u']
    w.append('| ion | basis | ' + ' | '.join(f'k={k}' for k in ks) + ' | last step |')
    w.append('|---|---|' + '---|' * (len(ks) + 1))
    for N, el in IONS:
        for b in 'cu':
            r = res[f'{el} {N} {b}']
            v = [r.get(f'd{k}', {}).get('T2') for k in ks]
            got = [x for x in v if x is not None]
            ls = f'{100 * (got[-1] - got[-2]) / got[-2]:+.1f} %' if len(got) > 1 else '-'
            w.append(f'| {ion(el, N)} | {"contracted" if b == "c" else "decontracted"} | ' +
                     ' | '.join(f'{1e3 * x:.3f}' if x is not None else '-' for x in v) + f' | {ls} |')
    w.append(f'\nAnalysis wall time {wall:.0f} s.')
    open(os.path.join(OUT, 'phase15f.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'orb':
        orb()
    elif cmd == 'dims':
        dims()
    elif cmd == 'prep':
        prep()
    elif cmd == 'run':
        run(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
    elif cmd == 'analyze':
        analyze()
