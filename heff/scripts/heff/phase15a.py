#!/usr/bin/env python3
"""Phase 15a: Cr (Z = 24) sectors d2 (Cr4+), d3 (Cr3+), d4 (Cr2+), d5 (Cr+) with own orbitals; comparison with V.

Orbitals (14a/14b recipe, Cr basis basis/cr.bas): configuration-average HF of [Ar]3d^N, core re-optimized;
  s, p: all eigenvectors of the converged closed-shell Fock (as the 14b V sets: the s virtuals are F_s eigenvectors,
  not IVOs; no space below contains a virtual s or p); d: 3d + generalized-Fock virtuals with (2J-K) self terms;
  f, g, h: bare-nucleus h eigenvectors in ascending order (as vq_orb, checked for V up to sign).
Decks: the V own-sector decks (14b) with z = 24 and basis cr.bas; the electron counts, CSF lists and orbital order
  are unchanged (Cr d^N is isoelectronic in 3d^N with V d^N and the basis layout is identical). Run names keep the
  14b pattern with tag cr15n<N> (e.g. vq_v2p_4Fe_cr15n3_s3d = Cr3+ 3d3, s3d space).
Analysis per sector (no lifts): exact H_eff (14b), e/t/f split, e^(1) from H_PP, (c)/(c2)/converged Bloch (14d).
usage: phase15a.py orb | prep | run | analyze
-> results/phase15/phase15a_orb.txt/.json, phase15a.txt/.json, orbsets_15a.npz
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
from atomci_heff import gto  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402
from atomci_heff.reader import read_vectors, records  # noqa: E402
import phase13b as B  # noqa: E402
import phase14a as A14  # noqa: E402
import phase14a_orb as O  # noqa: E402
import phase14b as P14  # noqa: E402
import phase14d as D14  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase15')
NS = (2, 3, 4, 5)
QP = {2: (2, 1), 3: (3, 3), 4: (4, 6), 5: (5, 10)}
TAG_CR = {N: f'cr15n{N}' for N in NS}
TAG_V = dict(P14.OWN)
NBF = list(O.NBF)
SPACES = ['s3d', 's6d', '3d', 'core', 'g']


class ElBasis(O.Basis):
    def __init__(self, bas, Z):
        self.fns = gto.read_bas(os.path.join(_C.BASIS, bas))
        self.mat = {l: gto.matrices(self.fns, l, Z) for l in range(len(NBF))}
        self.h = {l: self.mat[l]['T'] + self.mat[l]['V'] for l in self.mat}
        self.S = {l: self.mat[l]['S'] for l in self.mat}
        self.x, self.r = I.radial_grid()
        n = len(self.x)
        dx = self.x[1] - self.x[0]
        w = np.full(n, 2.0)
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        self.w = w * dx / 3 * self.r
        self.chi = I.radial_functions(self.fns, {l: np.eye(NBF[l]) for l in range(len(NBF))}, self.r)


def diag_info(bs, C, N):
    P = bs.P(2, C[2][:, 0])
    g = A14.gaps(bs, C)
    return dict(r=float(np.sum(P * P * bs.r * bs.w)), F0=bs.R(P, P, P, P, 0), F2=bs.R(P, P, P, P, 2),
                F4=bs.R(P, P, P, P, 4), fmax=1e3 * max(abs(g[o]['f'][N]) for o in A14.VIRT),
                D={o: g[o]['D'][N] for o in A14.VIRT[:3]}, rho={o: g[o]['rho'] for o in A14.VIRT[:3]})


def orb():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bs = ElBasis('cr.bas', 24)
    sets, js = {}, {'Cr': {}, 'V': {}}
    for N in NS:
        q, pair = QP[N]
        r = O.scf(bs, q, pair)
        Cv, _ = A14.virtuals(bs, r['C'], A14.own_coef(q, pair))
        C = {0: Cv[0], 1: Cv[1], 2: Cv[2]}
        for l in range(3, len(NBF)):
            C[l] = O.gen_eig(bs.h[l], bs.S[l])[1]
        for l in C:
            assert np.abs(C[l].T @ bs.S[l] @ C[l] - np.eye(NBF[l])).max() < 1e-9
        sets[N] = C
        js['Cr'][N] = dict(E=r['E'], it=r['it'], **diag_info(bs, C, N))
        print(f'Cr N={N} E {r["E"]:.6f} ({r["it"]} it) {time.time() - t0:.0f} s', flush=True)
    np.savez(os.path.join(OUT, 'orbsets_15a.npz'), **{f'{N}|{l}': C[l] for N, C in sets.items() for l in C})
    bv = O.Basis()
    z = np.load(os.path.join(ROOT, 'results', 'phase14', 'orbsets_14b.npz'))
    C0 = O.load_original()
    for N in NS:
        q, pair = QP[N]
        C = {l: (z[f'{N}|{l}'] if l < 3 else C0[l]) for l in range(len(NBF))}
        js['V'][N] = dict(E=O.energy(bv, C, q, pair), **diag_info(bv, C, N))
    json.dump(js, open(os.path.join(OUT, 'phase15a_orb.json'), 'w'), indent=1)
    w = ['# Phase 15a step 1: own orbitals, Cr vs V (configuration-average HF; Eh, bohr, mEh for max|f_N|)',
         '| N | ion Cr / V | E Cr | E V | 3d <r> Cr / V | F0 Cr / V | F2 Cr / V (ratio) | F4 Cr / V (ratio) | '
         'max abs(f_N) Cr / V | Delta 11d, 12d, 13d at N: Cr / V | rho 11d, 12d, 13d (own set): Cr / V |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    ion = {2: 'Cr4+ / V3+', 3: 'Cr3+ / V2+', 4: 'Cr2+ / V+', 5: 'Cr+ / V'}
    for N in NS:
        c, v = js['Cr'][N], js['V'][N]
        w.append(f'| {N} | {ion[N]} | {c["E"]:.6f} | {v["E"]:.6f} | {c["r"]:.4f} / {v["r"]:.4f} | '
                 f'{c["F0"]:.5f} / {v["F0"]:.5f} | {c["F2"]:.5f} / {v["F2"]:.5f} ({c["F2"] / v["F2"]:.3f}) | '
                 f'{c["F4"]:.5f} / {v["F4"]:.5f} ({c["F4"] / v["F4"]:.3f}) | {c["fmax"]:.4f} / {v["fmax"]:.4f} | ' +
                 ', '.join(f'{c["D"][o]:.3f}' for o in A14.VIRT[:3]) + ' / ' +
                 ', '.join(f'{v["D"][o]:.3f}' for o in A14.VIRT[:3]) + ' | ' +
                 ', '.join(f'{c["rho"][o]:.3f}' for o in A14.VIRT[:3]) + ' / ' +
                 ', '.join(f'{v["rho"][o]:.3f}' for o in A14.VIRT[:3]) + ' |')
    for el in ('Cr', 'V'):
        D = [js[el][N]['D']['11d'] for N in NS]
        w.append(f'- {el} own-sequence Delta(11d) N = 2..5: ' + ' / '.join(f'{x:.3f}' for x in D) +
                 '; effective rho (Delta(N+1) - Delta(N)) / Delta(3): ' +
                 ' / '.join(f'{(D[i + 1] - D[i]) / D[1]:+.3f}' for i in range(3)))
    w.append(f'\nWall time {time.time() - t0:.0f} s.')
    open(os.path.join(OUT, 'phase15a_orb.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


def write_be_no(C, tag):
    rec = records(os.path.join(ROOT, 'runs', 'vq_orb', 'be.NO'))
    nl = len(NBF)
    work = os.path.join(ROOT, 'runs', f'vq_orb_{tag}')
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, 'be.NO'), 'wb') as fo:
        for s in (0, 1):
            hdr = rec[s * (1 + nl)]
            fo.write(struct.pack('<i', len(hdr)) + hdr + struct.pack('<i', len(hdr)))
            for l in range(nl):
                r = struct.pack('<i', l) + C[l].T.astype('<f8').tobytes()
                fo.write(struct.pack('<i', len(r)) + r + struct.pack('<i', len(r)))
    chk = read_vectors(os.path.join(work, 'be.NO'), NBF)
    assert all(np.allclose(chk[k][l], C[l]) for k in (0, 1) for l in range(nl))


def prep():
    z = np.load(os.path.join(OUT, 'orbsets_15a.npz'))
    n = 0
    for N in NS:
        tag = TAG_CR[N]
        write_be_no({l: z[f'{N}|{l}'] for l in range(len(NBF))}, tag)
        for src, dst in P14.run_names(N, tag):
            os.makedirs(os.path.join(ROOT, 'inputs', dst), exist_ok=True)
            deck = open(os.path.join(ROOT, 'inputs', src, 'be.deck')).read()
            deck = deck.replace(src, dst) if src.startswith('vcm') else deck.replace('_ivo_', f'_{tag}_')
            assert deck.count('z=23,') == 1 and deck.count('FILE=v_qzp.bas') == 1, src
            deck = deck.replace('z=23,', 'z=24,').replace('FILE=v_qzp.bas', 'FILE=cr.bas')
            open(os.path.join(ROOT, 'inputs', dst, 'be.deck'), 'w').write(deck)
            shutil.copy(os.path.join(ROOT, 'inputs', src, 'be.csf'), os.path.join(ROOT, 'inputs', dst))
            n += 1
    print(f'{n} decks')


def run():
    t0 = time.time()
    for N in NS:
        tag = TAG_CR[N]
        for src, dst in P14.run_names(N, tag):
            if os.path.isdir(os.path.join(ROOT, 'runs', dst)):
                continue
            subprocess.run([sys.executable, _C.script('run_atomci.py'), dst,
                            '--vectors-from', f'vq_orb_{tag}'], check=True, stdout=subprocess.DEVNULL)
        print(f'{tag} runs {time.time() - t0:.0f} s', flush=True)


def hpp_vec(sp, N, tag):
    by = {}
    for b in P14.sector_blocks(sp, N, tag):
        by[(b['L'], b['twoS'])] = b['U'].T @ b['H'][np.ix_(b['P'], b['P'])] @ b['U']
    return B.vec_from_blocks(N, by)


def analyze():
    t0 = time.time()
    res = {}
    for el, tags in (('Cr', TAG_CR), ('V', TAG_V)):
        P14.OWN.clear()
        P14.OWN.update(tags)
        for sp in SPACES:
            for N in NS:
                x, dg = P14.sector(sp, N, tags[N])
                dec = P14.decompose(N, x)
                d1 = P14.decompose(N, hpp_vec(sp, N, tags[N]))
                out = dict(diag=dg, dec=dec, e1=d1['c'])
                if N >= 3:
                    out['ut4'] = P14.ut4(dec['c'])
                if N >= 3:
                    s = D14.sector(sp, N)
                    out['appr'] = dict(pred=s['pred'], iters=s['iters'], conv=s['conv'], conv_dev=s['conv_dev'])
                res[f'{el} {sp} {N}'] = out
            print(f'{el} {sp} {time.time() - t0:.0f} s', flush=True)
    P14.OWN.clear()
    P14.OWN.update(TAG_V)
    json.dump(res, open(os.path.join(OUT, 'phase15a.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    w = ['# Phase 15a steps 2-4: sector H_eff, approximations, parameters; Cr vs V']
    w.append('\n## Step 2. Exact H_eff: min P weight, lowest gap (eV), states below (F = flagged); |e\'|, |t|, |f| '
             '(mEh); f/t; t outside span{u,t4} (d3)')
    w.append('| space | N | el | P, gap, below | abs(e\') | abs(t) | abs(f) | f/t | t out of span (d3) |')
    w.append('|---|---|---|---|---|---|---|---|---|')
    for sp in SPACES:
        for N in NS:
            for el in ('Cr', 'V'):
                r = res[f'{el} {sp} {N}']
                d, n = r['diag'], r['dec']['n']
                ft = f'{n["f"] / n["t"]:.3f}' if N >= 4 and n['t'] > 0 else '-'
                w.append(f'| {sp} | {N} | {el} | {d["wP"]:.3f}, {d["gap"]:.1f}, {d["below"]}{" F" if d["flag"] else ""} | '
                         f'{1e3 * n["e"]:.2f} | {1e3 * n["t"]:.4f} | {1e3 * n["f"]:.4f} | {ft} | ' +
                         (f'{r["ut4"][2]:.3f}' if N == 3 else '-') + ' |')
    w.append('\n## Step 3. (c) / (c2 sym) / converged Bloch: t rel, f rel, f ratio, f angle; Bloch iterations, '
             'max |H_conv - H_exact| (Eh)')
    w.append('| space | N | el | (c) | (c2) | converged | iters, dev |')
    w.append('|---|---|---|---|---|---|---|')
    for sp in SPACES:
        for N in (3, 4, 5):
            for el in ('Cr', 'V'):
                a = res[f'{el} {sp} {N}']['appr']
                ex = res[f'{el} {sp} {N}']['dec']['n']

                def cell(k):
                    r = a['pred'][k]
                    t = f'{r["t"]["rel"]:.4f}'
                    f = (f'{r["f"]["rel"]:.4f} / {r["f"].get("ratio", 0):.4f} / {r["f"].get("ang", 0):.2f}'
                         if N >= 4 and ex['f'] > 1e-12 else '-')
                    return f'{t} / {f}'
                fl = ' F' if res[f'{el} {sp} {N}']['diag']['flag'] else ''
                w.append(f'| {sp} | {N} | {el}{fl} | {cell("c")} | {cell("c2sym")} | {cell("conv")} | '
                         f'{a["iters"]}{"" if a["conv"] else " NOT CONV"}, {a["conv_dev"]:.1e} |')
    w.append('\n## Step 4. Parameters: e^(1) (H_PP), De = e - e^(1), t (mEh units of the 14b coefficient tables; '
             'e0 omitted); Cr/V ratios')
    for sp in SPACES:
        w.append(f'\n### {sp}')
        w.append('| coef | N | Cr e^(1) | V e^(1) | ratio | Cr De | V De | ratio | Cr total | V total | ratio |')
        w.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for k in B.EK[1:] + B.TK:
            for N in NS:
                if k in B.TK and N < 3:
                    continue
                c, v = res[f'Cr {sp} {N}'], res[f'V {sp} {N}']
                ce, ve = c['dec']['c'][k], v['dec']['c'][k]
                c1, v1 = c['e1'].get(k, 0.0), v['e1'].get(k, 0.0)
                rat = lambda a, b: f'{a / b:.3f}' if abs(b) > 1e-12 else '-'  # noqa: E731
                fl = ' F' if (c['diag']['flag'] or v['diag']['flag']) else ''
                w.append(f'| {k} | {N}{fl} | {1e3 * c1:.4f} | {1e3 * v1:.4f} | {rat(c1, v1)} | {1e3 * (ce - c1):.4f} | '
                         f'{1e3 * (ve - v1):.4f} | {rat(ce - c1, ve - v1)} | {1e3 * ce:.4f} | {1e3 * ve:.4f} | '
                         f'{rat(ce, ve)} |')
    w.append(f'\nWall time {time.time() - t0:.0f} s.')
    open(os.path.join(OUT, 'phase15a.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    {'orb': orb, 'prep': prep, 'run': run, 'analyze': analyze}[sys.argv[1]]()
