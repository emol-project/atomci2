#!/usr/bin/env python3
"""Phase 14a steps 1-2: orbital diagnostics and integral-only screening of common orbital sets for the d path.

Sets (scripts/phase14a_orb.py, average-of-configuration HF with the core re-optimized):
  o   original vq_orb set 2 (V2+ 3d3 4F MCSCF, core + 3d; virtual d = ATOMCI's 4F generalized-Fock virtuals)
  d3, a = d4, d = d5   per-N configuration-average HF
  b   ensemble d3-d4 (q = 3.5, pair = 4.5);  c  ensemble d3-d5 (q = 4, pair = 19/3)
Virtual d of a new set ("same recipe" as o): eigenvectors of h + closed shells + sum_k c_k (2 J^k - K^k)[3d], with c_k
the 3d self-coupling coefficients of the set's own functional (o: (2, -10/49, -48/441) for 4F reproduces the
original virtuals to 4e-10).
step1  3d (3s, 3p) <r>, <r^2>, orbital energy, overlaps, relaxation energies E_av(d^N, own) - E_av(d^N, o).
step2  frozen-core configuration averages with each set, N = 2..5:
         Delta_i(N) = E_av(3d^(N-1) i) - E_av(3d^N) = e'(i) - e'(3d) + (N-1)[U(3d,i) - U(3d,3d)]  (linear in N)
         delta_i = U(3d,i) - U(3d,3d), rho_i = |delta_i / Delta_i(3)|
         A: f_N(i) = <i| h' + (N-1)[J0 - 5/9 sum_{k=2,4} w J^k](3d) |3d>  (configuration-average Fock, off-diagonal)
         M proxy: R^2, R^4(3d3d; 3d i) (the integrals that carry the traceless coupling).
-> results/phase14/phase14a_orb.txt/.json, results/phase14/orbsets.npz
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase14a_orb as O  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase14')
SETS = {'d3': (3, 3), 'a (d4)': (4, 6), 'b (d3-d4)': (3.5, 4.5), 'c (d3-d5)': (4, 19 / 3), 'd (d5)': (5, 10)}
VIRT = ['11d', '12d', '13d', '14d', '15d']


def virtuals(bs, C, coef):
    """Replace the virtual d (columns 1:) by eigenvectors of the generalized Fock with (2J-K) self terms."""
    F, _, _ = O.fock(bs, C, 0, 0)
    P3 = bs.P(2, C[2][:, 0])
    Fv = F[2] + sum(c * (2 * bs.J(2, P3, k) - bs.K(2, P3, k)) for k, c in coef.items())
    S = bs.S[2]
    c3 = C[2][:, 0]
    # orthogonal complement of 3d (S metric), then diagonalize Fv there
    Pr = np.eye(len(c3)) - np.outer(c3, c3) @ S
    Xc = Pr @ C[2][:, 1:]
    s, U = np.linalg.eigh(Xc.T @ S @ Xc)
    Xo = Xc @ U / np.sqrt(s)
    e, v = np.linalg.eigh(Xo.T @ Fv @ Xo)
    out = {l: C[l].copy() for l in C}
    out[2] = np.column_stack([c3, Xo @ v])
    return out, e


def own_coef(q, pair):
    g = 2 * pair / q
    return {0: g, 2: -g * 5 / 9 * 2 / 35, 4: -g * 5 / 9 * 2 / 35}


def gaps(bs, C):
    """Configuration-average quantities from integrals (frozen core of the set)."""
    F, _, _ = O.fock(bs, C, 0, 0)
    hp = F[2]                                          # h + closed shells
    cs = C[2]
    P = [bs.P(2, cs[:, i]) for i in range(cs.shape[1])]
    e1 = [cs[:, i] @ hp @ cs[:, i] for i in range(cs.shape[1])]
    R = bs.R
    d = P[0]
    w2 = O.w3(2, 2, 2)
    U33 = R(d, d, d, d, 0) - 5 / 9 * w2 * (R(d, d, d, d, 2) + R(d, d, d, d, 4))
    out = {}
    for i, o in enumerate(VIRT, start=1):
        p = P[i]
        Uab = R(d, d, p, p, 0) - 0.5 * (O.w3(2, 0, 2) * R(d, p, d, p, 0) + w2 * (R(d, p, d, p, 2) + R(d, p, d, p, 4)))
        dl = Uab - U33
        D = {N: e1[i] - e1[0] + (N - 1) * dl for N in range(1, 6)}
        f = {N: float(cs[:, i] @ hp @ cs[:, 0] + (N - 1) * (R(p, d, d, d, 0) - 5 / 9 * w2 * (R(p, d, d, d, 2) +
                                                                                              R(p, d, d, d, 4))))
             for N in range(1, 6)}
        out[o] = dict(D=D, delta=dl, rho=abs(dl / D[3]), f=f, R2=R(d, d, d, p, 2), R4=R(d, d, d, p, 4),
                      R0=R(d, d, d, p, 0))
    return out


def radial_moments(bs, C, l, i):
    c = C[l][:, i]
    return float(c @ bs.mat[l]['r'] @ c), None


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bs = O.Basis()
    C0 = O.load_original()
    # <r^2> via grid
    r = bs.r

    def mom(C, l, i, p):
        P = bs.P(l, C[l][:, i])
        return float(np.sum(P * P * r ** p * bs.w))
    sets = {'o (V2+ 4F)': C0}
    info = {'o (V2+ 4F)': dict(q=3, pair=3)}
    for nm, (q, pair) in SETS.items():
        res = O.scf(bs, q, pair)
        Cv, ev = virtuals(bs, res['C'], own_coef(q, pair))
        sets[nm] = Cv
        info[nm] = dict(q=q, pair=pair, E=res['E'], eps=[float(x) for x in res['eps'][2][:1]] +
                        [float(res['eps'][0][2]), float(res['eps'][1][1])], it=res['it'], ev=ev.tolist())
        print(f'{nm} SCF E = {res["E"]:.8f} ({res["it"]} it) {time.time() - t0:.0f} s', flush=True)
    # check: recipe on the original set reproduces its virtuals
    F4 = {0: 2.0, 2: -10 / 49, 4: -48 / 441}
    Cr, _ = virtuals(bs, C0, F4)
    ov = np.abs(Cr[2][:, 1:].T @ bs.S[2] @ C0[2][:, 1:])
    rec_dev = float(np.abs(ov - np.eye(5)).max())

    w = []
    a = w.append
    js = {}
    a('# Phase 14a steps 1-2: orbital diagnostics and integral-only screening')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip())
    a(f'\nRecipe check: the generalized-Fock recipe with the 4F coefficients reproduces the original virtual d '
      f'(max |<new|orig>| - delta) = {rec_dev:.1e}.')

    a('\n## Step 1. Orbital diagnostics (core re-optimized in every new set)')
    a('| set | E_av(d^N) of its own functional (Eh) | 3d <r> | 3d <r^2> | eps_3d (own Fock) | <3d|3d(o)> | 3s <r> | '
      '<3s|3s(o)> | 3p <r> | <3p|3p(o)> |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    s1 = {}
    for nm, C in sets.items():
        q, pair = info[nm]['q'], info[nm]['pair']
        E = O.energy(bs, C, q, pair)
        F, _, _ = O.fock(bs, C, q, pair)
        eps = float(C[2][:, 0] @ F[2] @ C[2][:, 0])
        ov = lambda l, i: float(abs(C[l][:, i] @ bs.S[l] @ C0[l][:, i]))  # noqa: E731
        s1[nm] = dict(E=E, r=mom(C, 2, 0, 1), r2=mom(C, 2, 0, 2), eps=eps, ov3d=ov(2, 0), r3s=mom(C, 0, 2, 1),
                      ov3s=ov(0, 2), r3p=mom(C, 1, 1, 1), ov3p=ov(1, 1))
        x = s1[nm]
        a(f'| {nm} | {E:.6f} | {x["r"]:.4f} | {x["r2"]:.4f} | {eps:.4f} | {x["ov3d"]:.6f} | {x["r3s"]:.4f} | '
          f'{x["ov3s"]:.6f} | {x["r3p"]:.4f} | {x["ov3p"]:.6f} |')
    names = list(sets)
    a('\n3d overlaps |<3d(X)|3d(Y)>|:')
    a('| | ' + ' | '.join(names) + ' |')
    a('|---|' + '---|' * len(names))
    for x in names:
        a(f'| {x} | ' + ' | '.join(f'{abs(sets[x][2][:, 0] @ bs.S[2] @ sets[y][2][:, 0]):.6f}' for y in names) + ' |')
    a('\nRelaxation energies E_av(d^N, set) - E_av(d^N, o) (mEh; whole atom, core relaxed):')
    a('| set | N = 3 | N = 4 | N = 5 |')
    a('|---|---|---|---|')
    rel = {}
    for nm, C in sets.items():
        rel[nm] = {N: 1e3 * (O.energy(bs, C, N, N * (N - 1) / 2) - O.energy(bs, C0, N, N * (N - 1) / 2))
                   for N in (3, 4, 5)}
        a(f'| {nm} | ' + ' | '.join(f'{rel[nm][N]:+.3f}' for N in (3, 4, 5)) + ' |')
    js['step1'] = dict(s1=s1, rel=rel, info=info, rec_dev=rec_dev)
    print(f'step1 {time.time() - t0:.0f} s', flush=True)

    a('\n## Step 2. Configuration-average gaps, rho and A from integrals (frozen core of each set)')
    a('Delta in Eh; rho = |delta/Delta(3)|; |A|_N = sqrt(sum_i f_N(i)^2) over 11d-15d (s6d) and 11d-12d (s3d), mEh; '
      'M proxy = sqrt(sum_i R^2(3d3d;3d i)^2 + R^4(...)^2), mEh.')
    a('| set | Delta(3) 11d / 12d / 13d | rho 11d / 12d / 13d | |A| s6d N=3 / 4 / 5 | |A| s3d N=3 / 4 / 5 | '
      'M proxy s6d / s3d | Delta(N) 11d, N = 2..5 |')
    a('|---|---|---|---|---|---|---|')
    s2 = {}
    for nm, C in sets.items():
        g = gaps(bs, C)
        A6 = {N: 1e3 * np.sqrt(sum(g[o]['f'][N] ** 2 for o in VIRT)) for N in (3, 4, 5)}
        A3 = {N: 1e3 * np.sqrt(sum(g[o]['f'][N] ** 2 for o in VIRT[:2])) for N in (3, 4, 5)}
        M6 = 1e3 * np.sqrt(sum(g[o]['R2'] ** 2 + g[o]['R4'] ** 2 for o in VIRT))
        M3 = 1e3 * np.sqrt(sum(g[o]['R2'] ** 2 + g[o]['R4'] ** 2 for o in VIRT[:2]))
        s2[nm] = dict(g=g, A6=A6, A3=A3, M6=M6, M3=M3)
        a(f'| {nm} | ' + ' / '.join(f'{g[o]["D"][3]:.4f}' for o in VIRT[:3]) + ' | ' +
          ' / '.join(f'{g[o]["rho"]:.3f}' for o in VIRT[:3]) + ' | ' + ' / '.join(f'{A6[N]:.1f}' for N in (3, 4, 5)) +
          ' | ' + ' / '.join(f'{A3[N]:.1f}' for N in (3, 4, 5)) + f' | {M6:.1f} / {M3:.1f} | ' +
          ' / '.join(f'{g["11d"]["D"][N]:.4f}' for N in range(2, 6)) + ' |')
    a('\nPer-orbital detail (Delta(3), delta, rho, f_N(i) N = 3 / 4 / 5 in mEh):')
    for nm in sets:
        g = s2[nm]['g']
        a(f'- {nm}: ' + '; '.join(f'{o} {g[o]["D"][3]:.3f} {g[o]["delta"]:+.4f} {g[o]["rho"]:.3f} ' +
                                  '/'.join(f'{1e3 * g[o]["f"][N]:+.1f}' for N in (3, 4, 5)) for o in VIRT))
    js['step2'] = s2
    np.savez(os.path.join(OUT, 'orbsets.npz'), **{f'{nm}|{l}': C[l] for nm, C in sets.items() for l in C})
    a(f'\nWall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase14a_orb.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    open(os.path.join(OUT, 'phase14a_orb.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
