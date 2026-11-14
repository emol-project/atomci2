#!/usr/bin/env python3
"""Phase 14d: where the remaining ~10% after the Sylvester sum (14c (c)) comes from.

Own-orbital sectors N = 2, 3, 4 (14b runs), spaces s3d, s6d, 3d, core, g. Partition (as 13h/14c): H0' = PHP + QHQ
(T in both blocks), V = PHQ + QHP; so H_PP = PHP and H_QQ = QHQ exactly and only the raw sector blocks are needed.
step1  Bloch first order: H_QQ X - X H_PP = -V_QP; H_B = H_PP + V_PQ X. Its symmetric part is (c) (canonical
       Van Vleck second order); the anti-Hermitian part is reported.
step2  Bloch equation H_QQ X - X H_PP = -V_QP + X V_PQ X: (c2) one more iteration from X1 (symmetric part, as (c);
       and the des Cloizeaux form (1 + X+X)^1/2 H_B (1 + X+X)^-1/2); iterated to convergence (des Cloizeaux),
       compared with the exact H_eff.
step3  intermediate Hamiltonian: H_QQ eigenstates per (L,S) block; Q_I = the n lowest, Q_O folded into P by the
       Van Vleck / Sylvester second-order term; des Cloizeaux H_eff of the dim(P) states of largest P weight.
Parts (e0, e', t, f) and metric as in 14b/14c.
-> results/phase14/phase14d.txt/.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_sylvester

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
import phase13b as B  # noqa: E402
import phase13f as F  # noqa: E402
import phase14b as P14  # noqa: E402
import phase14c as C14  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase14')
SPACES = ['s3d', 's6d', '3d', 'core', 'g']
NS = (2, 3, 4)
NSCAN = [0, 5, 10, 20, 50, 100, None]


def sqrtm_sym(A, p):
    w, U = np.linalg.eigh(A)
    return (U * w ** p) @ U.T


def dcl(HB, X):
    Sm = np.eye(HB.shape[0]) + X.T @ X
    return sqrtm_sym(Sm, 0.5) @ HB @ sqrtm_sym(Sm, -0.5)


def sym(A):
    return 0.5 * (A + A.T)


def bloch_solve(HPP, HQQ, V, rhs_extra=None):
    rhs = -V.T if rhs_extra is None else -V.T + rhs_extra
    return solve_sylvester(HQQ, -HPP, rhs)


def fold_vv(HPP, eO, VO):
    """Van Vleck second order of V_PO with H_OO = diag(eO): 1/2 (Y VO+ + VO Y+), H_PP Y - Y diag(eO) = VO."""
    if VO.shape[1] == 0:
        return np.zeros_like(HPP)
    Y = solve_sylvester(HPP, -np.diag(eO), VO)
    return 0.5 * (Y @ VO.T + VO @ Y.T)


def compare(N, x, pe, exn):
    pp, _ = C14.parts(N, x)
    r = {}
    for g in ('e0', 'e', 't', 'f'):
        d = pp[g] - pe[g]
        r[g] = dict(n=C14.nrm(N, pp[g]), dev=C14.nrm(N, d), rel=C14.nrm(N, d) / max(exn[g], 1e-300))
        if g in ('t', 'f') and exn[g] > 1e-12 and C14.nrm(N, pp[g]) > 0:
            cos = S.ip(N, pp[g], pe[g]) / (C14.nrm(N, pp[g]) * exn[g])
            r[g]['ratio'] = C14.nrm(N, pp[g]) / exn[g]
            r[g]['ang'] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return r


def sector(sp, N):
    bl = P14.sector_blocks(sp, N, P14.OWN[N])
    mats = {k: {} for k in ('exact', 'c', 'c2sym', 'c2dc', 'conv')}
    anti2, gsum, dP, dPmax, itn, conv_dev, conv_ok = 0.0, 0.0, 0.0, 0.0, 0, 0.0, True
    chk_vv = 0.0
    scan = {n: {} for n in NSCAN}
    scan_wP = {n: 1.0 for n in NSCAN}
    scan_dim = {n: 0 for n in NSCAN}
    for b in bl:
        P, Q, H, U = b['P'], b['Q'], b['H'], b['U']
        key = (b['L'], b['twoS'])
        g = (2 * b['L'] + 1) * (b['twoS'] + 1)
        Ex, _ = F.heff_block(b, H)
        mats['exact'][key] = Ex
        HPP, HQQ, V = H[np.ix_(P, P)], H[np.ix_(Q, Q)], H[np.ix_(P, Q)]
        if not Q:                                   # no Q in this block: every form is H_PP
            for k in ('c', 'c2sym', 'c2dc', 'conv'):
                mats[k][key] = U.T @ HPP @ U
            for n in NSCAN:
                scan[n][key] = U.T @ HPP @ U
            gsum += g * len(P)
            continue
        X1 = bloch_solve(HPP, HQQ, V)
        HB1 = HPP + V @ X1
        mats['c'][key] = U.T @ sym(HB1) @ U
        # canonical Van Vleck in eigenbases, as a check of (c)
        a, A = np.linalg.eigh(HPP)
        bq, Bq = np.linalg.eigh(HQQ)
        Vt = A.T @ V @ Bq
        Dn = 1.0 / (a[:, None] - bq[None, :])
        VV = 0.5 * ((Vt * Dn) @ Vt.T + Vt @ (Vt * Dn).T)
        chk_vv = max(chk_vv, float(np.abs(A @ (np.diag(a) + VV) @ A.T - sym(HB1)).max()))
        anti = U.T @ (0.5 * (HB1 - HB1.T)) @ U
        anti2 += g * float(np.sum(anti ** 2))
        XtX = X1.T @ X1
        dP += g * float(np.trace(XtX))
        gsum += g * len(P)
        dPmax = max(dPmax, float(np.linalg.eigvalsh(XtX).max()))
        X2 = bloch_solve(HPP, HQQ, V, X1 @ V @ X1)
        HB2 = HPP + V @ X2
        mats['c2sym'][key] = U.T @ sym(HB2) @ U
        mats['c2dc'][key] = U.T @ sym(dcl(HB2, X2)) @ U
        X = X1
        for it in range(1, 500):
            Xn = bloch_solve(HPP, HQQ, V, X @ V @ X)
            dx = float(np.abs(Xn - X).max())
            X = Xn
            if dx < 1e-13:
                break
            if not np.isfinite(dx) or dx > 1e3:
                conv_ok = False
                break
        itn = max(itn, it)
        if dx >= 1e-13:
            conv_ok = False
        Hc = U.T @ sym(dcl(HPP + V @ X, X)) @ U
        mats['conv'][key] = Hc
        conv_dev = max(conv_dev, float(np.abs(Hc - Ex).max()))
        # intermediate Hamiltonian scan
        VW = V @ Bq
        for n in NSCAN:
            nI = len(Q) if n is None else min(n, len(Q))
            H2 = fold_vv(HPP, bq[nI:], VW[:, nI:])
            M = np.zeros((len(P) + nI,) * 2)
            M[:len(P), :len(P)] = HPP + H2
            M[:len(P), len(P):] = VW[:, :nI]
            M[len(P):, :len(P)] = VW[:, :nI].T
            M[len(P):, len(P):] = np.diag(bq[:nI])
            fb = dict(P=list(range(len(P))), Q=list(range(len(P), len(P) + nI)), U=U, m=b['m'])
            Bk, inf = F.heff_block(fb, M)
            scan[n][key] = Bk
            scan_wP[n] = min(scan_wP[n], inf['wP'])
            scan_dim[n] = max(scan_dim[n], nI)
    xe = B.vec_from_blocks(N, mats['exact'])
    pe, _ = C14.parts(N, xe)
    exn = {g: C14.nrm(N, v) for g, v in pe.items()}
    out = dict(ex=exn, anti=float(np.sqrt(anti2)), dP_avg=dP / gsum, dP_max=dPmax, iters=itn, conv=conv_ok,
               conv_dev=conv_dev, chk_vv=chk_vv, nQmax=max(len(b['Q']) for b in bl), pred={}, scan={})
    for k in ('c', 'c2sym', 'c2dc', 'conv'):
        out['pred'][k] = compare(N, B.vec_from_blocks(N, mats[k]), pe, exn)
    for n in NSCAN:
        out['scan'][str(n)] = dict(r=compare(N, B.vec_from_blocks(N, scan[n]), pe, exn), wP=scan_wP[n],
                                   nI=scan_dim[n])
    # n = 0 must equal (c); n = all must equal exact
    xc = B.vec_from_blocks(N, mats['c'])
    out['chk_n0'] = float(np.abs(B.vec_from_blocks(N, scan[0]) - xc).max())
    out['chk_all'] = float(np.abs(B.vec_from_blocks(N, scan[None]) - xe).max())
    return out


def fmt_r(r, ex):
    t = f'{r["t"]["rel"]:.4f}' if ex['t'] > 1e-12 else '-'
    if ex['f'] > 1e-12:
        f = f'{r["f"]["rel"]:.4f} / {r["f"].get("ratio", 0):.4f} / {r["f"].get("ang", 0):.2f}'
    else:
        f = '- / - / -'
    return f'{t} / {f}'


def main():
    t0 = time.time()
    R = {}
    for sp in SPACES:
        for N in NS:
            R[(sp, N)] = sector(sp, N)
        print(f'{sp} {time.time() - t0:.0f} s', flush=True)
    w = []
    a = w.append
    a('# Phase 14d: residual after the Sylvester sum; Bloch, (c2), converged Bloch, intermediate Hamiltonian')
    a(open(__file__).read().split('"""')[1].split('-> results')[0].strip())
    a('\n## Checks')
    a('| space | N | max nQ per block | (c) vs canonical Van Vleck (Eh) | Bloch iterations | converged | converged dC vs '
      'exact (Eh) | IH n=0 vs (c) | IH n=all vs exact |')
    a('|---|---|---|---|---|---|---|---|---|')
    for (sp, N), x in R.items():
        a(f'| {sp} | {N} | {x["nQmax"]} | {x["chk_vv"]:.1e} | {x["iters"]} | {x["conv"]} | {x["conv_dev"]:.1e} | '
          f'{x["chk_n0"]:.1e} | {x["chk_all"]:.1e} |')
    a('\n## Step 1. Bloch first order: anti-Hermitian part and P-weight deficit')
    a('|anti| = weighted Frobenius norm of the anti-Hermitian part of H_B (seniority blocks, weight (2L+1)(2S+1)), '
      'same metric as |t|, |f|. P deficit = tr(X1+ X1) per P state (degeneracy-weighted mean) and the largest '
      'eigenvalue of X1+ X1.')
    a('| space | N | abs(anti) (mEh) | abs(t) | abs(f) | anti/abs(t) | anti/abs(f) | P deficit mean | max |')
    a('|---|---|---|---|---|---|---|---|---|')
    for (sp, N), x in R.items():
        ex = x['ex']
        a(f'| {sp} | {N} | {1e3 * x["anti"]:.4f} | {1e3 * ex["t"]:.4f} | {1e3 * ex["f"]:.4f} | ' +
          (f'{x["anti"] / ex["t"]:.3f}' if ex['t'] > 1e-12 else '-') + ' | ' +
          (f'{x["anti"] / ex["f"]:.3f}' if ex['f'] > 1e-12 else '-') + f' | {x["dP_avg"]:.5f} | {x["dP_max"]:.5f} |')
    a('\n## Step 2. (c), (c2) and converged Bloch vs exact; cells: t rel / f rel / f ratio / f angle (deg)')
    a('| space | N | (c) | (c2) symmetric part | (c2) des Cloizeaux | converged (dC) |')
    a('|---|---|---|---|---|---|')
    for (sp, N), x in R.items():
        if N < 3:
            continue
        a(f'| {sp} | {N} | ' + ' | '.join(fmt_r(x['pred'][k], x['ex']) for k in
                                         ('c', 'c2sym', 'c2dc', 'conv')) + ' |')
    a('\n## Step 3. Intermediate Hamiltonian scan (n = lowest H_QQ eigenstates per (L,S) block kept in Q_I; '
      'None = all). Cells: t rel / f rel / min P weight')
    a('| space | N | ' + ' | '.join(f'n={n}' for n in NSCAN) + ' |')
    a('|---|---|' + '---|' * len(NSCAN))
    for (sp, N), x in R.items():
        if N < 3:
            continue
        cells = []
        for n in NSCAN:
            s = x['scan'][str(n)]
            r = s['r']
            t = f'{r["t"]["rel"]:.4f}' if x['ex']['t'] > 1e-12 else '-'
            f = f'{r["f"]["rel"]:.4f}' if x['ex']['f'] > 1e-12 else '-'
            cells.append(f'{t} / {f} / {s["wP"]:.3f} (nI {s["nI"]})')
        a(f'| {sp} | {N} | ' + ' | '.join(cells) + ' |')
    a(f'\nWall time {time.time() - t0:.0f} s.')
    js = {f'{sp} {N}': x for (sp, N), x in R.items()}
    json.dump(js, open(os.path.join(OUT, 'phase14d.json'), 'w'), indent=1,
              default=lambda v: float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v)))
    open(os.path.join(OUT, 'phase14d.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
