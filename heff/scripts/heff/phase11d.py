#!/usr/bin/env python3
"""Phase 11d: the gap between second-order closure (B2) and CI(P + Q) for 3d -> nd and 3d -> ng.

Same P, Q, runs and reference as Phase 11c (vq_<ion>_<term>_ivo exports, QZP, [Ar] frozen); exports only, no tapes.
delta = R(X) - R(A), A = CI(P). Per path:
 1. Q-Q coupling hierarchy, each solved exactly (CI in P + Q, des Cloizeaux on P):
    E_diag (H_QQ -> its diagonal), E_blk (H_QQ block-diagonal by virtual orbital), E (full),
    E_noPrel (E with H_PP -> E_av(3d^n) 1: no P term splitting in the denominators).
 2. Order by order (heff.en_series after a block rotation that makes H0 diagonal), orders 2..6:
    EN: H0 = PHP + diag(QHQ);  BLK: H0 = PHP + QHQ_blk (orbital blocks).
    2D: the non-Hermitian Bloch H_eff is made Hermitian by des Cloizeaux with the right eigenvectors x_k
    normalized by |Omega x_k| (Omega = P + chi to the matching order), as for CI.
 3. Both paths together: CI(P + Q_d + Q_g) vs delta(E_d) + delta(E_g).
Table for n=3: Delta_i, R^2_i, R^4_i (Phase 11c fits) and the unit-integral angular vectors A22, A24, A44
(from phase11c.json, 9 components) with closed forms identified as rational x sqrt(k).

Usage: phase11d.py   -> results/phase11/phase11d.txt/.json
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import phase9 as P  # noqa: E402
from phase11c import TERMS, decomp, heff_E, pack, rms, term_data  # noqa: E402
from atomci_heff import cfp_l  # noqa: E402
from atomci_heff.heff import en_series  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase11')
COMP = ['4F', '4P', '2H', '2G', '2F', '2P', '2D 11', '2D 33', '2D 13']
SQF = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 21, 22, 30, 33, 35, 42, 55, 66, 70, 77, 105, 110, 154, 165, 210, 231,
       330, 385, 462, 770, 1155, 2310]


def _smooth_numbers(nmax=10 ** 8, primes=(2, 3, 5, 7, 11, 13)):
    out = [1]
    for p in primes:
        out = [m * p ** e for m in out for e in range(40) if m * p ** e <= nmax]
    return sorted(set(out))


SMOOTH = _smooth_numbers()


def closed(x, diag, tol=1e-12):
    """x = (N / D) sqrt(k), smallest D with prime factors <= 13; k = 1 for diagonal components."""
    if abs(x) < 1e-13:
        return '0'
    best = None
    for k in ([1] if diag else SQF):
        y = x / np.sqrt(k)
        for D in SMOOTH:
            N = round(y * D)
            if N and abs(N / D * np.sqrt(k) - x) < tol:
                if best is None or D < best[1]:
                    best = (N, D, k)
                break
    if best is None:
        return None
    N, D, k = best
    f = Fraction(N, D)
    return f'{f}' + ('' if k == 1 else f' sqrt({k})')


def hermit(M, chis, nP):
    """des Cloizeaux form of a (non-Hermitian) Bloch H_eff, eigenvectors normalized with Omega = P + sum chis."""
    if nP == 1:
        return M
    e, X = np.linalg.eig(M)
    if np.abs(e.imag).max() > 1e-12:
        return None
    e, X = e.real, X.real
    Om = np.vstack([np.eye(nP), sum(chis) if chis else np.zeros((0, nP))])
    X = X / np.linalg.norm(Om @ X, axis=0)[None, :]
    u, s, vt = np.linalg.svd(X)
    Xt = u @ vt
    return Xt @ np.diag(e) @ Xt.T


def series(Hs, nP, groups, nmax=16):
    """Partial sums 2..nmax of the RS series with H0 = PHP + block-diag(QHQ over groups) (groups of Q positions,
    relative to the Q part). Returns {order: H_eff (Hermitian, original P basis) or None}."""
    Hpp = Hs[:nP, :nP]
    _, Up = np.linalg.eigh(Hpp)
    Hqq = Hs[nP:, nP:]
    blocks = []
    for g in groups:
        _, Ug = np.linalg.eigh(Hqq[np.ix_(g, g)])
        blocks.append((g, Ug))
    Uq = np.zeros_like(Hqq)
    for g, Ug in blocks:
        Uq[np.ix_(g, g)] = Ug
    T = block_diag(Up, Uq)
    Hr = T.T @ Hs @ T
    terms, chis = en_series(Hr, list(range(nP)), nmax)
    out, acc = {}, terms[0] + terms[1]
    for n in range(2, nmax + 1):
        acc = acc + terms[n]
        h = hermit(acc, chis[1:n], nP)
        out[n] = None if h is None else Up @ h @ Up.T
    return out


def main():
    j11c = json.load(open(os.path.join(OUT, 'phase11c.json')))
    virt = {X: sorted(j11c['paths'][X]['labels'], key=lambda o: int(o[:-1])) for X in ('d', 'g')}
    w = []
    a = w.append
    a('# Phase 11d: the gap between closure (B2) and CI(P + Q) for 3d -> nd and 3d -> ng; V QZP, [Ar] frozen; mEh')
    a(f'Runs vq_<ion>_<term>_ivo (exports only). Virtual orbitals: d {virt["d"]}, g {virt["g"]}.')
    T = {(n, t): term_data(n, t, virt) for n, t in TERMS}

    def blk(d, rows, cols):
        return d['H'][np.ix_([d['pos'][i] for i in rows], [d['pos'][i] for i in cols])]

    Eav = {}
    for n in (2, 3):
        tl = [(m, t) for m, t in TERMS if m == n]
        Eav[n] = sum(P.g(k[1]) * np.trace(blk(T[k], T[k]['Pi'], T[k]['Pi'])) for k in tl) / \
            sum(P.g(t) * len(T[(m, t)]['Pi']) for m, t in tl)
    EA = {(n, t): pack(blk(d, d['Pi'], d['Pi']), n == 3 and t == '2D') for (n, t), d in T.items()}
    d2a = {t: EA[(2, t)]['E'] - sum(P.g(x) * EA[(2, x)]['E'] for x in P.T2) / 45 for t in P.T2}
    Ha = EA[(3, '2D')]['Heff']
    Eav3a = (sum(P.g(t) * EA[(3, t)]['E'] for t in P.T3) + 10 * np.trace(Ha)) / 120
    U, mis = P.seniority_map(Ha, Eav3a * np.eye(2) + P.pairsum(d2a)['2D'])
    assert mis < 1e-8, mis
    RA, _ = decomp(EA, U)
    a(f'Checks: rms R(A) {1e3 * rms(RA):.1e} mEh; E_av(3d^n) vs 11c: '
      + ', '.join(f'n={n} {abs(Eav[n] - j11c["Eav"][str(n)]):.1e}' for n in (2, 3)) + ' Eh')

    def vec(Ed):
        return decomp(Ed, U)[0] - RA

    def stats(r, dE):
        return dict(rms=rms(r), angle=P.wangle(r, dE), coef=P.ip(r, dE) / P.ip(dE, dE))

    def mat(d, orbs, mode, n):
        Qa = [i for o in orbs for i in d['Q'][o]]
        Hs = blk(d, d['Pi'] + Qa, d['Pi'] + Qa).copy()
        nP = len(d['Pi'])
        if mode in ('diag', 'blk'):
            Hq = Hs[nP:, nP:]
            keep = np.eye(len(Qa), dtype=bool)
            if mode == 'blk':
                lab = np.concatenate([[k] * len(d['Q'][o]) for k, o in enumerate(orbs)])
                keep = lab[:, None] == lab[None, :]
            Hs[nP:, nP:] = np.where(keep, Hq, 0.0)
        if mode == 'noPrel':
            Hs[:nP, :nP] = Eav[n] * np.eye(nP)
        return Hs, nP

    js = dict(paths={})
    dEX = {}
    for X in ('d', 'g'):
        orbs = virt[X]
        a(f'\n# Path 3d -> n{X} ({len(orbs)} orbitals)')
        Ev = {}
        for mode in ('diag', 'blk', 'full', 'noPrel'):
            Ev[mode] = {k: heff_E(*mat(d, orbs, mode, k[0]), k == (3, '2D')) for k, d in T.items()}
        R = {m: vec(v) for m, v in Ev.items()}
        dE = R['full']
        dEX[X] = dE
        RB2 = np.asarray(j11c['paths'][X]['R']['B2'])
        chkE = float(np.abs(dE - np.asarray(j11c['paths'][X]['R']['E'])).max())
        a(f'Check: delta(E) vs Phase 11c {chkE:.1e} Eh; rms delta(E) {1e3 * rms(dE):.4f} mEh')
        a('\n## 1. Q-Q coupling hierarchy (each exact in its P + Q space)')
        a('| X | rms | angle to delta(E) | coefficient on delta(E) | rms (delta - delta(E)) |')
        a('|---|---|---|---|---|')
        res = {}
        rows = [('B2 (11c)', RB2), ('E_diag', R['diag']), ('E_blk', R['blk']), ('E', R['full']), ('E_noPrel', R['noPrel'])]
        for nm, r in rows:
            s = stats(r, dE)
            s['rms_diff'] = rms(r - dE)
            res[nm] = s
            a(f'| {nm} | {1e3 * s["rms"]:.4f} | {s["angle"]:.1f} | {s["coef"]:.3f} | {1e3 * s["rms_diff"]:.4f} |')
        # energies: correlation contribution per term, mEh (E(X) - E(A)), lowest d^3 2D root for the block
        a('\nPath energy E(X) - E(A) per term, mEh (d3 2D: both roots of the des Cloizeaux block)')
        a('| term | E_diag | E_blk | E | E_noPrel (H_eff - E_av + H_PP) |')
        a('|---|---|---|---|---|')
        for (n, t) in TERMS:
            def ev(m):
                r = Ev[m][(n, t)]
                return np.linalg.eigvalsh(r['Heff']) if 'Heff' in r else np.array([r['E']])
            base = (np.linalg.eigvalsh(EA[(n, t)]['Heff']) if 'Heff' in EA[(n, t)] else np.array([EA[(n, t)]['E']]))
            cells = [ev(m) - base for m in ('diag', 'blk', 'full')]
            r = Ev['noPrel'][(n, t)]
            Hp = blk(T[(n, t)], T[(n, t)]['Pi'], T[(n, t)]['Pi'])
            M = (r['Heff'] if 'Heff' in r else np.array([[r['E']]])) - Eav[n] * np.eye(len(Hp)) + 0.5 * (Hp + Hp.T)
            cells.append(np.linalg.eigvalsh(M) - base)
            a(f'| d{n} {t} | ' + ' | '.join('/'.join(f'{1e3 * x:+.3f}' for x in c) for c in cells) + ' |')

        # Q-Q coupling magnitudes (n=3, all terms)
        within, between, dspread = [], [], []
        for (n, t), d in T.items():
            if n != 3:
                continue
            Qa = [i for o in orbs for i in d['Q'][o]]
            lab = np.concatenate([[k] * len(d['Q'][o]) for k, o in enumerate(orbs)])
            Hq = blk(d, Qa, Qa)
            off = ~np.eye(len(Qa), dtype=bool)
            within += list(np.abs(Hq[off & (lab[:, None] == lab[None, :])]))
            between += list(np.abs(Hq[lab[:, None] != lab[None, :]]))
        a(f'\nQ-Q couplings, d3 terms (Eh): within one orbital max |H| {max(within, default=0):.4f}, '
          f'rms {np.sqrt(np.mean(np.square(within))) if within else 0:.4f}; between orbitals max |H| {max(between):.4f}, '
          f'rms {np.sqrt(np.mean(np.square(between))):.4f}')
        res['couplings'] = dict(within_max=max(within, default=0.0), between_max=max(between),
                                between_rms=float(np.sqrt(np.mean(np.square(between)))))

        a('\n## 2. Order by order (partial sums, orders 2..6; diagnostic to 16)')
        ser = {}
        for nm in ('EN', 'BLK'):
            S = {}
            for k, d in T.items():
                Hs, nP = mat(d, orbs, 'full', k[0])
                if nm == 'EN':
                    groups = [[q] for q in range(Hs.shape[0] - nP)]
                else:
                    groups, off = [], 0
                    for o in orbs:
                        groups.append(list(range(off, off + len(d['Q'][o]))))
                        off += len(d['Q'][o])
                S[k] = series(Hs, nP, groups)
            ser[nm] = {}
            a(f'\n### H0 = PHP + {"diag(QHQ) (EN)" if nm == "EN" else "QHQ_blk (orbital blocks)"}')
            a('| order | rms | angle to delta(E) | coefficient on delta(E) | rms (delta - delta(E)) | max |dE| over terms vs E (mEh) |')
            a('|---|---|---|---|---|---|')
            for order in range(2, 17):
                if any(S[k][order] is None for k in S):
                    a(f'| {order} | complex eigenvalues in the 2D block | | | | |')
                    ser[nm][order] = None
                    continue
                Ed = {k: pack(S[k][order], k == (3, '2D')) for k in S}
                r = vec(Ed)
                s = stats(r, dE)
                s['rms_diff'] = rms(r - dE)
                emax = max(float(np.abs(np.linalg.eigvalsh(Ed[k]['Heff']) - np.linalg.eigvalsh(Ev['full'][k]['Heff'])).max())
                           if 'Heff' in Ed[k] else abs(Ed[k]['E'] - Ev['full'][k]['E']) for k in S)
                s['emax'] = emax
                s['worst'] = max(S, key=lambda k: float(np.abs(np.linalg.eigvalsh(Ed[k]['Heff']) - np.linalg.eigvalsh(Ev['full'][k]['Heff'])).max())
                                 if 'Heff' in Ed[k] else abs(Ed[k]['E'] - Ev['full'][k]['E']))
                s['worst'] = f'd{s["worst"][0]} {s["worst"][1]}'
                ser[nm][order] = s
                if order <= 6:
                    a(f'| {order} | {1e3 * s["rms"]:.4f} | {s["angle"]:.1f} | {s["coef"]:.3f} | {1e3 * s["rms_diff"]:.4f} | '
                      f'{1e3 * emax:.4f} |')
            a('Convergence diagnostic, orders 7..16: coefficient / max |dE| (mEh): ' + '; '.join(
                f'{o} {ser[nm][o]["coef"]:.4f}/{1e3 * ser[nm][o]["emax"]:.4f}' if ser[nm][o] else f'{o} complex'
                for o in range(7, 17)))
        # EN order 2 vs Phase 11c D (symmetrized EN with the CSF-basis diagonal of H_PP as denominators)
        mx, d2d = 0.0, None
        for k, d in T.items():
            Hs, nP = mat(d, orbs, 'full', k[0])
            V, hq, hp = Hs[:nP, nP:], np.diag(Hs)[nP:], np.diag(Hs)[:nP]
            Rr = 1.0 / (hp[:, None] - hq[None, :])
            Dm = Hs[:nP, :nP] + 0.5 * ((V * Rr) @ V.T + V @ (V * Rr).T)
            e2 = series(Hs, nP, [[q] for q in range(Hs.shape[0] - nP)], nmax=2)[2]
            if nP == 1:
                mx = max(mx, abs(e2[0, 0] - Dm[0, 0]))
            else:
                d2d = float(np.abs(U.T @ (e2 - Dm) @ U).max())
        a(f'\nEN order 2 vs Phase 11c D (coefficient {j11c["paths"][X]["results"]["D"]["coef_E"]:.3f}): identical for all '
          f'non-2D terms (max |dE| {mx:.1e} Eh); they differ only in the d3 2D block (max |dH| {1e3 * d2d:.4f} mEh), '
          'where 11c D used the CSF-basis diagonal of H_PP and here H0 contains the full PHP.')
        js['paths'][X] = dict(hierarchy=res, series=ser, R={m: v.tolist() for m, v in R.items()})

    # 3. both paths
    a('\n# 3. Both paths together: CI(P + Q_d + Q_g)')
    Edg = {k: heff_E(*mat(d, virt['d'] + virt['g'], 'full', k[0]), k == (3, '2D')) for k, d in T.items()}
    Rdg = vec(Edg)
    Sadd = dEX['d'] + dEX['g']
    nad = Rdg - Sadd
    a(f'rms delta(dg) {1e3 * rms(Rdg):.4f}, rms delta(E_d) + delta(E_g) {1e3 * rms(Sadd):.4f} mEh; '
      f'angle between them {P.wangle(Rdg, Sadd):.2f} deg; coefficient of delta(dg) on the sum {P.ip(Rdg, Sadd) / P.ip(Sadd, Sadd):.4f}')
    a(f'Non-additive part delta(dg) - sum: rms {1e3 * rms(nad):.4f} mEh ({100 * rms(nad) / rms(Sadd):.1f}% of the sum); '
      f'angle to the sum {P.wangle(nad, Sadd):.1f} deg, to delta(E_d) {P.wangle(nad, dEX["d"]):.1f}, '
      f'to delta(E_g) {P.wangle(nad, dEX["g"]):.1f} deg')
    js['both'] = dict(rms_dg=rms(Rdg), rms_sum=rms(Sadd), rms_nonadd=rms(nad), angle=P.wangle(Rdg, Sadd),
                      angle_nonadd_sum=P.wangle(nad, Sadd), R_dg=Rdg.tolist())

    # table: orbitals and angular vectors
    a('\n# Per orbital (n=3) and unit-integral angular vectors')
    gmap = {'d': ('g2', 'g3'), 'g': ('g0', 'g1')}
    js['angular'] = {}
    for X in ('d', 'g'):
        L = j11c['paths'][X]['labels']
        i2, i4 = gmap[X]
        a(f'\n## 3d -> n{X}')
        a('| orbital | Delta_i (Eh) | R^2_i(3d i;3d3d) | R^4_i(3d i;3d3d) |')
        a('|---|---|---|---|')
        for o in virt[X]:
            g = L[o]
            assert g['groups'][int(i2[1])].startswith('R2(') and g['groups'][int(i4[1])].startswith('R4('), g['groups']
            a(f'| {o} | {j11c["paths"][X]["Delta_orb"][f"3 {o}"]:.6f} | {g["values"][int(i2[1])]:+.8f} | '
              f'{g["values"][int(i4[1])]:+.8f} |')
        A = {}
        for nm, key in (('A22', f'{i2} x {i2}'), ('A24', f'{i2} x {i4}'), ('A44', f'{i4} x {i4}')):
            vs = np.array([L[o]['pairs'][key]['unit_R'] for o in virt[X]])
            A[nm] = vs[0]
            a(f'{nm}: max spread over orbitals {np.abs(vs - vs[0]).max():.1e} Eh')
        a('| component | ' + ' | '.join(f'{nm} | {nm} closed form' for nm in A) + ' |')
        a('|---|' + '---|---|' * len(A))
        cf = {nm: [closed(x, i < 8) for i, x in enumerate(A[nm])] for nm in A}
        for i, c in enumerate(COMP):
            a(f'| {c} | ' + ' | '.join(f'{A[nm][i]:+.10f} | {cf[nm][i] or "?"}' for nm in A) + ' |')
        js['angular'][X] = {nm: dict(values=A[nm].tolist(), closed=cf[nm]) for nm in A}

    # sign convention of the d^3 2D seniority basis
    St, q = cfp_l.stretched(2, 3, 2, 1)
    D = cfp_l.dets(2, 3)
    SO = cfp_l.spin_orbitals(2)
    desc = []
    for k in range(2):
        i = int(np.argmax(np.abs(St[:, k])))
        desc.append(' '.join(f'({SO[p][0]:+d},{"ab"[SO[p][1] < 0]})' for p in D[i]) + f' {St[i, k]:+.6f}')
    C = cfp_l.slater_condon(2, 3, 2, 1, {2: 49.0})     # F^2 = 49 -> F2 = 1 -> B = 1 (C = 0)
    a('\n## Sign convention of the d^3 2D block')
    a('Basis of atomci_heff/cfp_l.stretched(2, 3, L=2, 2S=1): stretched states M_L = 2, M_S = 1/2, eigenvectors of the '
      'pairing operator (v=1 first, v=3 second), each with its largest-magnitude determinant coefficient positive. '
      'Determinants: ascending spin-orbital index, spin-orbitals ordered (m_l = -2..2) x (alpha, beta); '
      f'largest component v=1: {desc[0]}; v=3: {desc[1]}. ATOMCI 2D CSFs are mapped onto this basis by the signed '
      'permutation of phase9.seniority_map (fit of the CI(P) block to the pair sum).')
    a(f'Check of the phase: in this basis the Coulomb operator with B = 1, C = 0 has <v1|H|v3> = {C[0, 1]:+.8f} '
      f'= {closed(C[0, 1], False) or "?"} (textbook magnitude 3 sqrt(21) = {3 * np.sqrt(21):.8f}); the d^3 2D off-diagonal '
      'is +3 sqrt(21) B in this convention if the sign above is +.')
    js['sign'] = dict(v1=desc[0], v3=desc[1], offdiag_B1=float(C[0, 1]))
    open(os.path.join(OUT, 'phase11d.txt'), 'w').write('\n'.join(w) + '\n')
    json.dump(js, open(os.path.join(OUT, 'phase11d.json'), 'w'), indent=1, default=float)
    print('\n'.join(w))


if __name__ == '__main__':
    main()
