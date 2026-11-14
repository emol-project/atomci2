#!/usr/bin/env python3
"""Phase 13d: orbital relaxation in the d channel (fixed vq_orb IVO orbitals, singles-only spaces).

Spaces (runs vq_<ion>_<term>e_ivo_s<spec>, scripts/gen_v.py suffix _s): P = 3d^N, Q_S = 3d^(N-1) nd, nd = 11d, 12d (s3d)
or 11d..15d (s6d); no g, no doubles. d^0 = [Ar] core energy (as 13c).

step1   d-channel decomposition (13c pipeline) of the singles spaces, N = 0..5; comparison with 13c d-only (3d) and
        5d1g / 6d2g (JL coordinates from phase13c_valence.json).
step2   coupling split <3d^(N-1) nd|H|3d^N> = A + M from the EE tape (fort.15):
          E_nd = coefficient matrix of the label h(3d,nd) on the P x Q_nd block (the one-body excitation pattern);
          A = f_N(nd) E_nd, f_N(nd) = degeneracy-weighted projection of H_PQ on E_nd over all (L,S) of d^N
              (= off-diagonal Fock element of the d^N average of configuration; checked label by label);
          M = H_PQ - A (R2, R4 traceless parts).
        Model check: H = sum over labels (tape coefficient x label value), label values fitted to all runs.
        Variants (a) A + M, (b) M only, (c) A only; H_PP and H_QQ unchanged.
step3   cost estimate of a d-channel run with doubles in variant (b) (no runs).
-> results/phase13/phase13d.txt/.json
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import dshell as S  # noqa: E402
from atomci_heff import eetape, heff  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402
from atomci_heff.reader import read_csfs  # noqa: E402
import phase13b as B  # noqa: E402
import phase13c as C13  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase13')
EK, TK = B.EK, B.TK
ION = {1: 'v4p', 2: 'v3p', 3: 'v2p', 4: 'v1p', 5: 'v0p'}
SPACES = {'s3d': ['11d', '12d'], 's6d': ['11d', '12d', '13d', '14d', '15d']}
HART_EV = 27.211386


def canon(lab, names):
    if lab[0] == '1e':
        a, b = sorted((names[lab[1] - 1], names[lab[2] - 1]))
        return f'h({a},{b})'
    _, k, a, b, c, d = lab
    p1 = tuple(sorted((names[a - 1], names[b - 1])))
    p2 = tuple(sorted((names[c - 1], names[d - 1])))
    p1, p2 = sorted((p1, p2))
    return f'R{k}({p1[0]}{p1[1]};{p2[0]}{p2[1]})'


def term_list(N):
    out = []
    for L, twoS, vs in S.terms(N):
        out.append((L, twoS, len(vs), f'{twoS + 1}{S.LCH[L]}'))
    return out


def load(sp, N, t):
    """H (ATOMCI export), labels, P / Q_nd indices and the tape coefficient matrices {label: n x n}."""
    run = f'vq_{ION[N]}_{t}e_ivo_{sp}'
    H, lab = B.load_run(run)
    n = len(lab)
    P = [i for i, l in enumerate(lab) if l == f'3d{N}']
    Q = {o: [i for i, l in enumerate(lab) if o in B.cfg_orbitals(l)] for o in SPACES[sp]}
    assert len(P) + sum(map(len, Q.values())) == n
    tape = os.path.join(ROOT, 'runs', run, 'fort.15')
    hd, _ = read_csfs(tape)
    names = hd['orbitals']
    tab = I.Addressing(['spdfgh'.index(o[-1]) + 1 for o in names]).label_table()
    W = defaultdict(lambda: np.zeros((n, n)))
    for i, j, wt, adr, n1 in eetape.read_ee(tape):
        for x, a in zip(wt, adr):
            nm = canon(tab[int(a)], names)
            W[nm][i, j] += x
            if i != j:
                W[nm][j, i] += x
    return dict(run=run, H=H, lab=lab, P=P, Q=Q, W=dict(W))


def heff_block(H, lab, N, t, mult):
    """Exact des Cloizeaux H_eff in the seniority basis (13b maps) and target diagnostics."""
    Pi = [i for i, l in enumerate(lab) if l == f'3d{N}']
    e, Cv, wP, _ = heff.spectrum(H, Pi)
    st, amb = heff.select_states(wP, mult)
    other = [k for k in range(len(e)) if k not in st]
    r = heff.exact_heff(H, Pi, st) if other else dict(hermitian=H, condX=1.0)
    U = B.core_map(N, t)
    dom = ''
    if other:
        c2 = defaultdict(float)
        for i, l in enumerate(lab):
            c2[l] += Cv[i, other[0]] ** 2
        dom = max(c2, key=c2.get)
    return U.T @ r['hermitian'] @ U, dict(
        ncsf=len(lab), wP=float(wP[st].min()), amb=float(amb), cond=float(r['condX']),
        below=int(sum(1 for k in other if e[k] < e[st].max())),
        gap_eV=float((e[other[0]] - e[st].max()) * HART_EV) if other else np.inf, dom=dom)


# ---------------------------------------------------------------- coupling split

def split(data):
    """f_N(nd) (projection of H_PQ on E_nd, degeneracy weighted over all (L,S) of d^N), per-label projections alpha and
    traceless remainders, for one space. data: {(N, t): load(...)}."""
    f, alpha = {}, {}
    for N in range(1, 6):
        uniq = [(data[(N, t)], (2 * L + 1) * (twoS + 1)) for L, twoS, m, t in term_list(N)]
        for o in next(iter(data.values()))['Q']:
            hl = f'h({o},3d)' if o < '3d' else f'h(3d,{o})'
            num = den = 0.0
            lab_num = defaultdict(float)
            for d, g in uniq:
                E = d['W'][hl][np.ix_(d['P'], d['Q'][o])]
                num += g * np.sum(E * d['H'][np.ix_(d['P'], d['Q'][o])])
                den += g * np.sum(E * E)
                for nm, Wm in d['W'].items():
                    blk = Wm[np.ix_(d['P'], d['Q'][o])]
                    if np.any(blk):
                        lab_num[nm] += g * np.sum(E * blk)
            f[(N, o)] = num / den
            al = {}
            for nm, x in lab_num.items():
                a = x / den
                # traceless remainder of this label's pattern (weighted norm relative to the E pattern)
                rem = sum(g * np.sum((d['W'][nm][np.ix_(d['P'], d['Q'][o])] - a * d['W'][hl][np.ix_(d['P'], d['Q'][o])])
                                     ** 2) for d, g in uniq if nm in d['W'])
                al[nm] = (a, float(np.sqrt(rem / den)))
            alpha[(N, o)] = al
    return f, alpha


def variant_H(d, f, kind):
    H = d['H'].copy()
    if kind == 'a':
        return H
    for o, Qi in d['Q'].items():
        N = int(d['lab'][d['P'][0]][2:])
        hl = f'h({o},3d)' if o < '3d' else f'h(3d,{o})'
        E = d['W'][hl][np.ix_(d['P'], Qi)]
        A = f[(N, o)] * E
        new = H[np.ix_(d['P'], Qi)] - A if kind == 'b' else A
        H[np.ix_(d['P'], Qi)] = new
        H[np.ix_(Qi, d['P'])] = new.T
    return H


def vectors(data, f, kind):
    vecs, info = {0: np.array([B.ECORE])}, {}
    for N in range(1, 6):
        blocks = {}
        for L, twoS, m, t in term_list(N):
            d = data[(N, t)]
            Bk, inf = heff_block(variant_H(d, f, kind), d['lab'], N, t, m)
            blocks[(L, twoS)] = Bk
            info[(N, t)] = inf
        vecs[N] = B.vec_from_blocks(N, blocks)
    return vecs, info


def d5_metrics(vecs, V):
    n = lambda x: float(np.sqrt(S.ip(5, x, x)))
    r5 = vecs[5] - B.predict(V, 5)
    V4 = V + [S.from_vec(4, vecs[4]) - S.apply_parts(V, 4)]
    r5b = vecs[5] - B.predict(V4, 5)
    _, nr = B.jl_split(5, r5)
    tab = S.jl_table(5)
    rms = lambda x: np.sqrt(S.ip(5, x, x) / comb(10, 5))
    return dict(r5=n(r5), rms5=float(rms(r5)), rms5_V4=float(rms(r5b)),
                h5=B.eav(5, vecs[5]) - B.eav(5, B.predict(V, 5)),
                e=float(np.sqrt(sum(nr[k] ** 2 for k in EK))), t=float(np.sqrt(sum(nr[k] ** 2 for k in TK))),
                f=float(np.sqrt(sum(nr[k] ** 2 for k in tab if k.startswith('f')))))


def jl_vec(nr):
    return np.array([nr[k] for k in S.jl_table(4)])


def main():
    t0 = time.time()
    w = []
    a = w.append
    js = {}
    a('# Phase 13d: orbital relaxation in the d channel (fixed vq_orb IVO orbitals, singles-only spaces)')
    a('P = 3d^N, Q_S = 3d^(N-1) nd; s3d: nd = 11d, 12d; s6d: nd = 11d..15d. d^0 = [Ar] core energy. Exact des Cloizeaux '
      'H_eff in the seniority basis (13b maps), V0..V3 from d^0..d^3, r4 = d^4 - lift(V0..V3). mEh unless noted.')
    c13 = json.load(open(os.path.join(OUT, 'phase13c_valence.json')))['spaces']
    data = {sp: {(N, t): load(sp, N, t) for N in range(1, 6) for L, twoS, m, t in term_list(N)} for sp in SPACES}
    ncsf = {sp: max(len(d['lab']) for d in dd.values()) for sp, dd in data.items()}

    # ---- model check: label values fitted to all runs; reproduce H and the eigenvalues
    a('\n## Model check (EE tape)')
    labs = sorted({nm for dd in data.values() for d in dd.values() for nm in d['W']})
    col = {nm: k for k, nm in enumerate(labs)}
    rows, rhs = [], []
    for dd in data.values():
        for d in dd.values():
            n = len(d['lab'])
            iu = np.tril_indices(n)
            R = np.zeros((len(iu[0]), len(labs)))
            for nm, Wm in d['W'].items():
                R[:, col[nm]] = Wm[iu]
            rows.append(R)
            rhs.append(d['H'][iu])
    R, y = np.vstack(rows), np.concatenate(rhs)
    g, *_ = np.linalg.lstsq(R, y, rcond=None)
    rank = np.linalg.matrix_rank(R)
    dev_h = dev_e = 0.0
    for dd in data.values():
        for d in dd.values():
            Hm = sum(g[col[nm]] * Wm for nm, Wm in d['W'].items())
            dev_h = max(dev_h, np.abs(Hm - d['H']).max())
            dev_e = max(dev_e, np.abs(np.linalg.eigvalsh(Hm) - np.linalg.eigvalsh(d['H'])).max())
    a(f'{sum(len(dd) for dd in data.values())} runs, {len(y)} matrix elements, {len(labs)} labels (rank {rank}; core '
      f'labels are collinear with h, values min-norm). sum_label (tape coefficient x fitted value) reproduces the ATOMCI '
      f'H to {dev_h:.1e} Eh and the eigenvalues to {dev_e:.1e} Eh.')
    js['model_check'] = dict(labels=len(labs), rank=int(rank), dev_h=dev_h, dev_eig=dev_e)
    gval = {nm: float(g[col[nm]]) for nm in labs}

    # ---- split
    res = {}
    for sp, dd in data.items():
        f, alpha = split(dd)
        res[sp] = dict(f=f, alpha=alpha)
        for kind in 'abc':
            vecs, info = vectors(dd, f, kind)
            m = C13.metrics(vecs)
            m5 = d5_metrics(vecs, m['V'])
            res[sp][kind] = dict(vecs=vecs, info=info, m=m, m5=m5)
    # (a) is the ATOMCI H itself
    for sp, dd in data.items():
        dev = 0.0
        for (N, t), d in dd.items():
            dev = max(dev, np.abs(variant_H(d, res[sp]['f'], 'a') - d['H']).max())
        a(f'- {sp}: (a) = ATOMCI H exactly; max |A + M - H_PQ| = 0 by construction; largest CSF block {ncsf[sp]}')

    a('\n## Off-diagonal average-of-configuration Fock element f_N(3d, nd) = A coefficient (mEh)')
    a('Projection of H_PQ on the h(3d,nd) pattern, weighted by (2L+1)(2S+1) over all terms of d^N. Brillouin '
      '(average) would make it 0. Orbitals: V2+ 3d3 4F HF + IVO virtuals, so N = 3 is the reference.')
    for sp in SPACES:
        f = res[sp]['f']
        a(f'\n{sp}:')
        a('| nd | ' + ' | '.join(f'N={N}' for N in range(1, 6)) + ' | slope per electron |')
        a('|---|' + '---|' * 6)
        for o in SPACES[sp]:
            v = [f[(N, o)] for N in range(1, 6)]
            a(f'| {o} | ' + ' | '.join(f'{1e3 * x:+.3f}' for x in v) + f' | {1e3 * (v[4] - v[0]) / 4:+.3f} |')
    js['f'] = {sp: {f'{N} {o}': v for (N, o), v in res[sp]['f'].items()} for sp in SPACES}

    # label check
    a('\nLabel by label (s6d, nd = 11d): projection alpha_N of each label pattern on E (H_PQ = sum label x pattern) '
      'and the traceless remainder |pattern - alpha E| / |E|. Expected for the average Fock element: h and the closed '
      'core labels alpha = 1 (N independent), R0(3d3d;3d nd) alpha = N - 1, R2/R4 alpha = -(2/63)(N - 1), and only '
      'R2/R4 have a remainder (= M).')
    al = res['s6d']['alpha']
    names = sorted({nm for N in range(1, 6) for nm in al[(N, '11d')]})
    a('| label | ' + ' | '.join(f'alpha N={N} | rem N={N}' for N in range(1, 6)) + ' |')
    a('|---|' + '---|' * 10)
    lab_ok = []
    for nm in names:
        cells = []
        for N in range(1, 6):
            x = al[(N, '11d')].get(nm)
            cells.append(f'{x[0]:+.6f} | {x[1]:.2e}' if x else '- | -')
        a(f'| {nm} | ' + ' | '.join(cells) + ' |')
    js['alpha_11d'] = {nm: {N: al[(N, '11d')].get(nm) for N in range(1, 6)} for nm in names}
    # analytic f_N from the fitted label values: f_N = sum_label alpha_label(N) g_label
    devf = max(abs(sum(x[0] * gval[nm] for nm, x in res[sp]['alpha'][(N, o)].items()) - res[sp]['f'][(N, o)])
               for sp in SPACES for N in range(1, 6) for o in SPACES[sp])
    a(f'\nsum_label alpha x fitted label value = f_N to {devf:.1e} Eh (all spaces, N, nd).')
    # M content: sum over R2, R4(3d3d;3d nd) of value x traceless pattern
    a('Size of the parts (weighted Frobenius norm over all terms of d^N, sum over nd, mEh):')
    a('| space | part | ' + ' | '.join(f'N={N}' for N in range(1, 6)) + ' |')
    a('|---|---|' + '---|' * 5)
    for sp, dd in data.items():
        nA, nM = defaultdict(float), defaultdict(float)
        for (N, t), d in dd.items():
            L, twoS = B.term_key(N, t)
            g_ = (2 * L + 1) * (twoS + 1)
            for o, Qi in d['Q'].items():
                hl = f'h({o},3d)' if o < '3d' else f'h(3d,{o})'
                E = d['W'][hl][np.ix_(d['P'], Qi)]
                A = res[sp]['f'][(N, o)] * E
                M = d['H'][np.ix_(d['P'], Qi)] - A
                nA[N] += g_ * np.sum(A * A)
                nM[N] += g_ * np.sum(M * M)
        a(f'| {sp} | A | ' + ' | '.join(f'{1e3 * np.sqrt(nA[N]):.2f}' for N in range(1, 6)) + ' |')
        a(f'| {sp} | M | ' + ' | '.join(f'{1e3 * np.sqrt(nM[N]):.2f}' for N in range(1, 6)) + ' |')
        res[sp]['normsAM'] = dict(A={N: float(np.sqrt(nA[N])) for N in nA}, M={N: float(np.sqrt(nM[N])) for N in nM})

    # ---- targets
    a('\n## Targets per variant (min P weight / max ambiguity / max cond / roots below a target / min gap eV '
      '(block, dominant configuration of the lowest non-target root))')
    a('| space | variant | N | min P weight | max amb | max cond | below | min gap (eV) |')
    a('|---|---|---|---|---|---|---|---|')
    tg = {}
    for sp in SPACES:
        for kind in 'abc':
            info = res[sp][kind]['info']
            for N in range(1, 6):
                it = {t: x for (NN, t), x in info.items() if NN == N}
                tmin = min(it, key=lambda t: it[t]['gap_eV'])
                row = dict(wP=min(x['wP'] for x in it.values()), amb=max(x['amb'] for x in it.values()),
                           cond=max(x['cond'] for x in it.values()), below=sum(x['below'] for x in it.values()),
                           gap=it[tmin]['gap_eV'], blk=tmin, dom=it[tmin]['dom'])
                tg[(sp, kind, N)] = row
                if N >= 2:
                    a(f'| {sp} | {kind} | {N} | {row["wP"]:.4f} | {row["amb"]:.2e} | {row["cond"]:.3f} | {row["below"]} | '
                      f'{row["gap"]:+.1f} ({tmin}, {row["dom"]}) |')
    js['targets'] = {f'{sp} {k} {N}': v for (sp, k, N), v in tg.items()}

    # ---- origin of the intruders: the Q diagonal (unchanged in every variant)
    a('\n## Q diagonal vs P (ATOMCI singles H; the same in (a), (b), (c))')
    a('E_av = (2L+1)(2S+1)-weighted mean of the CSF diagonal over all terms; min-gap = lowest eigenvalue of the '
      'H_{Q_nd Q_nd} block - highest eigenvalue of H_PP, worst (L,S) block (eV).')
    a('| space | N | E_av(3d^(N-1) nd) - E_av(3d^N) (eV) | min-gap (eV, block, nd) |')
    a('|---|---|---|---|')
    dg = {}
    for sp, dd in data.items():
        for N in range(2, 6):
            acc, worst = defaultdict(lambda: [0.0, 0.0]), None
            for L, twoS, m, t in term_list(N):
                d = dd[(N, t)]
                g_ = (2 * L + 1) * (twoS + 1)
                eP = np.linalg.eigvalsh(d['H'][np.ix_(d['P'], d['P'])])
                for i in d['P']:
                    acc['P'][0] += g_ * d['H'][i, i]
                    acc['P'][1] += g_
                for o, Qi in d['Q'].items():
                    for i in Qi:
                        acc[o][0] += g_ * d['H'][i, i]
                        acc[o][1] += g_
                    gap = (np.linalg.eigvalsh(d['H'][np.ix_(Qi, Qi)]).min() - eP.max()) * HART_EV
                    if worst is None or gap < worst[0]:
                        worst = (gap, t, o)
            av = {k: v[0] / v[1] for k, v in acc.items()}
            dg[f'{sp} {N}'] = dict(eav={o: (av[o] - av['P']) * HART_EV for o in SPACES[sp]}, worst=worst)
            a(f'| {sp} | {N} | ' + ', '.join(f'{o} {(av[o] - av["P"]) * HART_EV:+.1f}' for o in SPACES[sp]) +
              f' | {worst[0]:+.1f} ({worst[1]}, {worst[2]}) |')
    js['q_diagonal'] = dg

    # ---- decomposition table
    a('\n## d^4 four-body residual and d^5 (norms in the JL metric, sum over states)')
    a('| space | variant | |V3 lifted| | |V3 t lifted| | |r4| | r4/|V3| | r4/|V3 t| | max-term | h4 | |e| | |t| | |f| | '
      'd5: rms/state V0..V3 | d5: rms/state V0..V4 | d5: E_av resid |')
    a('|---|---|' + '---|' * 13)
    rowsJ = {}

    def row(lbl, kind, m, m5=None):
        a(f'| {lbl} | {kind} | {1e3 * m["V3"]:.3f} | {1e3 * m["V3t"]:.3f} | {1e3 * m["r4"]:.3f} | {m["ratio"]:.4f} | '
          f'{m["ratio_t"]:.3f} | {1e3 * m["maxterm"]:.3f} | {1e3 * m["h4"]:+.3f} | {1e3 * m["e"]:.3f} | '
          f'{1e3 * m["t"]:.3f} | {1e3 * m["f"]:.3f} | ' +
          (f'{1e3 * m5["rms5"]:.3f} | {1e3 * m5["rms5_V4"]:.3f} | {1e3 * m5["h5"]:+.3f} |' if m5 else '- | - | - |'))

    for sp in SPACES:
        for kind in 'abc':
            r = res[sp][kind]
            row(sp, kind, r['m'], r['m5'])
            rowsJ[f'{sp} {kind}'] = dict(m={k: v for k, v in r['m'].items() if k not in ('r', 'V')}, m5=r['m5'])
    for sp in ('3d', '5d1g', '6d2g'):
        row(f'13c {sp}', 'full CI', c13[sp]['m'])
    js['rows'] = rowsJ

    # ---- fractions of the 13c d-channel r4 (JL coordinates; the d^4 JL operators are a complete orthogonal set)
    a('\n## Overlap with the 13c r4 (d^4): projection fraction <r, r_ref>/|r_ref|^2, norm ratio, angle')
    a('| this | reference | fraction | norm ratio | angle (deg) | e0 coordinate this / ref |')
    a('|---|---|---|---|---|---|')
    fr = {}
    for sp in SPACES:
        for kind in 'abc':
            x = jl_vec(res[sp][kind]['m']['jl'])
            for ref in ('3d', '5d1g', '6d2g'):
                y = jl_vec(c13[ref]['m']['jl'])
                fr[f'{sp} {kind} / {ref}'] = dict(frac=float(x @ y / (y @ y)), ratio=float(np.linalg.norm(x) / np.linalg.norm(y)),
                                                  angle=float(np.degrees(np.arccos(np.clip(x @ y / np.linalg.norm(x) /
                                                                                           np.linalg.norm(y), -1, 1)))))
                if (sp, ref) in (('s3d', '3d'), ('s6d', '5d1g'), ('s6d', '6d2g')) or kind == 'a' and ref == '3d':
                    q = fr[f'{sp} {kind} / {ref}']
                    a(f'| {sp} {kind} | 13c {ref} | {q["frac"]:.3f} | {q["ratio"]:.3f} | {q["angle"]:.1f} | '
                      f'{1e3 * res[sp][kind]["m"]["jl"]["e0"]:+.2f} / {1e3 * c13[ref]["m"]["jl"]["e0"]:+.2f} |')
    js['fractions'] = fr
    # term-dependent part only (t and f coordinates)
    a('\nTerm-dependent part only (t1..t4 and f coordinates of r4):')
    a('| this | reference | fraction | norm ratio | angle (deg) |')
    a('|---|---|---|---|---|')
    tabk = list(S.jl_table(4))
    tf = [i for i, k in enumerate(tabk) if k.startswith('t') or k.startswith('f')]
    for sp, ref in (('s3d', '3d'), ('s6d', '5d1g'), ('s6d', '6d2g')):
        y = jl_vec(c13[ref]['m']['jl'])[tf]
        for kind in 'abc':
            x = jl_vec(res[sp][kind]['m']['jl'])[tf]
            a(f'| {sp} {kind} | 13c {ref} | {x @ y / (y @ y):.3f} | {np.linalg.norm(x) / np.linalg.norm(y):.3f} | '
              f'{np.degrees(np.arccos(np.clip(x @ y / np.linalg.norm(x) / np.linalg.norm(y), -1, 1))):.1f} |')

    # ---- where does the term-dependent four-body part live: (a) vs (b) + (c)
    a('\n## Split of r4 between (b) M only and (c) A only (d^4 JL coordinates)')
    a('| space | part | |r4(a)| | |r4(b)| | |r4(c)| | |r4(a) - r4(b) - r4(c)| | angle(a, b) | angle(a, c) |')
    a('|---|---|---|---|---|---|---|---|')
    sel = {'all': list(range(len(tabk))), 'e': [i for i, k in enumerate(tabk) if k.startswith('e')],
           't': [i for i, k in enumerate(tabk) if k.startswith('t')], 'f': [i for i, k in enumerate(tabk) if k.startswith('f')],
           't+f': tf}
    ang = lambda x, y: float(np.degrees(np.arccos(np.clip(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-300), -1, 1))))
    for sp in SPACES:
        X = {k: jl_vec(res[sp][k]['m']['jl']) for k in 'abc'}
        for part, ix in sel.items():
            xa, xb, xc = X['a'][ix], X['b'][ix], X['c'][ix]
            a(f'| {sp} | {part} | {1e3 * np.linalg.norm(xa):.3f} | {1e3 * np.linalg.norm(xb):.3f} | '
              f'{1e3 * np.linalg.norm(xc):.3f} | {1e3 * np.linalg.norm(xa - xb - xc):.3f} | {ang(xa, xb):.1f} | {ang(xa, xc):.1f} |')
    a('\nJL coordinates of r4 (normalized, mEh):')
    a('| space | variant | ' + ' | '.join(tabk) + ' |')
    a('|---|---|' + '---|' * len(tabk))
    for sp in SPACES:
        for k in 'abc':
            a(f'| {sp} | {k} | ' + ' | '.join(f'{1e3 * res[sp][k]["m"]["jl"][q]:+.3f}' for q in tabk) + ' |')
    for ref in ('3d', '5d1g', '6d2g'):
        a(f'| 13c {ref} | full | ' + ' | '.join(f'{1e3 * c13[ref]["m"]["jl"][q]:+.3f}' for q in tabk) + ' |')

    # V3 t-part coordinates
    a('\nt-part of V3 on (u, t4, w, t3) (normalized, mEh):')
    a('| space | variant | u | t4 | w | t3 |')
    a('|---|---|---|---|---|---|')
    for sp in SPACES:
        for k in 'abc':
            _, co, _ = B.tpart(3, S.to_vec(3, res[sp][k]['m']['V'][3]))
            a(f'| {sp} | {k} | ' + ' | '.join(f'{1e3 * co[q]:+.4f}' for q in ('u', 't4', 'w', 't3')) + ' |')
            rowsJ[f'{sp} {k}']['V3t_coords'] = co

    js['elapsed_s'] = time.time() - t0
    a(f'\nAnalysis wall time {time.time() - t0:.0f} s.')
    json.dump(js, open(os.path.join(OUT, 'phase13d.json'), 'w'), indent=1, default=lambda x: float(x) if np.isscalar(x)
              else str(x))
    open(os.path.join(OUT, 'phase13d.txt'), 'w').write('\n'.join(w) + '\n')
    print('\n'.join(w))


if __name__ == '__main__':
    main()
