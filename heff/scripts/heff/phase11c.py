#!/usr/bin/env python3
"""Phase 11c: closure analysis of the valence paths 3d -> nd and 3d -> ng (V QZP, [Ar] frozen, vq_orb set 2).

Exports/tapes of the Phase 9 valence full CIs vq_<ion>_<term>_ivo (the runs of Phase 9b/9c, QZP).
Per path X (d or g), per term of d2 and d3:
  P = 3d^n; Q_X = 3d^(n-1) x1 for every virtual orbital x of l = X (d: 11d..15d, g: 11g..13g).
Step 0: Q counts from coupling rules (d^(n-1) parent terms x one l electron), asserted against the export.
Effective Hamiltonians on P (as Phase 11a; no diagonalization for B-D):
  A   CI(P)
  B1  closure, one Delta per n for the whole path: E_avQ(path, n) - E_av(3d^n)
  B2  closure per orbital: sum_i H_PQi H_QiP / (-Delta_i), Delta_i = E_avQ(i, n) - E_av(3d^n)
  C   sum_q H_Pq H_qP / (E_av(3d^n) - H_qq);   D symmetrized EN;   E CI(P + Q_X), des Cloizeaux on P
  (E_avQ: g-weighted average of diagonal H over the Q CSFs of the target LS only.)
Label decomposition of B2 (as 11b): H_PQ = sum_a v_a W_a from the tapes; labels with proportional coefficient
columns (over all P-Q elements of all terms) are merged into one group; group values by least squares.
Bare-nucleus check: virtual orbital vectors of l = f, g, h vs the eigenvectors of h = T - Z/r.

Usage: phase11c.py   -> results/phase11/phase11c.txt/.json
"""
import itertools
import json
import os
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
import gen_v  # noqa: E402
import phase9 as P  # noqa: E402
import phase10 as P10  # noqa: E402
from phase11a import d_terms  # noqa: E402
from atomci_heff import eetape, gto  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402
from atomci_heff.reader import read_csfs, read_vectors, records  # noqa: E402
from atomci_heff.sparse import SymLower  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'phase11')
LET = 'spdfgh'
ION = {2: 'v3p', 3: 'v2p'}
TERMS = [(2, t) for t in P.T2] + [(3, t) for t in P.T3 + ['2D']]
LX = {'d': 2, 'g': 4}


def rms(x):
    return float(np.sqrt(P.ip(x, x) / 120))


def n_q(n, L, S2, l):
    """CSFs of 3d^(n-1) x (x of angular momentum l) with (L, 2S), per orbital x."""
    par = d_terms(n - 1) if n > 1 else {(0, 0): 1}
    return sum(k for (Lp, Sp2), k in par.items() if abs(Lp - l) <= L <= Lp + l and abs(Sp2 - S2) == 1)


def read_pairs(tape, keep):
    """eetape.read_ee restricted to CSF pairs with both indices (0-based) in keep."""
    rec = records(tape)
    k0 = eetape._skip_header(rec)
    mask = np.zeros(int(np.max(keep)) + 2, bool)
    mask[keep] = True
    out = []
    for r in rec[k0:]:
        itw, jtw, noe, n1, n2, nij, nii, ltri = struct.unpack('<8i', r[:32])
        if noe == 0:
            continue
        off = 32
        wt = np.frombuffer(r[off:off + 8 * noe], '<f8')
        off += 8 * noe
        isqn = np.frombuffer(r[off:off + 4 * ltri], '<i4')
        off += 4 * ltri
        ii = np.frombuffer(r[off:off + 4 * nij], '<i4') - 1
        off += 4 * nij
        jj = np.frombuffer(r[off:off + 4 * nii], '<i4') - 1 if nii else ii
        ic, jc = np.minimum(ii, len(mask) - 1), np.minimum(jj, len(mask) - 1)
        m = mask[ic] & mask[jc] & (ii < len(mask) - 1) & (jj < len(mask) - 1)
        if not m.any():
            continue
        for p in np.nonzero(m)[0]:
            out.append((int(ii[p]), int(jj[p]), wt, isqn[noe * p:noe * (p + 1)]))
    return out


def canon(lab, names):
    if lab[0] == '1e':
        a, b = sorted((names[lab[1] - 1], names[lab[2] - 1]))
        return f'h({a},{b})'
    _, k, a, b, c, d = lab
    p1 = tuple(sorted((names[a - 1], names[b - 1])))
    p2 = tuple(sorted((names[c - 1], names[d - 1])))
    p1, p2 = sorted((p1, p2))
    return f'R{k}({p1[0]}{p1[1]};{p2[0]}{p2[1]})'


def term_data(n, t, virt):
    run = f'vq_{ION[n]}_{t}e_ivo'
    z = np.load(os.path.join(ROOT, 'results', f'{run}.npz'))
    labels = [str(s) for s in z['labels']]
    op = SymLower.from_npz(z)
    Pi = [i for i, lb in enumerate(labels) if P10.occ(lb) == {'3d': n}]
    Q = {}
    for X, orbs in virt.items():
        for o in orbs:
            want = {'3d': n - 1, o: 1} if n > 1 else {o: 1}
            Q[o] = [i for i, lb in enumerate(labels) if P10.occ(lb) == want]
    idx = sorted(set(Pi) | {i for q in Q.values() for i in q})
    H = op.sub(np.array(idx)).dense()
    pos = {i: k for k, i in enumerate(idx)}
    return dict(run=run, labels=labels, Pi=Pi, Q=Q, idx=idx, H=H, pos=pos)


def heff_E(Hs, nP, two):
    """CI(P + Q) with P = first nP rows; des Cloizeaux on P from the root(s) of largest P weight."""
    e, C = np.linalg.eigh(Hs)
    wP = (C[:nP] ** 2).sum(0)
    if not two:
        i = int(np.argmax(wP))
        return dict(E=float(e[i]), w=float(wP[i]))
    r = sorted(np.argsort(-wP)[:2].tolist())
    X = C[:nP][:, r]
    u, sv, vt = np.linalg.svd(X)
    Xt = u @ vt
    return dict(Heff=Xt @ np.diag(e[r]) @ Xt.T, w=[float(wP[k]) for k in r])


def decomp(E, U):
    E = dict(E)
    E[(0, '1S')] = dict(E=0.0)       # d0/d1 enter only the third difference, which is not used here
    E[(1, '2D')] = dict(E=0.0)
    return np.asarray(P10.decompose(E, U)['R']), P10.decompose(E, U)


def pack(M, two):
    M = 0.5 * (M + M.T)
    return dict(Heff=M) if two else dict(E=float(M[0, 0]))


def main():
    os.makedirs(OUT, exist_ok=True)
    w = []
    a = w.append
    hd0, _ = read_csfs(os.path.join(ROOT, 'runs', 'vq_v3p_3Fe_ivo', 'fort.15'))
    names0 = hd0['orbitals']
    virt = {X: [o for o in names0 if o[-1] == X and o != '3d'] for X in LX}
    a('# Phase 11c: closure analysis of 3d -> nd and 3d -> ng; V QZP, [Ar] frozen, vq_orb set 2 (Phase 9 runs vq_*_ivo); mEh')
    a(f'Virtual orbitals: d {virt["d"]}, g {virt["g"]}')
    T = {}
    for n, t in TERMS:
        T[(n, t)] = term_data(n, t, virt)
        print(T[(n, t)]['run'], flush=True)

    # step 0
    a('\n## Step 0: CSF counts (rule per orbital x number of orbitals vs export)')
    a('| term | P | Q_d rule | Q_d export | Q_g rule | Q_g export |')
    a('|---|---|---|---|---|---|')
    for (n, t), d in T.items():
        L, S2 = 'SPDFGH'.index(t[1]), int(t[0]) - 1
        cnt = {X: (n_q(n, L, S2, LX[X]) * len(virt[X]), sum(len(d['Q'][o]) for o in virt[X])) for X in LX}
        per_orb_ok = all(len(d['Q'][o]) == n_q(n, L, S2, LX[o[-1]]) for X in LX for o in virt[X])
        a(f'| d{n} {t} | {len(d["Pi"])} | {cnt["d"][0]} | {cnt["d"][1]} | {cnt["g"][0]} | {cnt["g"][1]} |')
        assert per_orb_ok and len(d['Pi']) == (2 if t == '2D' and n == 3 else 1), (n, t)
    a('All counts agree (also per orbital).')

    # P block, U, averages
    Eav = {}
    for n in (2, 3):
        tl = [(m, t) for m, t in TERMS if m == n]
        Eav[n] = sum(P.g(t) * np.trace(T[(m, t)]['H'][np.ix_(*[[T[(m, t)]['pos'][i] for i in T[(m, t)]['Pi']]] * 2)])
                     for m, t in tl) / sum(P.g(t) * len(T[(m, t)]['Pi']) for m, t in tl)

    def blk(d, rows, cols):
        return d['H'][np.ix_([d['pos'][i] for i in rows], [d['pos'][i] for i in cols])]

    EA = {(n, t): pack(blk(d, d['Pi'], d['Pi']), n == 3 and t == '2D') for (n, t), d in T.items()}
    d2a = {t: EA[(2, t)]['E'] - sum(P.g(x) * EA[(2, x)]['E'] for x in P.T2) / 45 for t in P.T2}
    Ha = EA[(3, '2D')]['Heff']
    Eav3a = (sum(P.g(t) * EA[(3, t)]['E'] for t in P.T3) + 10 * np.trace(Ha)) / 120
    U, mis = P.seniority_map(Ha, Eav3a * np.eye(2) + P.pairsum(d2a)['2D'])
    assert mis < 1e-8, mis
    RA, _ = decomp(EA, U)
    a(f'\nCheck: rms R(A) {1e3 * rms(RA):.1e} mEh; seniority map mismatch {mis:.1e} Eh')
    p9c = json.load(open(os.path.join(ROOT, 'results', 'phase9', 'phase9c.json')))['QZP']['D']

    js = dict(Eav=Eav, paths={})
    for X in LX:
        orbs = virt[X]
        # averages per orbital and per path
        Eq, Gq = defaultdict(float), defaultdict(float)
        for (n, t), d in T.items():
            for o in orbs:
                hq = np.diag(blk(d, d['Q'][o], d['Q'][o]))
                Eq[(n, o)] += P.g(t) * hq.sum()
                Gq[(n, o)] += P.g(t) * len(hq)
        Dorb = {(n, o): Eq[(n, o)] / Gq[(n, o)] - Eav[n] for n in (2, 3) for o in orbs}
        Dpath = {n: sum(Eq[(n, o)] for o in orbs) / sum(Gq[(n, o)] for o in orbs) - Eav[n] for n in (2, 3)}

        def H2(d, n, f):
            Pq = d['Pi']
            M = blk(d, Pq, Pq).copy()
            return M + f(d, n, Pq)

        def closure_path(d, n, Pq):
            Qa = [i for o in orbs for i in d['Q'][o]]
            V = blk(d, Pq, Qa)
            return -V @ V.T / Dpath[n]

        def closure_orb(d, n, Pq, only=None):
            M = 0
            for o in (orbs if only is None else [only]):
                V = blk(d, Pq, d['Q'][o])
                M = M - V @ V.T / Dorb[(n, o)]
            return M

        def termdep(d, n, Pq):
            Qa = [i for o in orbs for i in d['Q'][o]]
            V = blk(d, Pq, Qa)
            hq = np.diag(blk(d, Qa, Qa))
            return (V / (Eav[n] - hq)[None, :]) @ V.T

        def en(d, n, Pq):
            Qa = [i for o in orbs for i in d['Q'][o]]
            V = blk(d, Pq, Qa)
            hq = np.diag(blk(d, Qa, Qa))
            hp = np.diag(blk(d, Pq, Pq))
            R = 1.0 / (hp[:, None] - hq[None, :])
            return 0.5 * ((V * R) @ V.T + V @ (V * R).T)

        Ev = {}
        for nm, f in (('B1', closure_path), ('B2', closure_orb), ('C', termdep), ('D', en)):
            Ev[nm] = {(n, t): pack(H2(d, n, f), n == 3 and t == '2D') for (n, t), d in T.items()}
        Ev['E'] = {}
        for (n, t), d in T.items():
            Qa = [i for o in orbs for i in d['Q'][o]]
            Ev['E'][(n, t)] = heff_E(blk(d, d['Pi'] + Qa, d['Pi'] + Qa), len(d['Pi']), n == 3 and t == '2D')
        R = {k: decomp(v, U)[0] - RA for k, v in Ev.items()}
        dE = R['E']
        a(f'\n# Path 3d -> n{X} ({len(orbs)} virtual orbitals)')
        a('\n## Delta (Eh): path average (B1) and per orbital (B2), g-weighted over target-LS Q CSFs')
        a(f'B1: n=2 {Dpath[2]:.6f}, n=3 {Dpath[3]:.6f}')
        a('| orbital | Delta_i n=2 | Delta_i n=3 | H_qq spread n=2 (max-min) | H_qq spread n=3 |')
        a('|---|---|---|---|---|')
        spread = {}
        for o in orbs:
            sp = {}
            for n in (2, 3):
                hq = np.concatenate([np.diag(blk(d, d['Q'][o], d['Q'][o])) for (m, t), d in T.items() if m == n])
                sp[n] = float(hq.max() - hq.min())
            spread[o] = sp
            a(f'| {o} | {Dorb[(2, o)]:.6f} | {Dorb[(3, o)]:.6f} | {sp[2]:.6f} | {sp[3]:.6f} |')
        tsp = {n: float(max(np.diag(blk(d, d['Pi'], d['Pi'])).max() for (m, t), d in T.items() if m == n)
                        - min(np.diag(blk(d, d['Pi'], d['Pi'])).min() for (m, t), d in T.items() if m == n)) for n in (2, 3)}
        a(f'Term splitting of P (max-min of H_pp over terms): n=2 {tsp[2]:.6f}, n=3 {tsp[3]:.6f} Eh')

        a('\n## Three-body vector delta = R(X) - R(A)')
        a('| X | rms | angle to delta(E) | <delta,dE>/<dE,dE> | R3 (q1..q4) |')
        a('|---|---|---|---|---|')
        res = {}
        for k in ('B1', 'B2', 'C', 'D', 'E'):
            c = P.ip(R[k], dE) / P.ip(dE, dE)
            res[k] = dict(rms=rms(R[k]), angle_E=P.wangle(R[k], dE), coef_E=c, R3=[P.ip(R[k], q) for q in P.Q3])
            a(f'| {k} | {1e3 * rms(R[k]):.4f} | {P.wangle(R[k], dE):.1f} | {c:.3f} | '
              + ', '.join(f'{1e3 * x:+.3f}' for x in res[k]['R3']) + ' |')
        D9 = np.asarray(p9c[X])
        a(f'Phase 9c D_{X} (QZP): rms {1e3 * rms(D9):.4f}; delta(E) vs D_{X}: angle {P.wangle(dE, D9):.1f} deg, '
          f'<dE,D>/<D,D> {P.ip(dE, D9) / P.ip(D9, D9):.3f}; delta(B2) vs D_{X}: angle {P.wangle(R["B2"], D9):.1f} deg, '
          f'coefficient {P.ip(R["B2"], D9) / P.ip(D9, D9):.3f}')

        # orbital contributions to B2
        a('\n## Orbital contributions to B2 (ordered)')
        a('| orbital | rms | <R_i,B2>/<B2,B2> | cumulative |')
        a('|---|---|---|---|')
        Ri = {}
        for o in orbs:
            Eo = {(n, t): pack(H2(d, n, lambda d_, n_, Pq, o=o: closure_orb(d_, n_, Pq, o)), n == 3 and t == '2D')
                  for (n, t), d in T.items()}
            Ri[o] = decomp(Eo, U)[0] - RA
        cum = 0.0
        orbfrac = {}
        for o in sorted(orbs, key=lambda o: -abs(P.ip(Ri[o], R['B2']))):
            f = P.ip(Ri[o], R['B2']) / P.ip(R['B2'], R['B2'])
            cum += f
            orbfrac[o] = f
            a(f'| {o} | {1e3 * rms(Ri[o]):.4f} | {f:.4f} | {cum:.4f} |')
        js['paths'][X] = dict(Delta_path=Dpath, Delta_orb={f'{n} {o}': v for (n, o), v in Dorb.items()}, spread=spread,
                              term_splitting=tsp, results=res, orbital_fraction=orbfrac,
                              R={k: v.tolist() for k, v in R.items()}, D9c=D9.tolist())
        js['paths'][X]['_ctx'] = dict(orbs=orbs, Dorb=Dorb)
        js['paths'][X]['_U'] = U
        js['paths'][X]['_RB2'] = R['B2']
        js['paths'][X]['_orbvec'] = Ri
    labels_part(T, virt, js, w, U, RA)
    gcheck(w, js)
    for X in LX:
        for k in ('_ctx', '_U', '_RB2', '_orbvec'):
            js['paths'][X].pop(k)
    open(os.path.join(OUT, 'phase11c.txt'), 'w').write('\n'.join(w) + '\n')
    json.dump(js, open(os.path.join(OUT, 'phase11c.json'), 'w'), indent=1, default=float)
    print('\n'.join(w))


def labels_part(T, virt, js, w, U, RA):
    a = w.append
    # coefficient matrices from the tapes
    WPQ, WPP = {}, {}
    for (n, t), d in T.items():
        tape = os.path.join(ROOT, 'runs', d['run'], 'fort.15')
        hd, _ = read_csfs(tape)
        names = hd['orbitals']
        ad = I.Addressing([LET.index(o[-1]) + 1 for o in names])
        tab = ad.label_table()
        keep = np.array(d['idx'])
        where = {i: ('P', k) for k, i in enumerate(d['Pi'])}
        for o, q in d['Q'].items():
            where.update({i: (o, k) for k, i in enumerate(q)})
        wpq = defaultdict(lambda: defaultdict(lambda: None))
        wpp = defaultdict(lambda: np.zeros((len(d['Pi']), len(d['Pi']))))
        und = 0
        for i, j, wt, adr in read_pairs(tape, keep):
            (si, ki), (sj, kj) = where[i], where[j]
            if si != 'P' and sj != 'P':
                continue
            for x, aa in zip(wt, adr):
                aa = int(aa)
                if aa not in tab:
                    und += 1
                    continue
                nm = canon(tab[aa], names)
                if si == 'P' and sj == 'P':
                    wpp[nm][max(ki, kj), min(ki, kj)] += x
                    continue
                o, (kp, kq) = (sj, (ki, kj)) if si == 'P' else (si, (kj, ki))
                if wpq[o][nm] is None:
                    wpq[o][nm] = np.zeros((len(d['Pi']), len(d['Q'][o])))
                wpq[o][nm][kp, kq] += x
        assert und == 0, (d['run'], und)
        WPQ[(n, t)] = {o: dict(v) for o, v in wpq.items()}
        WPP[(n, t)] = dict(wpp)
        print('tape', d['run'], flush=True)

    def blk(d, rows, cols):
        return d['H'][np.ix_([d['pos'][i] for i in rows], [d['pos'][i] for i in cols])]

    a('\n# Label decomposition of B2 (tapes of the same runs)')
    # validation: all labels, P-Q and P-P'
    labs = sorted({nm for x in WPQ.values() for o in x.values() for nm in o} | {nm for x in WPP.values() for nm in x})
    col = {nm: c for c, nm in enumerate(labs)}
    rows, rhs = [], []
    for (n, t), d in T.items():
        for o, Wo in WPQ[(n, t)].items():
            Hq = blk(d, d['Pi'], d['Q'][o])
            for i in range(len(d['Pi'])):
                for j in range(len(d['Q'][o])):
                    r = np.zeros(len(labs))
                    for nm, W in Wo.items():
                        r[col[nm]] += W[i, j]
                    rows.append(r)
                    rhs.append(Hq[i, j])
        Hp = blk(d, d['Pi'], d['Pi'])
        for i in range(len(d['Pi'])):
            for j in range(i + 1):
                r = np.zeros(len(labs))
                for nm, W in WPP[(n, t)].items():
                    r[col[nm]] += W[i, j]
                rows.append(r)
                rhs.append(Hp[i, j])
    A, b = np.array(rows), np.array(rhs)
    v, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.abs(A @ v - b).max())
    a(f'Decoding check: {A.shape[0]} P-Q and P-P\' elements (d2, d3; both paths), {len(labs)} labels, rank {rank}; '
      f'max |fit - export| = {resid:.2e} Eh')
    js['decoding_residual'] = resid
    if resid > 1e-8:
        a('STOP: residual not small; label decomposition not done.')
        return
    for X in LX:
        ctx = js['paths'][X]['_ctx']
        orbs, Dorb = ctx['orbs'], ctx['Dorb']
        RB2 = js['paths'][X]['_RB2']
        a(f'\n## Path 3d -> n{X}')
        out = {}
        for o in orbs:
            lo = sorted({nm for x in WPQ.values() for nm in x.get(o, {})})
            # P-Q equations of orbital o, all terms
            cols, rr = [], []
            for (n, t), d in T.items():
                Wo = WPQ[(n, t)].get(o, {})
                Hq = blk(d, d['Pi'], d['Q'][o])
                for i in range(len(d['Pi'])):
                    for j in range(len(d['Q'][o])):
                        cols.append([Wo[nm][i, j] if nm in Wo else 0.0 for nm in lo])
                        rr.append(Hq[i, j])
            M, y = np.array(cols), np.array(rr)
            # merge proportional columns
            groups = []
            for c, nm in enumerate(lo):
                x = M[:, c]
                for g in groups:
                    r0 = M[:, g['cols'][0]]
                    lam = (x @ r0) / (r0 @ r0)
                    if np.abs(x - lam * r0).max() < 1e-10 * max(1.0, np.abs(x).max()):
                        g['cols'].append(c)
                        g['lam'].append(lam)
                        break
                else:
                    groups.append(dict(cols=[c], lam=[1.0]))
            G = np.array([M[:, g['cols'][0]] for g in groups]).T
            gv, _, grank, _ = np.linalg.lstsq(G, y, rcond=None)
            gres = float(np.abs(G @ gv - y).max())
            gname = [' + '.join((f'{lam:+.6g} ' if abs(lam - 1) > 1e-12 else '') + lo[c] for c, lam in zip(g['cols'], g['lam']))
                     for g in groups]
            a(f'\n### {o}: {len(lo)} labels in H_PQ -> {len(groups)} groups (proportional columns merged), '
              f'rank {grank}; P-Q fit residual {gres:.1e} Eh')
            a('| group | labels (coefficient relative to the first) | value (Eh) |')
            a('|---|---|---|')
            for k, (g, gn) in enumerate(zip(groups, gname)):
                a(f'| g{k} | {gn} | {gv[k]:+.10f} |')
            # pair vectors: unit (Delta = 1) and contribution (fitted values, Delta_i)
            Wg = {}
            for (n, t), d in T.items():
                Wo = WPQ[(n, t)].get(o, {})
                Wg[(n, t)] = [Wo.get(lo[g['cols'][0]], np.zeros((len(d['Pi']), len(d['Q'][o])))) for g in groups]
            pairs = {}
            for k1 in range(len(groups)):
                for k2 in range(k1, len(groups)):
                    def f(Wl, k1=k1, k2=k2):
                        return -(Wl[k1] @ Wl[k2].T) if k1 == k2 else -(Wl[k1] @ Wl[k2].T + Wl[k2] @ Wl[k1].T)
                    Eu, Ec = {}, {}
                    for (n, t), d in T.items():
                        two = (n == 3 and t == '2D')
                        Mu = f(Wg[(n, t)])
                        Eu[(n, t)] = pack(Mu, two)
                        Ec[(n, t)] = pack(gv[k1] * gv[k2] * Mu / Dorb[(n, o)], two)
                    unit = decomp(Eu, U)[0]          # R is linear and homogeneous in the input
                    contrib = decomp(Ec, U)[0]
                    pairs[(k1, k2)] = (unit, contrib)
            tot = sum(c for _, c in pairs.values())
            Ro = js_orb_vec(js, X, o)
            a(f'Sum over group pairs vs the B2 contribution of {o}: max |diff| {1e3 * np.abs(tot - Ro).max():.1e} mEh')
            a('| pair | rms contribution (mEh) | <c,R_o>/<R_o,R_o> | unit-integral R3 (q1..q4), Delta = 1 Eh |')
            a('|---|---|---|---|')
            for (k1, k2), (un, c) in sorted(pairs.items(), key=lambda kv: -rms(kv[1][1])):
                if rms(c) <= 1e-6 * rms(tot):
                    continue
                a(f'| g{k1} x g{k2} | {1e3 * rms(c):.6f} | {P.ip(c, Ro) / P.ip(Ro, Ro):+.4f} | '
                  + ', '.join(f'{P.ip(un, q):+.5e}' for q in P.Q3) + ' |')
            out[o] = dict(labels=lo, groups=gname, values=gv.tolist(), fit_residual=gres,
                          pairs={f'g{k1} x g{k2}': dict(unit_R3=[P.ip(un, q) for q in P.Q3], unit_R=un.tolist(),
                                                         contribution_rms=rms(c))
                                 for (k1, k2), (un, c) in pairs.items()})
        js['paths'][X]['labels'] = out


def js_orb_vec(js, X, o):
    return np.asarray(js['paths'][X]['_orbvec'][o])


def gcheck(w, js):
    """Virtual f, g, h vectors of vq_orb set 2 vs eigenvectors of the bare-nucleus h = T - 23/r."""
    a = w.append
    NBF = list(gen_v.nbf('vq'))
    V = read_vectors(os.path.join(ROOT, 'runs', 'vq_orb', 'be.NO'), NBF)[1]
    fns = gto.read_bas(os.path.join(_C.BASIS, gen_v.BASES['vq']))
    a('\n# Bare-nucleus check: orbital vectors (vq_orb set 2) vs eigenvectors of h = T - 23/r in the same basis')
    a('| l | nbf | max |offdiag C^T h C| | max |diag C^T h C - eps| | max (1 - |overlap|) | eps (Eh) |')
    a('|---|---|---|---|---|---|')
    out = {}
    for l in (3, 4, 5):
        M = gto.matrices(fns, l, 23)
        h = M['T'] + M['V']
        S = M['S']
        from scipy.linalg import eigh
        eps, Ce = eigh(h, S)
        C = V[l]
        hC = C.T @ h @ C
        off = float(np.abs(hC - np.diag(np.diag(hC))).max())
        ddev = float(np.abs(np.sort(np.diag(hC)) - eps).max())
        ov = np.abs(C.T @ S @ Ce)
        dev = float(np.abs(1 - ov.max(1)).max())
        out[LET[l]] = dict(offdiag=off, diag_dev=ddev, overlap_dev=dev, eps=eps.tolist())
        a(f'| {LET[l]} | {len(eps)} | {off:.1e} | {ddev:.1e} | {dev:.1e} | ' + ', '.join(f'{x:.5f}' for x in eps) + ' |')
    js['bare_nucleus'] = out


if __name__ == '__main__':
    main()
