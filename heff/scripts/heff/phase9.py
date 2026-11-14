#!/usr/bin/env python3
"""Phase 9: vanadium d^1..d^3 ladder ([Ar] frozen), two-body prediction of d^3 and the three-body residual.

Runs v_<ion>_<term>_<orb> (scripts/gen_v.py), one complete-space valence full CI per term; the
levels are subspace CIs on its CSF list, selected by configuration:
  a  3d^n only
  b  3d/4s complex: configurations within {3d, 4s} (4s = outer 11s = IVO of 3d2(3F)4s 4F, v_orb.py)
  s  b + all configurations one electron away from a b configuration (all couplings)
  d  b + all configurations within two electrons of b (for d^2 this is already full)
  c  valence full CI
Targets: lowest root (checked to be 3d^n-dominated); for d^3 2D the 2x2 des Cloizeaux H_eff on the
3d^3 2D model space from the two roots of largest model-space weight, transformed to the seniority
basis of atomci_heff/cfp_l.py (identified at level a, see seniority_map).

Decomposition (standard definition of Phase 8, d shell):
  E_av(n) over all terms, weights g = (2L+1)(2S+1) (2D twice); d2(T) = E(d2 T) - E_av(2);
  pred(t) = E_av(3) + 3 sum_T W(t,T) d2(T)   (2x2 for 2D); t3'(t) = E(t) - pred(t).
Two-body parameters on d^2 (exact, 5 terms): E(T) = A' + B b_T + C c_T + alpha L(L+1) + beta Q_T,
  b, c = Racah coefficients (3F -8,0; 3P 7,0; 1G 4,2; 1D -3,2; 1S 14,7), Q = A00^dag A00 with the
  normalized 1S pair (eigenvalue (n-v)(4l+4-n-v)/(4(2l+1)): 1 for d2 1S, 0.8 for d3 2D(v=1), 0 else).
Refit (step 4): the 9 numbers of d^3 (6 terms + 2D 2x2) as vectors with the weighted inner product
  <x,y> = sum_single g x y + 10 (x11 y11 + x33 y33 + 2 x13 y13)  (= 10 x Frobenius on the 2D block,
  invariant under rotations of the 2D basis); mean-free part has 8 components; fit with the d^3
  response of dB, dC, dalpha, dbeta; residual = genuinely three-body.

The three-body basis (4 vectors, weighted-orthonormal complement of 1, B, C, alpha, beta) is fixed
in results/phase9/threebody_basis.json; the code recomputes it and asserts equality.

Usage: scripts/phase9.py [b]     b = basis prefix of scripts/gen_v.py: v (TZP, default) or vq (QZP);
                                  output results/phase9/phase9.txt/.json (v) or phase9_qzp.txt/.json (vq)
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import cfp, cfp_l as C  # noqa: E402
from atomci_heff import gto, integrals as I  # noqa: E402
from atomci_heff.reader import read_vectors, records  # noqa: E402
from atomci_heff.sparse import SymLower  # noqa: E402

_C.setup_path()
import gen_v  # noqa: E402

import struct  # noqa: E402

EV = 27.211386
T2 = ['3F', '3P', '1G', '1D', '1S']
T3 = ['4F', '4P', '2H', '2G', '2F', '2P']           # single d^3 terms; 2D handled as a block
LCH = 'SPDFGH'
ION = {1: 'v4p', 2: 'v3p', 3: 'v2p'}
LEVELS = ['a', 'b', 's', 'd', 'c']
LNAME = {'a': '3d^n', 'b': '3d/4s complex', 's': 'b + singles', 'd': 'b + doubles', 'c': 'full CI'}
RB = {'3F': -8, '3P': 7, '1G': 4, '1D': -3, '1S': 14}
RC = {'3F': 0, '3P': 0, '1G': 2, '1D': 2, '1S': 7}


def LS(t):
    return LCH.index(t[1]), int(t[0]) - 1


def g(t):
    L, twoS = LS(t)
    return (2 * L + 1) * (twoS + 1)


def occ(label):
    return {m.group(1): int(m.group(2)) for m in (re.match(r'(\d+[spdfgh])(\d+)', s) for s in label.split())}


def dist(a, b):
    return sum(max(a[k] - b.get(k, 0), 0) for k in a)


def subsets(labels, n):
    cfg = {lb: occ(lb) for lb in set(labels)}
    bset = [o for o in cfg.values() if set(o) <= {'3d', '11s'}]
    lev = {}
    for lb, o in cfg.items():
        if o == {'3d': n}:
            lev[lb] = 'a'
        elif set(o) <= {'3d', '11s'}:
            lev[lb] = 'b'
        else:
            dmin = min(dist(o, b) for b in bset)
            lev[lb] = 's' if dmin <= 1 else ('d' if dmin <= 2 else 'c')
    order = {k: i for i, k in enumerate(LEVELS)}
    out = {L: np.array([i for i, lb in enumerate(labels) if order[lev[lb]] <= order[L]]) for L in LEVELS}
    out['c_nog'] = np.array([i for i, lb in enumerate(labels) if 'g' not in lb])   # full CI without any g
    # virtual channels of level s (Phase 9c): a level-s configuration outside b has exactly one electron
    # outside {3d, 4s}; its l (s, d or g by parity) is the channel. S_<chs> = b + singles into those l.
    chan = {}
    for lb, o in cfg.items():
        if lev[lb] == 's':
            ext = [k for k in o if k not in ('3d', '11s')]
            assert len(ext) == 1 and o[ext[0]] == 1, (lb, o)
            assert ext[0][-1] in 'sdg', lb
            chan[lb] = ext[0][-1]
    for chs in ('s', 'd', 'g', 'sd', 'sg', 'dg', 'sdg'):
        out[f'S_{chs}'] = np.array([i for i, lb in enumerate(labels)
                                    if lev[lb] in ('a', 'b') or (lev[lb] == 's' and chan[lb] in chs)])
    return out


def load(run):
    z = np.load(os.path.join(ROOT, 'results', f'{run}.npz'))
    labels = [str(s) for s in z['labels']]
    H = SymLower.from_npz(z).dense()
    e37 = struct.unpack('<d', records(os.path.join(ROOT, 'runs', run, 'fort.37'))[1][8:16])[0]
    out = open(os.path.join(ROOT, 'runs', run, 'be.out'), errors='replace').read()
    eci = [float(x) for x in re.findall(r'STATE\s+CI ENERGY =\s+(-?\d+\.\d+)', out)]
    return labels, H, e37, eci


def cls(o, n):
    d3 = o.get('3d', 0)
    s = sum(q for k, q in o.items() if k.endswith('s'))
    if d3 == n:
        return 'dn'
    if d3 == n - 1 and s == 1 and sum(o.values()) == n:
        return 'dn-1 s'
    return 'other'


def solve_term(labels, H, n, idx, two=False):
    Hs = H[np.ix_(idx, idx)]
    e, Cv = np.linalg.eigh(Hs)
    lab = [labels[i] for i in idx]
    P = [k for k, lb in enumerate(lab) if occ(lb) == {'3d': n}]
    wP = (Cv[P] ** 2).sum(0)
    if not two:
        assert wP[0] > 0.5, (n, wP[0])
        return dict(E=float(e[0]), w=float(wP[0]), e=e, C=Cv, lab=lab)
    roots = sorted(np.argsort(-wP)[:2].tolist())
    X = Cv[np.ix_(P, roots)]
    u, sv, vt = np.linalg.svd(X)
    Xt = u @ vt                                                  # X (X^T X)^(-1/2)
    Heff = Xt @ np.diag(e[roots]) @ Xt.T
    return dict(Heff=Heff, roots=roots, E2=e[roots].tolist(), w2=wP[roots].tolist(), condX=float(sv[0] / sv[-1]),
                e=e, C=Cv, lab=lab, P_labels=[lab[k] for k in P])


# ---------------------------------------------------------------- d-shell algebra
W3 = {t: C.pair_weights(2, 3, *LS(t)) for t in T3 + ['2D']}
T2KEY = {nm: T for T, nm in C.D2_TERMS.items()}
QD2 = {'3F': 0.0, '3P': 0.0, '1G': 0.0, '1D': 0.0, '1S': 1.0}
LLD2 = {t: LS(t)[0] * (LS(t)[0] + 1) for t in T2}


def pairsum(x):
    """d^3 blocks of the two-body operator with d^2 values x(T): {t: float, '2D': 2x2}."""
    out = {t: 3 * sum(W3[t][T2KEY[T]][0, 0] * x[T] for T in T2) for t in T3}
    out['2D'] = 3 * sum(W3['2D'][T2KEY[T]] * x[T] for T in T2)
    return out


GV = np.array([g(t) for t in T3] + [10, 10, 20], float)


def tovec(b):
    return np.array([b[t] for t in T3] + [b['2D'][0, 0], b['2D'][1, 1], b['2D'][0, 1]])


def ip(x, y):
    return float(np.sum(GV * x * y))


def meanv(x):
    return (sum(g(t) * x[i] for i, t in enumerate(T3)) + 10 * (x[6] + x[7])) / 120


def meanfree(x):
    m = meanv(x)
    y = x.copy()
    y[:8] -= m
    return y


ONE = np.array([1.0] * 8 + [0.0])


def d2_params(E2):
    M = np.array([[1, RB[T], RC[T], LLD2[T], QD2[T]] for T in T2], float)
    return dict(zip(['A', 'B', 'C', 'alpha', 'beta'], np.linalg.solve(M, np.array([E2[T] for T in T2]))))


RESP = {'B': pairsum(RB), 'C': pairsum(RC), 'alpha': pairsum(LLD2), 'beta': pairsum(QD2)}
RVEC = {k: meanfree(tovec(v)) for k, v in RESP.items()}


def threebody_basis():
    vs = [ONE / np.sqrt(ip(ONE, ONE))]
    for v in [RVEC[k] for k in ('B', 'C', 'alpha', 'beta')] + list(np.eye(9)):
        w = v.copy()
        for q in vs:
            w = w - ip(q, w) * q
        nrm = np.sqrt(ip(w, w))
        if nrm > 1e-8:
            vs.append(w / nrm)
    assert len(vs) == 9
    return vs[5:]


def fixed_threebody_basis():
    path = os.path.join(ROOT, 'results', 'phase9', 'threebody_basis.json')
    Q = threebody_basis()
    if os.path.exists(path):
        F = [np.array(v) for v in json.load(open(path))['vectors']]
        assert all(np.abs(a - b).max() < 1e-12 for a, b in zip(F, Q)), 'three-body basis changed'
        return F
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(components=T3 + ['2D 11', '2D 33', '2D 13'], weights=GV.tolist(),
                   construction='weighted Gram-Schmidt of 1, r_B, r_C, r_alpha, r_beta, then unit vectors e1..e9; '
                                'the last 4 vectors', vectors=[q.tolist() for q in Q]), open(path, 'w'), indent=1)
    return Q


Q3 = fixed_threebody_basis()


def wangle(x, y):
    """Angle (degrees) between two 9-vectors in the weighted metric."""
    c = ip(x, y) / np.sqrt(ip(x, x) * ip(y, y))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def seniority_map(Ha, Pa):
    """Signed permutation U with U^T Ha U = Pa (both 2x2, mean-free compared); returns U, mismatch."""
    best = None
    for perm in ([0, 1], [1, 0]):
        for s in ([1, 1], [1, -1]):
            U = np.zeros((2, 2))
            U[perm[0], 0], U[perm[1], 1] = s[0], s[1]
            r = np.abs(U.T @ Ha @ U - Pa).max()
            if best is None or r < best[1]:
                best = (U, r)
    return best


def angle(M):
    w, V = np.linalg.eigh(M)
    v = V[:, 0] * np.sign(V[0, 0] if abs(V[0, 0]) > 1e-12 else V[1, 0])
    return float(np.degrees(np.arctan2(v[1], v[0])))


# ---------------------------------------------------------------- E(d^0): frozen-core energy outside ATOMCI
def core_energy(b='v', Z=23):
    fns = gto.read_bas(os.path.join(_C.BASIS, gen_v.BASES[b]))
    V = read_vectors(os.path.join(ROOT, 'runs', f'{b}_hf_v2', 'be.NO'), list(gen_v.nbf(b)))[0]
    x, r = I.radial_grid()
    orbs = {'1s': (0, 0), '2s': (0, 1), '3s': (0, 2), '2p': (1, 0), '3p': (1, 1), '3d': (2, 0)}
    P = I.radial_functions(fns, {l: V[l] for l in range(3)}, r)
    Pf = {o: P[l][k] for o, (l, k) in orbs.items()}
    h = {}
    for o, (l, k) in orbs.items():
        M = gto.matrices(fns, l, Z)
        h[o] = float(V[l][:, k] @ (M['T'] + M['V']) @ V[l][:, k])
    q = {'1s': 2, '2s': 2, '3s': 2, '2p': 6, '3p': 6}
    lv = {o: orbs[o][0] for o in orbs}

    def Fk(a, b, k):
        return I.slater(Pf[a], Pf[a], Pf[b], Pf[b], k, x, r)

    def Gk(a, b, k):
        return I.slater(Pf[a], Pf[b], Pf[b], Pf[a], k, x, r)

    def e_same(a):
        la = lv[a]
        return Fk(a, a, 0) - (2 * la + 1) / (4 * la + 1) * sum(
            cfp.threej(la, k, la, 0, 0, 0) ** 2 * Fk(a, a, k) for k in range(2, 2 * la + 1, 2))

    def e_pair(a, b):
        la, lb = lv[a], lv[b]
        return Fk(a, b, 0) - 0.5 * sum(cfp.threej(la, k, lb, 0, 0, 0) ** 2 * Gk(a, b, k)
                                       for k in range(abs(la - lb), la + lb + 1, 2))

    core = list(q)
    Ec = sum(q[a] * h[a] for a in core) + sum(q[a] * (q[a] - 1) / 2 * e_same(a) for a in core) + \
        sum(q[a] * q[b] * e_pair(a, b) for i, a in enumerate(core) for b in core[i + 1:])
    # check: single-CSF V2+ 3d3 4F = E_core + 3 h_3d + 3 sum q_a E(a,3d) + E_dd(4F)
    Fdd = {k: Fk('3d', '3d', k) for k in (0, 2, 4)}
    Edd = float(C.slater_condon(2, 3, 3, 3, Fdd)[0, 0])
    E4F = Ec + 3 * h['3d'] + 3 * sum(q[a] * e_pair(a, '3d') for a in core) + Edd
    return Ec, E4F


# ---------------------------------------------------------------- analysis
def analyze(orb, levels, b='v'):
    runs = {(1, '2D'): f'{b}_v4p_2De_{orb}'}
    runs.update({(2, t): f'{b}_v3p_{t}e_{orb}' for t in T2})
    runs.update({(3, t): f'{b}_v2p_{t}e_{orb}' for t in T3 + ['2D']})
    data, ver = {}, []
    for (n, t), run in runs.items():
        labels, H, e37, eci = load(run)
        sub = subsets(labels, n)
        assert len(sub['c']) == len(labels)
        two = (n == 3 and t == '2D')
        data[(n, t)] = {L: solve_term(labels, H, n, sub[L], two) for L in levels}
        if 'c' not in levels:
            continue
        full = data[(n, t)]['c']
        ver.append(dict(run=run, N=len(labels), d_e37=float(abs(full['e'][0] - e37)),
                        d_root2=(float(abs(full['e'][1] - eci[1])) if two else None),
                        sizes={L: int(len(sub[L])) for L in levels}))
    res = {}
    for L in levels:
        E1 = data[(1, '2D')][L]['E']
        E2 = {t: data[(2, t)][L]['E'] for t in T2}
        Eav2 = sum(g(t) * E2[t] for t in T2) / 45
        d2 = {t: E2[t] - Eav2 for t in T2}
        par = d2_params(E2)
        H2D = data[(3, '2D')][L]['Heff']
        E3 = {t: data[(3, t)][L]['E'] for t in T3}
        Eav3 = (sum(g(t) * E3[t] for t in T3) + 10 * np.trace(H2D)) / 120
        pred = pairsum(d2)
        pv = tovec(pred)
        assert abs(meanv(pv)) < 1e-10, meanv(pv)
        res[L] = dict(E1=E1, E2=E2, Eav2=Eav2, d2=d2, par=par, H2D_atomci=H2D, E3=E3, Eav3=Eav3, pred=pred)
    # seniority identification at level a
    a = res['a'] if 'a' in res else None
    if a is not None:
        Pa = a['pred']['2D'] + a['Eav3'] * np.eye(2)
        U, mis = seniority_map(a['H2D_atomci'], Pa)
        Ha_sen = U.T @ a['H2D_atomci'] @ U
        eig_mis = float(np.abs(np.linalg.eigvalsh(a['H2D_atomci']) - np.linalg.eigvalsh(Pa)).max())
        json.dump(U.tolist(), open(os.path.join(ROOT, 'results', 'phase9', 'seniority_U.json' if b == 'v' else f'seniority_U_{b}.json'), 'w'))
    else:
        U = np.array(json.load(open(os.path.join(ROOT, 'results', 'phase9', 'seniority_U.json' if b == 'v' else f'seniority_U_{b}.json'))))
        mis = eig_mis = None
    for L in levels:
        r = res[L]
        H = U.T @ r['H2D_atomci'] @ U
        r['H2D'] = H
        ex = {**r['E3'], '2D': H}
        t3 = {t: ex[t] - r['Eav3'] - r['pred'][t] for t in T3}
        t3['2D'] = H - r['Eav3'] * np.eye(2) - r['pred']['2D']
        tv = tovec(t3)
        assert abs(meanv(tv)) < 1e-9
        r['t3'] = t3
        G = np.array([[ip(RVEC[i], RVEC[j]) for j in RVEC] for i in RVEC])
        b = np.array([ip(RVEC[i], tv) for i in RVEC])
        dpar = np.linalg.solve(G, b)
        fit = sum(c * RVEC[k] for c, k in zip(dpar, RVEC))
        R = tv - fit
        r['dpar'] = dict(zip(RVEC, dpar))
        r['R'] = R
        r['rms_t3'] = np.sqrt(ip(tv, tv) / 120)
        r['rms_R'] = np.sqrt(ip(R, R) / 120)
        r['frac_R'] = np.sqrt(ip(R, R) / ip(tv, tv)) if ip(tv, tv) > 0 else 0.0
        r['R3'] = [ip(R, q) for q in Q3]
        # term energies relative to 4F: exact, two-body prediction, refit model
        def rel(blocks):
            e2d = np.linalg.eigvalsh(blocks['2D'])
            out = {t: blocks[t] - blocks['4F'] for t in T3}
            out['2D-'], out['2D+'] = e2d[0] - blocks['4F'], e2d[1] - blocks['4F']
            return out
        pb = {t: r['Eav3'] + r['pred'][t] for t in T3}
        pb['2D'] = r['Eav3'] * np.eye(2) + r['pred']['2D']
        fb = {t: pb[t] + fit[i] for i, t in enumerate(T3)}
        fb['2D'] = pb['2D'] + np.array([[fit[6], fit[8]], [fit[8], fit[7]]])
        r['rel_exact'], r['rel_pred'], r['rel_fit'] = rel(ex), rel(pb), rel(fb)
        r['angle'] = dict(exact=angle(H), pred=angle(pb['2D']), fit=angle(fb['2D']))
    return data, res, ver, dict(U=U.tolist(), mismatch=mis, eig_mismatch=eig_mis)


def step1(data):
    rows = []
    for (n, t), lv in data.items():
        c = lv['c']
        e, Cv, lab = c['e'], c['C'], c['lab']
        cl = np.array([cls(occ(x), n) for x in lab])
        wd = (Cv[cl == 'dn'] ** 2).sum(0)
        ws = (Cv[cl == 'dn-1 s'] ** 2).sum(0)
        tgt = c['roots'] if 'roots' in c else [0]
        first_s = next((i for i in range(len(e)) if ws[i] > 0.5), None)
        rows.append(dict(term=f'd{n} {t}', w=[float(wd[i]) for i in tgt],
                         s_root=first_s, s_gap_eV=(float((e[first_s] - e[tgt[0]]) * EV) if first_s is not None else None),
                         s_w=(float(ws[first_s]) if first_s is not None else None), condX=c.get('condX')))
    return rows


def main(b='v'):
    m = lambda x: f'{1e3 * x:+.3f}'          # noqa: E731
    out = []
    w = out.append
    data, res, ver, sen = analyze('ivo', LEVELS, b)
    Ec, E4F = core_energy(b)
    hf = float(np.load(os.path.join(ROOT, 'results', f'{b}_hf_v2.npz'))['H'][0, 0])
    w(f'# Phase 9: V d^1..d^3 ladder, basis {gen_v.BASES[b]}, [Ar] frozen in the HF orbitals of V2+ 3d3 4F '
      '(4s = IVO), complete valence space')
    w('Energies in mEh unless stated. Levels: ' + '; '.join(f'{k} = {v}' for k, v in LNAME.items()) + '.')
    w('\n## Verification')
    w('| run | N | |E0 - ATOMCI| | |E1 - ATOMCI| (2D) | CSFs a/b/s/d/c |')
    w('|---|---|---|---|---|')
    for v in ver:
        r2s = '' if v['d_root2'] is None else f"{v['d_root2']:.1e}"
        w(f'| {v["run"]} | {v["N"]} | {v["d_e37"]:.1e} | {r2s} | '
          + '/'.join(str(v['sizes'][L]) for L in LEVELS) + ' |')
    w(f'2D seniority basis from level a: U = {np.round(np.array(sen["U"]), 6).tolist()} (rows: ATOMCI CSFs in H order); '
      f'max |U^T H_a U - pred| = {sen["mismatch"]:.1e} Eh, eigenvalue mismatch {sen["eig_mismatch"]:.1e} Eh '
      '(the sign of the off-diagonal element is fixed to the cfp_l convention).')
    w(f'E(d^0) outside ATOMCI: E_core = {Ec:.8f} Eh; check: rebuilt V2+ 3d3 4F single CSF {E4F:.8f} vs ATOMCI {hf:.8f} '
      f'(diff {E4F - hf:+.1e} Eh).')

    w('\n## Step 1: model-space quality (full CI)')
    w('| target | w(3d^n) | lowest 3d^(n-1) ns root of the same symmetry: index, gap to target (eV), weight | cond(X) |')
    w('|---|---|---|---|')
    for r in step1(data):
        w(f'| {r["term"]} | ' + '/'.join(f'{x:.4f}' for x in r['w']) + ' | '
          + ('none' if r['s_root'] is None else f'#{r["s_root"]}, {r["s_gap_eV"]:+.2f}, {r["s_w"]:.3f}')
          + ' | ' + ('' if r['condX'] is None else f"{r['condX']:.4f}") + ' |')

    w('\n## Step 3: decomposition by level')
    w('| level | E(d1) Eh | E_av(2) Eh | E_av(3) Eh | ' + ' | '.join(f'd2({t})' for t in T2) + ' |')
    w('|---|---|---|---|' + '---|' * 5)
    for L in LEVELS:
        r = res[L]
        w(f'| {L} | {r["E1"]:.8f} | {r["Eav2"]:.8f} | {r["Eav3"]:.8f} | ' + ' | '.join(m(r['d2'][t]) for t in T2) + ' |')
    if Ec is not None:
        w('\nThird difference of E_av (E_av(0) = E_core, the same for all levels):')
        for L in LEVELS:
            r = res[L]
            d3 = r['Eav3'] - 3 * r['Eav2'] + 3 * r['E1'] - Ec
            w(f'  {L}: E_av(3) - 3 E_av(2) + 3 E_av(1) - E_av(0) = {m(d3)}')
            r['third_diff'] = d3
    w('\nd^2 parameters: B, C (Racah), alpha (L(L+1)), beta (Q pairing, 1 on d2 1S):')
    w('| level | B | C | C/B | alpha | beta |')
    w('|---|---|---|---|---|---|')
    for L in LEVELS:
        p = res[L]['par']
        w(f'| {L} | {1e3 * p["B"]:.4f} | {1e3 * p["C"]:.4f} | {p["C"] / p["B"]:.4f} | {1e3 * p["alpha"]:+.4f} | {1e3 * p["beta"]:+.4f} |')
    w('\nt3\'(t), seniority basis for 2D (11 = v1, 33 = v3):')
    w('| level | ' + ' | '.join(T3) + ' | 2D 11 | 2D 33 | 2D 13 | rms t3\' |')
    w('|---|' + '---|' * 10)
    for L in LEVELS:
        t = res[L]['t3']
        w(f'| {L} | ' + ' | '.join(m(t[x]) for x in T3) + f' | {m(t["2D"][0, 0])} | {m(t["2D"][1, 1])} | {m(t["2D"][0, 1])} | '
          f'{1e3 * res[L]["rms_t3"]:.3f} |')
    w('\nd^3 term energies relative to 4F (exact / two-body prediction from d^2 / error), mEh; 2D- and 2D+ the 2D eigenvalues:')
    keys = [t for t in T3 if t != '4F'] + ['2D-', '2D+']
    w('| level | ' + ' | '.join(keys) + ' | 2D angle exact / pred (deg) |')
    w('|---|' + '---|' * (len(keys) + 1))
    for L in LEVELS:
        r = res[L]
        w(f'| {L} | ' + ' | '.join(f'{1e3 * r["rel_exact"][k]:.2f} / {1e3 * r["rel_pred"][k]:.2f} / '
                                   f'{1e3 * (r["rel_exact"][k] - r["rel_pred"][k]):+.2f}' for k in keys)
          + f' | {r["angle"]["exact"]:+.3f} / {r["angle"]["pred"]:+.3f} |')

    w('\n## Step 4: two-body refit (dB, dC, dalpha, dbeta from d^2 to d^3) and the three-body residual')
    w('| level | dB (frac of B) | dC (frac of C) | dalpha | dbeta | rms t3\' | rms residual | residual / t3\' (norm) |')
    w('|---|---|---|---|---|---|---|---|')
    for L in LEVELS:
        r = res[L]
        dp, p = r['dpar'], r['par']
        fr = lambda k: (f'{100 * dp[k] / p[k]:+.1f}%' if abs(p[k]) > 1e-9 else '--')   # noqa: E731
        w(f'| {L} | {1e3 * dp["B"]:+.4f} ({fr("B")}) | {1e3 * dp["C"]:+.4f} ({fr("C")}) | '
          f'{1e3 * dp["alpha"]:+.4f} ({fr("alpha")}) | {1e3 * dp["beta"]:+.4f} ({fr("beta")}) | '
          f'{1e3 * r["rms_t3"]:.4f} | {1e3 * r["rms_R"]:.4f} | '
          + (f'{100 * r["frac_R"]:.1f}% |' if r['rms_t3'] > 1e-9 else '-- |'))
    w('\nResidual components (mEh) and its coordinates in an orthonormal three-body basis of my own '
      '(weighted Gram-Schmidt of the unit vectors after 1, B, C, alpha, beta; NOT Judd\'s t_i):')
    w('| level | ' + ' | '.join(T3) + ' | 2D 11 | 2D 33 | 2D 13 | q1 | q2 | q3 | q4 |')
    w('|---|' + '---|' * 13)
    for L in LEVELS:
        R = res[L]['R']
        w(f'| {L} | ' + ' | '.join(m(x) for x in R) + ' | ' + ' | '.join(f'{1e3 * x:+.3f}' for x in res[L]['R3']) + ' |')
    w('\nResidual as errors of the d^3 term energies relative to 4F (exact - refit model), mEh:')
    w('| level | ' + ' | '.join(keys) + ' | 2D angle exact / refit (deg) |')
    w('|---|' + '---|' * (len(keys) + 1))
    for L in LEVELS:
        r = res[L]
        w(f'| {L} | ' + ' | '.join(f'{1e3 * (r["rel_exact"][k] - r["rel_fit"][k]):+.3f}' for k in keys)
          + f' | {r["angle"]["exact"]:+.3f} / {r["angle"]["fit"]:+.3f} |')

    # checks
    w('\n## Step 5: checks')
    e_g = data[(3, '4F')]['c']['E']
    # no g at all: the g orbitals are pure virtuals (no g in the core or in the HF 3d^3), so removing the
    # g functions equals the full CI on the CSFs without any g electron (a run with NOB(g)=0 stops in
    # CIDRIV, return code 7135)
    labels, H, _, _ = load(f'{b}_v2p_4Fe_ivo')
    keep = np.array([i for i, lb in enumerate(labels) if 'g' not in lb])
    e_nog = float(np.linalg.eigvalsh(H[np.ix_(keep, keep)])[0])
    n_nog = len(keep)
    w(f'g functions: E(4F) full CI with g {e_g:.8f} Eh ({len(labels)} CSFs, g^3 excluded), without any g {e_nog:.8f} Eh '
      f'({n_nog} CSFs): all g configurations together contribute {m(e_g - e_nog)}; the omitted g^3 configurations '
      '(counts in phase9b.txt) are a part of this.')
    _, rng, _, _ = analyze('ivo', ['c_nog'], b)
    rn = rng['c_nog']
    w('Full CI without any g, all terms (bounds the g^3 omission on the quantities of steps 3-4):')
    w('| | B | C | alpha | beta | dB | dC | dalpha | dbeta | rms t3\' | rms residual | q1..q4 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for nm, r in (('with g (g^3 excluded)', res['c']), ('no g at all', rn)):
        p, dp = r['par'], r['dpar']
        w(f'| {nm} | ' + ' | '.join(f'{1e3 * p[k]:.4f}' for k in ('B', 'C', 'alpha', 'beta')) + ' | '
          + ' | '.join(f'{1e3 * dp[k]:+.4f}' for k in ('B', 'C', 'alpha', 'beta'))
          + f' | {1e3 * r["rms_t3"]:.4f} | {1e3 * r["rms_R"]:.4f} | ' + ', '.join(f'{1e3 * x:+.3f}' for x in r['R3']) + ' |')
    w(f'Angle between the residuals with and without g (weighted metric): {wangle(res["c"]["R"], rn["R"]):.1f} deg')
    r3c, v3c = {}, []
    if b == 'v':
        d3c, r3c, v3c, _ = analyze('hfv3', ['c'], b)
    if r3c:
        rc, r2 = res['c'], r3c['c']
        w('Core from V3+ 3d2 3F HF (full CI only):')
        w('| core | B | C | alpha | beta | dB | dC | dalpha | dbeta | rms t3\' | rms residual | residual / t3\' | q1..q4 |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for nm, r in (('V2+ 4F', rc), ('V3+ 3F', r2)):
            p, dp = r['par'], r['dpar']
            w(f'| {nm} | ' + ' | '.join(f'{1e3 * p[k]:.4f}' for k in ('B', 'C', 'alpha', 'beta')) + ' | '
              + ' | '.join(f'{1e3 * dp[k]:+.4f}' for k in ('B', 'C', 'alpha', 'beta'))
              + f' | {1e3 * r["rms_t3"]:.4f} | {1e3 * r["rms_R"]:.4f} | {100 * r["frac_R"]:.1f}% | '
              + ', '.join(f'{1e3 * x:+.3f}' for x in r['R3']) + ' |')
        w('V3+-core verification: ' + ', '.join(f'{v["run"].split("_")[2]} {v["d_e37"]:.0e}' for v in v3c))
    w('\nResidual rms by level (excitation classes): ' + ', '.join(f'{L} {1e3 * res[L]["rms_R"]:.4f}' for L in LEVELS))

    os.makedirs(os.path.join(ROOT, 'results', 'phase9'), exist_ok=True)
    tag = 'phase9' if b == 'v' else 'phase9_qzp'
    open(os.path.join(ROOT, 'results', 'phase9', f'{tag}.txt'), 'w').write('\n'.join(out) + '\n')

    def js(x):
        if isinstance(x, dict):
            return {str(k): js(v) for k, v in x.items() if k not in ('C', 'e', 'lab')}
        if isinstance(x, (list, tuple)):
            return [js(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x
    json.dump(js(dict(ivo=res, ivo_nog=rn, hfv3=r3c, verification=ver, verification_v3=v3c, seniority=sen, E_core=Ec,
                      E4F_rebuilt=E4F, E4F_g=e_g, E4F_nog=e_nog)),
              open(os.path.join(ROOT, 'results', 'phase9', f'{tag}.json'), 'w'), indent=1)
    print('\n'.join(out))
    return dict(res=res, nog=rn, data=data, ver=ver)


if __name__ == '__main__':
    main(*sys.argv[1:])
