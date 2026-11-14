#!/usr/bin/env python3
"""Measured-side fit of a d^N spectrum with the 15i operators (results/phase15/15i_trees: dn.py, ops.py, e1_d3.npy,
e2_d3.npy), generalised from fitlib.D6 to any N (half-integer J for odd N).

Fits (cm-1; T1, T2 reported in mEh, our normalisation: coefficients of the pure three-body operators of Frobenius
norm 1 in d3, as in 15i section 1):
  (1) J levels:  A + B + C + zeta + alpha + beta + T1 + T2
  (2) LS terms:  (2J+1)-weighted term centroids of the same levels, A + B + C + alpha + beta + T1 + T2, no spin-orbit
  (3) as (1) with T1, T2 fixed (default: best space + all h, 15t besth: 12.211, 1.693 mEh)
Spin-orbit: only the one-body zeta, the only spin-orbit operator the 15i tools have.

usage: phase15u_crfit.py levels.json [T1fix T2fix]   levels.json: [["4F", "3/2", 0.0], ["4F", "5/2", 241.5], ...]
                                                     (term with multiplicity and L letter, optional seniority/second
                                                     index as "2D2" / "2D1" or "a2D"/"b2D", J, E in cm-1)
       phase15u_crfit.py selftest                    synthetic d3 spectrum from known parameters, all three fits"""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
TOOLS = os.path.join(ROOT, 'results', 'phase15', '15i_trees')
sys.path[:0] = [TOOLS]
from dn import onebody_ang, build1  # noqa: E402
from ops import dN_ops  # noqa: E402

H2E = 219.4746        # cm-1 per mEh
LET = 'SPDFGHIK'


class DN:
    def __init__(self, N):
        o = dN_ops(N)
        self.o, self.N = o, N
        lz, lp, sz, sp = (build1(x, o['dets'], o['dets']) for x in onebody_ang(o['sp'])[:4])
        self.L2, self.S2, self.lz, self.sz = o['L2'], o['S2'], lz, sz
        self.J2 = o['L2'] + o['S2'] + 2 * (lz @ sz + 0.5 * (lp @ sp.T + lp.T @ sp))
        jz2 = np.round(2 * np.diag(lz + sz)).astype(int)
        self.blocks = {j2: np.where(jz2 == j2)[0] for j2 in sorted(set(jz2)) if j2 >= 0}
        # Racah pairing operator scaled to Q(N, v) = (N - v)(12 - N - v)/4 at the lowest seniority present above v = N
        P = o['Qpair']
        ev = [x for x in np.unique(np.round(np.linalg.eigvalsh(P), 8)) if x > 1e-8]
        v = N - 2
        self.Q = P * ((N - v) * (12 - N - v) / 4 / ev[0])
        E1, E2 = (np.load(next(f for f in (os.path.join(TOOLS, f'{c}{k}_d{N}.npy') for c in 'eE') if os.path.exists(f)))
                  for k in (1, 2))      # d3: e1/e2 (norm 1); d6: E1/E2 (the same operators lifted, norm 2)
        I = np.eye(len(o['dets']))
        self.ops = {'A': I, 'B': 49 * o['F2'], 'C': 7 * o['F2'] + 12.6 * o['F4'], 'zeta': o['zeta'],
                    'alpha': o['L2'], 'beta': self.Q, 'T1': E1 * H2E, 'T2': E2 * H2E}   # T1, T2 in mEh

    def levels(self, H):
        """{2J: (energies, vectors, idx)} from the M_J = J block."""
        out = {}
        for j2, idx in self.blocks.items():
            w, v = np.linalg.eigh(H[np.ix_(idx, idx)])
            jj = np.einsum('ia,ij,ja->a', v, self.J2[np.ix_(idx, idx)], v)
            J = j2 / 2
            sel = abs(jj - J * (J + 1)) < 1e-4
            out[j2] = (w[sel], v[:, sel], idx)
        return out

    def terms(self, H):
        """{(2S+1, L): (energies, vectors, idx)} from the M_L = L, M_S = S block (H without spin-orbit)."""
        out = {}
        dl, ds = np.round(np.diag(self.lz), 8), np.round(2 * np.diag(self.sz)).astype(int)
        for L in range(9):
            for s2 in range(self.N % 2, self.N + 1, 2):
                idx = np.where((abs(dl - L) < 1e-8) & (ds == s2))[0]
                if not len(idx):
                    continue
                w, v = np.linalg.eigh(H[np.ix_(idx, idx)])
                l2 = np.einsum('ia,ij,ja->a', v, self.L2[np.ix_(idx, idx)], v)
                ss = np.einsum('ia,ij,ja->a', v, self.S2[np.ix_(idx, idx)], v)
                sel = (abs(l2 - L * (L + 1)) < 1e-4) & (abs(ss - s2 / 2 * (s2 / 2 + 1)) < 1e-4)
                if sel.any():
                    out[(s2 + 1, L)] = (w[sel], v[:, sel], idx)
        return out


def lsq(d, names, fixed, obs, get, p0, iters=60):
    """Gauss-Newton on obs = [(key, k, E)]; get(H) -> {key: (w, v, idx)}; states matched by order within a key on the
    first pass, then by overlap with the previous iteration's vectors."""
    ops = d.ops
    p = np.array(p0, float)
    E = np.array([e for *_, e in obs])
    ref = None
    for _ in range(iters):
        H = sum(a * ops[n] for a, n in zip(p, names)) + sum(a * ops[n] for n, a in fixed.items())
        S = get(H)
        ks = []
        for key, k, e in obs:
            w, v, idx = S[key]
            if ref is None:
                ks.append(k)
            else:
                ov = abs(ref[(key, k)] @ v)
                ks.append(int(np.argmax(ov)))
        assert len(set(zip([o[0] for o in obs], ks))) == len(obs), 'state assignment collision'
        calc = np.array([S[key][0][kk] for (key, k, e), kk in zip(obs, ks)])
        Jm = np.array([[S[key][1][:, kk] @ ops[n][np.ix_(S[key][2], S[key][2])] @ S[key][1][:, kk] for n in names]
                       for (key, k, e), kk in zip(obs, ks)])
        dp = np.linalg.lstsq(Jm, E - calc, rcond=None)[0]
        p += dp
        ref = {(key, k): S[key][1][:, kk] for (key, k, e), kk in zip(obs, ks)}
        if abs(dp).max() < 1e-9:
            break
    r = E - calc
    dof = max(len(E) - len(names), 1)
    s = float(np.sqrt(r @ r / dof))
    cov = np.linalg.pinv(Jm.T @ Jm) * s * s
    err = np.sqrt(np.diag(cov))
    return dict(p=dict(zip(names, p)), err=dict(zip(names, err)), rms=s, n=len(E), dof=dof, resid=r.tolist(),
                cov=cov.tolist())


def parse(data):
    """[[term, J, E]] -> levels [(term, 2J, E)] with term = (label, 2S+1, L, index within (2S+1, L) by energy)."""
    lev = []
    for t, J, e in data:
        lab = t
        core = t.lstrip('abcdefgh') if t[0].isalpha() and t[0].islower() else t
        mult, L = int(core[0]), LET.index(core[1])
        lev.append((lab, mult, L, int(2 * Fraction(J)), float(e)))
    order = {}
    for lab, mult, L, *_ in sorted(lev, key=lambda x: x[4]):
        order.setdefault((mult, L), [])
        if lab not in order[(mult, L)]:
            order[(mult, L)].append(lab)
    return [(lab, mult, L, order[(mult, L)].index(lab), j2, e) for lab, mult, L, j2, e in lev]


def run(d, data, fix=(12.211, 1.693)):
    lev = parse(data)
    e0 = min(x[5] for x in lev)
    # (1) J levels: key 2J, index by energy among the observed levels of that J (assumes the observed set is the
    # lowest levels of each J only if complete; the overlap tracking then follows the states)
    byj = {}
    for x in sorted(lev, key=lambda x: x[5]):
        byj.setdefault(x[4], []).append(x)
    obs1 = [(j2, k, x[5]) for j2, xs in byj.items() for k, x in enumerate(xs)]
    names = ['A', 'B', 'C', 'zeta', 'alpha', 'beta', 'T1', 'T2']
    p0 = [e0 + 2.0e4, 1000, 3700, 300, 70, 0, 10, 1.5]
    f1 = lsq(d, names, {}, obs1, d.levels, p0)
    # (2) LS centroids, (2J+1) weights
    cen = {}
    for lab, mult, L, k, j2, e in lev:
        c = cen.setdefault((mult, L, k), [0.0, 0.0, lab])
        c[0] += (j2 + 1) * e
        c[1] += j2 + 1
    obs2 = [((m, L), k, v[0] / v[1]) for (m, L, k), v in sorted(cen.items())]
    n2 = ['A', 'B', 'C', 'alpha', 'beta', 'T1', 'T2']
    f2 = lsq(d, n2, {}, obs2, d.terms, [f1['p'][n] for n in n2])
    # (3) as (1), T1, T2 fixed
    n3 = [n for n in names if n not in ('T1', 'T2')]
    f3 = lsq(d, n3, {'T1': fix[0], 'T2': fix[1]}, obs1, d.levels, [f1['p'][n] for n in n3])
    return dict(fit1=f1, fit2=f2, fit3=f3, fix=fix, n_levels=len(obs1), n_terms=len(obs2))


def show(res):
    for k, lab in (('fit1', '(1) J levels, zeta'), ('fit2', '(2) LS centroids'), ('fit3', '(3) = (1), T1 T2 fixed')):
        f = res[k]
        print(f'{lab:24s} n {f["n"]:3d} dof {f["dof"]:3d} rms {f["rms"]:8.2f} cm-1 | ' +
              ', '.join(f'{n} {v:.3f}' + (f' +- {f["err"][n]:.3f}' if n in f['err'] else '') if n in ('T1', 'T2')
                        else f'{n} {v:.1f}' for n, v in f['p'].items() if n != 'A'))
    print(f'(3) fixed T1, T2 = {res["fix"]}')


def selftest():
    d = DN(3)
    P = dict(A=30000.0, B=1030.0, C=3850.0, zeta=275.0, alpha=60.0, beta=-100.0, T1=11.0, T2=1.5)
    H = sum(v * d.ops[n] for n, v in P.items())
    L = d.levels(H)
    # label every level by its dominant (S, L) term, all levels of d3
    T = d.terms(sum(v * d.ops[n] for n, v in P.items() if n != 'zeta'))
    data = []
    for j2, (w, v, idx) in L.items():
        for a in range(len(w)):
            best = None
            for (m, l), (tw, tv, tidx) in T.items():
                pass
            ls = np.einsum('i,ij,j->', v[:, a], d.L2[np.ix_(idx, idx)], v[:, a])
            ss = np.einsum('i,ij,j->', v[:, a], d.S2[np.ix_(idx, idx)], v[:, a])
            l = int(round((-1 + np.sqrt(1 + 4 * ls)) / 2))
            m = int(round(2 * (-1 + np.sqrt(1 + 4 * ss)) / 2)) + 1
            best = f'{m}{LET[l]}'
            data.append([best, str(Fraction(j2, 2)), float(w[a])])
    # same-(S, L) terms (two 2D in d3): give them a/b labels by the 2J+1-weighted energy order
    cnt = {}
    for x in data:
        cnt.setdefault((x[0], x[1]), []).append(x)
    for (t, J), xs in cnt.items():
        if len(xs) > 1:
            for i, x in enumerate(sorted(xs, key=lambda x: x[2])):
                x[0] = 'ab'[i] + t
    for x in data:
        if x[0][0].isdigit() and any(y[0] == 'b' + x[0] for y in data):
            x[0] = 'a' + x[0]
    print('self-test: synthetic d3 spectrum,', len(data), 'levels; true', P)
    res = run(d, data, fix=(P['T1'], P['T2']))
    show(res)


if __name__ == '__main__':
    if sys.argv[1] == 'selftest':
        selftest()
    else:
        res = run(DN(3), json.load(open(sys.argv[1])), tuple(map(float, sys.argv[2:4])) or (12.211, 1.693))
        show(res)
        json.dump(res, open(os.path.splitext(sys.argv[1])[0] + '_fit.json', 'w'), indent=1)
