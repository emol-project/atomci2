"""Determinant model of the core minimal space {3s2 3d^N, 3s 3d^(N+1)} (12 spin-orbitals: 3s up/down, then the 10 3d
spin-orbitals in cfp_l order).

H = sum_k p_k O_k with operators for the radial parameters
    const, n_s, n_d, F0(ss), F0(sd), G2(sd), F0(dd), F2(dd), F4(dd), R2(sd;dd)
(F^k, G^k, R^k unnormalized, Condon-Shortley Gaunt coefficients; h_sd = 0 by symmetry, so 3s -> 3d couples only through
R^2(3s 3d; 3d 3d)). Blocks are built on the highest-weight states (M_L = L, M_S = S, annihilated by L+ and S+).
P states are |3s2> x the dshell stretched states of d^N (seniority basis, dshell.SIGN); Q states are an orthonormal
basis of the highest-weight 3s 3d^(N+1) space.
"""
import itertools
from functools import lru_cache
from math import sqrt

import numpy as np

from atomci_heff import cfp_l, dshell
from atomci_heff.cfp import ck

PARAMS = ['const', 'n_s', 'n_d', 'F0ss', 'F0sd', 'G2sd', 'F0dd', 'F2dd', 'F4dd', 'R2']
SO = [(0, 0, 1), (0, 0, -1)] + [(2, m, s) for m, s in cfp_l.spin_orbitals(2)]
NO = len(SO)


def _radial(la, lb, lc, ld, k):
    """Parameter name of the radial integral R^k(ab; cd) (electron 1: a -> c, electron 2: b -> d) or None."""
    key = ''.join('sd'[x // 2] for x in (la, lb, lc, ld))
    ns = key.count('s')
    if ns == 4:
        return 'F0ss' if k == 0 else None
    if ns == 0:
        return {0: 'F0dd', 2: 'F2dd', 4: 'F4dd'}.get(k)
    if ns == 2:
        if key in ('sdsd', 'dsds'):
            return 'F0sd' if k == 0 else None
        if key in ('sdds', 'dssd'):
            return 'G2sd' if k == 2 else None
        return None                  # ss;dd and dd;ss (3s2 <-> 3d2): outside the space
    if ns == 3:
        return None
    return 'R2' if k == 2 else None    # one s, three d


@lru_cache(maxsize=None)
def gtensor():
    """{(p, q, r, s): {param: value}} for <pq|1/r12|rs> (electron 1 p -> r, electron 2 q -> s)."""
    out = {}
    for p, q, r, s in itertools.product(range(NO), repeat=4):
        (l1, m1, s1), (l2, m2, s2), (l3, m3, s3), (l4, m4, s4) = SO[p], SO[q], SO[r], SO[s]
        if s1 != s3 or s2 != s4 or m1 + m2 != m3 + m4:
            continue
        d = {}
        for k in (0, 2, 4):
            name = _radial(l1, l2, l3, l4, k)
            if name is None:
                continue
            v = ck(k, l1, m1, l3, m3) * ck(k, l4, m4, l2, m2)
            if abs(v) > 1e-14:
                d[name] = d.get(name, 0.0) + v
        if d:
            out[(p, q, r, s)] = d
    return out


def _ml_ms(det):
    return sum(SO[i][1] for i in det), sum(SO[i][2] for i in det)


@lru_cache(maxsize=None)
def sector(N, ML, twoMS):
    """Dets of P (3s2 3d^N) and Q (3s 3d^(N+1)) with the given M_L, 2M_S."""
    P, Q = [], []
    for D in itertools.combinations(range(2, NO), N):
        d = (0, 1) + D
        if _ml_ms(d) == (ML, twoMS):
            P.append(d)
    if N < 10:
        for s0 in (0, 1):
            for D in itertools.combinations(range(2, NO), N + 1):
                d = (s0,) + D
                if _ml_ms(d) == (ML, twoMS):
                    Q.append(d)
    return P, Q


def _ops_on(dets):
    """{param: matrix} of the parameter operators on the det list."""
    ix = {d: i for i, d in enumerate(dets)}
    n = len(dets)
    M = {k: np.zeros((n, n)) for k in PARAMS}
    by_rs = {}
    for (p, q, r, s), d in gtensor().items():
        by_rs.setdefault((r, s), []).append((p, q, d))
    for j, D in enumerate(dets):
        ns = sum(1 for x in D if x < 2)
        M['const'][j, j] += 1
        M['n_s'][j, j] += ns
        M['n_d'][j, j] += len(D) - ns
        for r, s in itertools.permutations(D, 2):
            for p, q, d in by_rs.get((r, s), ()):
                res = cfp_l._apply([('c', p), ('c', q), ('a', s), ('a', r)], D)
                if res and res[1] in ix:
                    for k, v in d.items():
                        M[k][ix[res[1]], j] += 0.5 * v * res[0]
    return M


def _one_body_on(dets_from, dets_to, f):
    ix = {d: i for i, d in enumerate(dets_to)}
    X = np.zeros((len(dets_to), len(dets_from)))
    for j, D in enumerate(dets_from):
        for p, q in zip(*np.nonzero(f)):
            r = cfp_l._apply([('c', p), ('a', q)], D)
            if r and r[1] in ix:
                X[ix[r[1]], j] += f[p, q] * r[0]
    return X


@lru_cache(maxsize=None)
def _raisers():
    lp = np.zeros((NO, NO))
    sp = np.zeros((NO, NO))
    for p, (l, m, s) in enumerate(SO):
        for q, (l2, m2, s2) in enumerate(SO):
            if l == l2 and s == s2 and m == m2 + 1:
                lp[p, q] = sqrt(l * (l + 1) - m2 * (m2 + 1))
            if l == l2 and m == m2 and s == 1 and s2 == -1:
                sp[p, q] = 1.0
    return lp, sp


@lru_cache(maxsize=None)
def block(N, L, twoS):
    """(P basis (dets x m), Q basis (dets x nq), dets, {param: block matrix on [P states, Q states]})."""
    P, Q = sector(N, L, twoS)
    dets = P + Q
    # P states: dshell stretched states embedded behind 3s2 (same sign: the s pair stands first)
    St = dshell.stretched(N, L, twoS)
    dix = {d: i for i, d in enumerate(dshell.dets(N))}
    Pb = np.zeros((len(dets), St.shape[1]))
    for i, d in enumerate(P):
        Pb[i] = St[dix[tuple(x - 2 for x in d[2:])]]
    # Q states: highest weight in the Q dets
    if Q:
        lp, sp = _raisers()
        up = [sector(N, L + 1, twoS)[1], sector(N, L, twoS + 2)[1]]
        A = np.vstack([_one_body_on(Q, up[0], lp), _one_body_on(Q, up[1], sp)]) if (up[0] or up[1]) else \
            np.zeros((0, len(Q)))
        if A.shape[0]:
            u, sv, vt = np.linalg.svd(A)
            rank = int((sv > 1e-9).sum())
            Z = vt[rank:].T
        else:
            Z = np.eye(len(Q))
        Qb = np.zeros((len(dets), Z.shape[1]))
        Qb[len(P):] = Z
    else:
        Qb = np.zeros((len(dets), 0))
    B = np.hstack([Pb, Qb])
    ops = _ops_on(dets)
    return Pb.shape[1], Qb.shape[1], {k: B.T @ M @ B for k, M in ops.items()}


def hamiltonian(N, L, twoS, p):
    m, nq, ops = block(N, L, twoS)
    return m, sum(p[k] * ops[k] for k in PARAMS)
