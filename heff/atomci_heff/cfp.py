"""Two-particle fractional-parentage weights of p^n terms, derived numerically.

Determinant space of n electrons in the six p spin-orbitals (m = -1, 0, 1;
spin up/down). LS terms are the joint eigenspaces of L^2 and S^2. For a
p^n term state Psi and a p^2 term T,

    w(T) = <Psi| sum_{i<j} P_T(ij) |Psi> / C(n, 2),
    sum_{i<j} P_T(ij) = sum_{phi in T multiplet} A_phi^dagger A_phi,
    A_phi^dagger = sum_{p<q} phi_pq a_p^dagger a_q^dagger,

so that for any two-body operator that is scalar in the p shell,
<p^n term|V|p^n term> = C(n, 2) sum_T w(term, T) <p^2 T|V|p^2 T>.
The weights are the squared two-particle coefficients of fractional
parentage summed over the spectator.

An independent check: slater_condon() evaluates <Psi| sum_{i<j} 1/r_ij |Psi>
from Gaunt coefficients (3j symbols, Racah formula) with radial integrals
F0 and F2 (Condon-Shortley F2 = F^2/25).
"""
import itertools
from math import factorial, sqrt

import numpy as np

SO = [(m, s) for m in (-1, 0, 1) for s in (1, -1)]       # spin-orbitals (m, 2*ms)


def dets(n):
    return [d for d in itertools.combinations(range(6), n)]


def _apply(op_list, det):
    """Apply a product of (kind, orbital) operators right to left to a sorted determinant tuple.
    kind 'c' = creation, 'a' = annihilation. Returns (sign, det) or None."""
    occ = list(det)
    sign = 1
    for kind, p in reversed(op_list):
        if kind == 'a':
            if p not in occ:
                return None
            k = occ.index(p)
            sign *= (-1) ** k
            occ.pop(k)
        else:
            if p in occ:
                return None
            k = sum(1 for q in occ if q < p)
            sign *= (-1) ** k
            occ.insert(k, p)
    return sign, tuple(occ)


def one_body(n, f):
    """Matrix of sum_pq f[p,q] a_p^dag a_q on the n-electron determinant space."""
    D = dets(n)
    ix = {d: i for i, d in enumerate(D)}
    M = np.zeros((len(D), len(D)))
    for j, d in enumerate(D):
        for p in range(6):
            for q in range(6):
                if f[p, q] == 0:
                    continue
                r = _apply([('c', p), ('a', q)], d)
                if r:
                    M[ix[r[1]], j] += r[0] * f[p, q]
    return M


def ang_ops(n):
    lz = np.diag([SO[p][0] for p in range(6)]).astype(float)
    sz = np.diag([SO[p][1] / 2 for p in range(6)])
    lp = np.zeros((6, 6))
    sp = np.zeros((6, 6))
    for p, (m, s) in enumerate(SO):
        for q, (m2, s2) in enumerate(SO):
            if s == s2 and m == m2 + 1:
                lp[p, q] = sqrt(2 - m2 * (m2 + 1))          # l = 1
            if m == m2 and s == 1 and s2 == -1:
                sp[p, q] = 1.0
    Lz, Sz, Lp, Sp = (one_body(n, x) for x in (lz, sz, lp, sp))
    L2 = Lp.T @ Lp + Lz @ Lz + Lz
    S2 = Sp.T @ Sp + Sz @ Sz + Sz
    return L2, S2, Lz, Sz


def terms(n):
    """{(L, 2S): orthonormal basis (columns) of the multiplet} for p^n."""
    L2, S2, Lz, Sz = ang_ops(n)
    w, V = np.linalg.eigh(L2 + np.pi * S2)
    out = {}
    for L in range(4):
        for twoS in range(0, 4):
            val = L * (L + 1) + np.pi * (twoS / 2) * (twoS / 2 + 1)
            k = np.abs(w - val) < 1e-8
            if k.any():
                out[(L, twoS)] = V[:, k]
    return out


def stretched(n, L, twoS):
    """The M_L = L, M_S = S state of the term (unique for p^n terms)."""
    L2, S2, Lz, Sz = ang_ops(n)
    B = terms(n)[(L, twoS)]
    A = B.T @ (Lz + np.pi * Sz) @ B
    w, U = np.linalg.eigh(A)
    return B @ U[:, -1]


def pair_weights(n, L, twoS):
    """w(T) over p^2 terms T = (L, 2S) for the p^n term (L, 2S)."""
    psi = stretched(n, L, twoS)
    Dn, D2, Dm = dets(n), dets(2), dets(n - 2)
    ixm = {d: i for i, d in enumerate(Dm)}
    out = {}
    for T, B in terms(2).items():
        tot = 0.0
        for c in B.T:                                   # each phi of the multiplet
            v = np.zeros(len(Dm))
            for (p, q), cpq in zip(D2, c):
                if abs(cpq) < 1e-14:
                    continue
                for j, d in enumerate(Dn):
                    if abs(psi[j]) < 1e-14:
                        continue
                    r = _apply([('a', q), ('a', p)], d)   # A_phi = sum c_pq a_q a_p
                    if r:
                        v[ixm[r[1]]] += cpq * r[0] * psi[j]
            tot += v @ v
        out[T] = tot / (n * (n - 1) / 2)
    return out


def threej(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol (integer arguments), Racah formula."""
    if m1 + m2 + m3 != 0 or not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    f = factorial
    tri = f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
    pre = sqrt(tri * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
    s = 0.0
    for t in range(0, 100):
        a = [t, j3 - j2 + t + m1, j3 - j1 + t - m2, j1 + j2 - j3 - t, j1 - t - m1, j2 - t + m2]
        if min(a) < 0:
            continue
        s += (-1) ** t / np.prod([f(x) for x in a])
    return (-1) ** (j1 - j2 - m3) * pre * s


def ck(k, l1, m1, l2, m2):
    """Gaunt coefficient c^k(l1 m1, l2 m2) (Condon-Shortley)."""
    return (-1) ** m1 * sqrt((2 * l1 + 1) * (2 * l2 + 1)) * threej(l1, k, l2, 0, 0, 0) * threej(l1, k, l2, -m1, m1 - m2, m2)


def slater_condon(n, L, twoS, F0, F2):
    """<p^n LS| sum_{i<j} 1/r_ij |p^n LS> with radial integrals F0 and F2 (= F^2/25)."""
    Rk = {0: F0, 2: 25.0 * F2}
    g = np.zeros((6, 6, 6, 6))           # <pq|g|rs>, electron 1: p->r, electron 2: q->s
    for p, (m1, s1) in enumerate(SO):
        for q, (m2, s2) in enumerate(SO):
            for r, (m3, s3) in enumerate(SO):
                for s, (m4, s4) in enumerate(SO):
                    if s1 != s3 or s2 != s4 or m1 + m2 != m3 + m4:
                        continue
                    g[p, q, r, s] = sum(ck(k, 1, m1, 1, m3) * ck(k, 1, m4, 1, m2) * Rk[k] for k in (0, 2))
    D = dets(n)
    ix = {d: i for i, d in enumerate(D)}
    G = np.zeros((len(D), len(D)))
    for j, d in enumerate(D):
        for p, q, r, s in itertools.product(range(6), repeat=4):
            if g[p, q, r, s] == 0:
                continue
            res = _apply([('c', p), ('c', q), ('a', s), ('a', r)], d)
            if res:
                G[ix[res[1]], j] += 0.5 * g[p, q, r, s] * res[0]
    psi = stretched(n, L, twoS)
    return float(psi @ G @ psi)
