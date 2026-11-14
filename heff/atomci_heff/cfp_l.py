"""Two-particle fractional-parentage weights of l^n terms for any l, with repeated terms.

Generalizes atomci_heff/cfp.py (p shell only) to the d shell, where a term
can occur more than once (d^3 2D twice). Determinant space of n electrons in
the 2(2l+1) spin-orbitals. For each term (L, 2S) the stretched states
(M_L = L, M_S = S) span a space of dimension = multiplicity; for a repeated
term the basis is fixed by seniority (eigenvectors of the pairing operator
A00^dagger A00, ordered by increasing seniority).

Pair-weight matrices, for an l^2 term T:

    W_T[i, j] = <Psi_i| sum_{phi in T multiplet} A_phi^dagger A_phi |Psi_j> / C(n, 2),

so that for any two-body operator V scalar in the l shell
    <l^n t_i|V|l^n t_j> = C(n, 2) sum_T W_T[i, j] <l^2 T|V|l^2 T>.

Independent checks: slater_condon() builds sum_{i<j} 1/r_ij in the
determinant space from Gaunt coefficients with radial integrals F^k, and
racah_d2 / racah_d3 give the textbook single-configuration energies.
"""
import itertools
from functools import lru_cache
from math import comb, sqrt

import numpy as np

from atomci_heff.cfp import ck


def spin_orbitals(l):
    return [(m, s) for m in range(-l, l + 1) for s in (1, -1)]


def dets(l, n):
    return list(itertools.combinations(range(2 * (2 * l + 1)), n))


def _apply(op_list, det):
    """Product of ('c'|'a', orbital) operators, applied right to left. Returns (sign, det) or None."""
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


def one_body(l, n, f):
    D = dets(l, n)
    ix = {d: i for i, d in enumerate(D)}
    M = np.zeros((len(D), len(D)))
    no = len(f)
    for j, d in enumerate(D):
        for p in range(no):
            for q in range(no):
                if f[p, q] == 0:
                    continue
                r = _apply([('c', p), ('a', q)], d)
                if r:
                    M[ix[r[1]], j] += r[0] * f[p, q]
    return M


@lru_cache(maxsize=None)
def ang_ops(l, n):
    SO = spin_orbitals(l)
    no = len(SO)
    lz = np.diag([float(m) for m, _ in SO])
    sz = np.diag([s / 2 for _, s in SO])
    lp = np.zeros((no, no))
    sp = np.zeros((no, no))
    for p, (m, s) in enumerate(SO):
        for q, (m2, s2) in enumerate(SO):
            if s == s2 and m == m2 + 1:
                lp[p, q] = sqrt(l * (l + 1) - m2 * (m2 + 1))
            if m == m2 and s == 1 and s2 == -1:
                sp[p, q] = 1.0
    Lz, Sz, Lp, Sp = (one_body(l, n, x) for x in (lz, sz, lp, sp))
    L2 = Lp.T @ Lp + Lz @ Lz + Lz
    S2 = Sp.T @ Sp + Sz @ Sz + Sz
    return L2, S2, Lz, Sz


def pair_annihilator(l, n, phi):
    """Matrix of A_phi = sum_{p<q} phi[(p, q)] a_q a_p from n to n-2 electrons."""
    Dn, Dm = dets(l, n), dets(l, n - 2)
    ixm = {d: i for i, d in enumerate(Dm)}
    A = np.zeros((len(Dm), len(Dn)))
    for (p, q), c in phi.items():
        if abs(c) < 1e-14:
            continue
        for j, d in enumerate(Dn):
            r = _apply([('a', q), ('a', p)], d)
            if r:
                A[ixm[r[1]], j] += c * r[0]
    return A


@lru_cache(maxsize=None)
def multiplets(l, n):
    """{(L, 2S): orthonormal basis (columns) of the full multiplet space} for l^n."""
    L2, S2, Lz, Sz = ang_ops(l, n)
    w, V = np.linalg.eigh(L2 + np.pi * S2)
    out = {}
    for L in range(n * l + 1):
        for twoS in range(n % 2, n + 1, 2):
            val = L * (L + 1) + np.pi * (twoS / 2) * (twoS / 2 + 1)
            k = np.abs(w - val) < 1e-8
            if k.any():
                out[(L, twoS)] = V[:, k]
    return out


def pairing(l, n):
    """A00^dagger A00 on l^n, A00 the normalized 1S pair of the shell."""
    SO = spin_orbitals(l)
    ix = {so: i for i, so in enumerate(SO)}
    # |l^2 1S> = sum_m (-1)^(l-m) / sqrt(2l+1) |m up, -m down>  (up to a global sign)
    phi = {}
    for m in range(-l, l + 1):
        p, q = ix[(m, 1)], ix[(-m, -1)]
        c = (-1) ** (l - m) / sqrt(2 * l + 1)
        if p < q:
            phi[(p, q)] = phi.get((p, q), 0.0) + c
        else:
            phi[(q, p)] = phi.get((q, p), 0.0) - c
    A = pair_annihilator(l, n, phi)
    return A.T @ A


@lru_cache(maxsize=None)
def stretched(l, n, L, twoS):
    """Columns: stretched states of the term (L, 2S); several for a repeated term, seniority-ordered.
    Returns (states, pairing eigenvalues)."""
    L2, S2, Lz, Sz = ang_ops(l, n)
    B = multiplets(l, n)[(L, twoS)]
    A = B.T @ (Lz + np.pi * Sz) @ B
    w, U = np.linalg.eigh(A)
    top = w > w.max() - 1e-8
    St = B @ U[:, top]
    if St.shape[1] > 1:
        Q = St.T @ pairing(l, n) @ St
        q, R = np.linalg.eigh(Q)
        St = St @ R[:, ::-1]          # largest pairing first = lowest seniority first
        q = q[::-1]
    else:
        q = np.array([float(St[:, 0] @ pairing(l, n) @ St[:, 0])])
    # fix the sign: largest component positive; ties (magnitude within 1e-10 of the largest, relative) go to the
    # lowest index, so the choice does not depend on rounding (np.argmax over near-equal components did, e.g. with
    # the BLAS thread count)
    for k in range(St.shape[1]):
        a = np.abs(St[:, k])
        i = int(np.flatnonzero(a >= a.max() * (1 - 1e-10))[0])
        if St[i, k] < 0:
            St[:, k] *= -1
    return St, q


def pair_projector(l, n, T):
    """sum_{phi in multiplet T of l^2} A_phi^dagger A_phi on l^n."""
    B2 = multiplets(l, 2)[T]
    D2 = dets(l, 2)
    P = 0.0
    for c in B2.T:
        A = pair_annihilator(l, n, {pq: c[i] for i, pq in enumerate(D2)})
        P = P + A.T @ A
    return P


def pair_weights(l, n, L, twoS):
    """{T: W_T (k x k)} for the term (L, 2S) of l^n with k = its multiplicity."""
    St, _ = stretched(l, n, L, twoS)
    return {T: St.T @ pair_projector(l, n, T) @ St / comb(n, 2) for T in multiplets(l, 2)}


def coulomb(l, n, Fk):
    """sum_{i<j} 1/r_ij on l^n determinants; Fk = {k: F^k} (unnormalized Slater integrals)."""
    SO = spin_orbitals(l)
    no = len(SO)
    g = np.zeros((no, no, no, no))
    for p, (m1, s1) in enumerate(SO):
        for q, (m2, s2) in enumerate(SO):
            for r, (m3, s3) in enumerate(SO):
                for s, (m4, s4) in enumerate(SO):
                    if s1 != s3 or s2 != s4 or m1 + m2 != m3 + m4:
                        continue
                    g[p, q, r, s] = sum(ck(k, l, m1, l, m3) * ck(k, l, m4, l, m2) * F for k, F in Fk.items())
    D = dets(l, n)
    ix = {d: i for i, d in enumerate(D)}
    G = np.zeros((len(D), len(D)))
    nz = np.argwhere(np.abs(g) > 0)
    for j, d in enumerate(D):
        for p, q, r, s in nz:
            res = _apply([('c', p), ('c', q), ('a', s), ('a', r)], d)
            if res:
                G[ix[res[1]], j] += 0.5 * g[p, q, r, s] * res[0]
    return G


def slater_condon(l, n, L, twoS, Fk):
    """Block <t_i|sum 1/r_ij|t_j> of the term (L, 2S) of l^n."""
    St, _ = stretched(l, n, L, twoS)
    return St.T @ coulomb(l, n, Fk) @ St


def racah_abc(F0, F2, F4):
    """Racah A, B, C from F^k (unnormalized): F2 = F^2/49, F4 = F^4/441."""
    f2, f4 = F2 / 49, F4 / 441
    return F0 - 49 * f4, f2 - 5 * f4, 35 * f4


D2_TERMS = {(3, 2): '3F', (1, 2): '3P', (4, 0): '1G', (2, 0): '1D', (0, 0): '1S'}
D3_TERMS = {(3, 3): '4F', (1, 3): '4P', (5, 1): '2H', (4, 1): '2G', (3, 1): '2F', (2, 1): '2D', (1, 1): '2P'}


def racah_d2(A, B, C):
    return {'3F': A - 8 * B, '3P': A + 7 * B, '1G': A + 4 * B + 2 * C, '1D': A - 3 * B + 2 * C,
            '1S': A + 14 * B + 7 * C}


def racah_d3(A, B, C):
    """Single-configuration d^3 term energies; 2D as its two eigenvalues (lower, upper)."""
    r = sqrt(193 * B * B + 8 * B * C + 4 * C * C)
    return {'4F': 3 * A - 15 * B, '4P': 3 * A, '2H': 3 * A - 6 * B + 3 * C, '2P': 3 * A - 6 * B + 3 * C,
            '2G': 3 * A - 11 * B + 3 * C, '2F': 3 * A + 9 * B + 3 * C,
            '2D': (3 * A + 5 * B + 5 * C - r, 3 * A + 5 * B + 5 * C + r)}
