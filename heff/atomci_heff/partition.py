"""Block-diagonal H0 partitionings for QDPT.

Every partitioning here is block-diagonal in P/Q. We rotate P and/or Q
into a basis in which H0 is diagonal. In that basis H0 = diag(Ht), and the
Epstein-Nesbet recursion of heff.en_series applies unchanged.

    'EN'   (i)   H0 = diag(H)                     no rotation
    'P'    (ii)  H0 = PHP + diag(QHQ)             P rotated to the eigenbasis of PHP
    'Q'    (iii) H0 = diag(PHP) + QHQ             Q rotated to the eigenbasis of QHQ
    'PQ'   (iv)  H0 = PHP + QHQ, V = PHQ + QHP    both rotated
"""
import numpy as np

from . import heff

MODES = ('EN', 'P', 'Q', 'PQ')


def transform(H, P, mode):
    """Return (Ht, U, P') with Ht = U^T H U and H0 = diag(Ht).

    P' = [0..m-1]: the basis is reordered so that P comes first.
    """
    n = H.shape[0]
    P, Q = heff.split(n, P)
    order = P + Q
    m = len(P)
    U = np.zeros((n, n))
    UP = np.eye(m)
    UQ = np.eye(n - m)
    if mode in ('P', 'PQ'):
        _, UP = np.linalg.eigh(H[np.ix_(P, P)])
    if mode in ('Q', 'PQ'):
        _, UQ = np.linalg.eigh(H[np.ix_(Q, Q)])
    if mode not in MODES:
        raise ValueError(mode)
    U[np.ix_(P, range(m))] = UP
    U[np.ix_(Q, range(m, n))] = UQ
    Ht = U.T @ H @ U
    return Ht, U, list(range(m))


def series(H, P, mode, nmax):
    """H_eff^(n), n = 0..nmax (in the rotated P basis), plus the transform."""
    Ht, U, Pt = transform(H, P, mode)
    terms, chis = heff.en_series(Ht, Pt, nmax)
    return terms, chis, Ht, U, Pt


def radius_from_terms(terms, n0=15):
    """Convergence radius estimate from the geometric growth of ||H_eff^(n)||, n >= n0."""
    t = np.array([np.linalg.norm(x) for x in terms])
    n = np.arange(len(t))
    # drop roundoff-level terms (e.g. the odd orders of the even series of mode 'PQ')
    floor = 1e-8 * np.max(t[n >= n0]) if np.any(n >= n0) else 0.0
    sel = (n >= n0) & (t > floor)
    if sel.sum() < 3:
        return np.nan
    slope = np.polyfit(n[sel], np.log(t[sel]), 1)[0]
    return float(np.exp(-slope))


def eig_errors(terms, targets):
    """|eig(partial sum) - target| per order, matching each target to its nearest eigenvalue."""
    out = []
    for S in heff.partial_sums(terms):
        ev = np.linalg.eigvals(S)
        out.append([np.min(np.abs(ev - t)) for t in targets])
    return np.array(out)
