"""Pade resummation of the H_eff power series H_eff(z) = sum_n z^n H^(n), evaluated at z = 1."""
import numpy as np


def pade_coeffs(coeffs, L, M):
    """Numerator p[0..L] and denominator q[0..M] (q0 = 1) of the [L/M] Pade.

    The denominator equations sum_k q_k c_{n-k} = 0, n = L+1..L+M, are solved
    by minimum-norm least squares, so rank-deficient cases (e.g. series with
    only even powers) still give a usable approximant.
    """
    c = np.asarray(coeffs[:L + M + 1], dtype=float)
    cc = lambda n: c[n] if n >= 0 else 0.0  # noqa: E731
    if M == 0:
        return c[:L + 1].copy(), np.array([1.0])
    A = np.array([[cc(n - k) for k in range(1, M + 1)] for n in range(L + 1, L + M + 1)])
    b = -np.array([cc(n) for n in range(L + 1, L + M + 1)])
    qt = np.linalg.lstsq(A, b, rcond=1e-13)[0]
    q = np.concatenate([[1.0], qt])
    p = np.array([sum(q[k] * cc(n - k) for k in range(0, min(n, M) + 1)) for n in range(L + 1)])
    return p, q


def pade_at(coeffs, L, M=None, z=1.0):
    """[L/M] Pade of a scalar series (coeffs[0..L+M]) evaluated at z."""
    M = L if M is None else M
    p, q = pade_coeffs(coeffs, L, M)
    return np.polyval(p[::-1], z) / np.polyval(q[::-1], z)


def series_mul(a, b):
    n = min(len(a), len(b))
    return np.array([sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(n)])


def series_sqrt(s):
    """Power series of sqrt(s), with s[0] > 0."""
    n = len(s)
    r = np.zeros(n)
    r[0] = np.sqrt(s[0])
    for k in range(1, n):
        r[k] = (s[k] - sum(r[i] * r[k - i] for i in range(1, k))) / (2 * r[0])
    return r


def eigen_series(terms):
    """Power series of the two eigenvalues of a 2x2 H_eff(z), as (lower, upper).

    "Lower" is the branch that is lower at z = 0. Needs the two diagonal
    elements of H^(0) to differ.
    """
    T = np.array(terms)
    a, b, c, d = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    half = (a - d) / 2
    s = series_mul(half, half) + series_mul(b, c)
    r = series_sqrt(s)
    mean = (a + d) / 2
    sgn = 1.0 if half[0] > 0 else -1.0
    # sqrt(s) is |half| at z=0; the eigenvalue that starts at min(a0, d0) is mean - r
    return mean - r, mean + r, sgn


def pade_table(terms, Ls=range(1, 11)):
    """Pade [L/L] at z=1 for every H_eff element and for the 2 eigenvalue series."""
    T = np.array(terms)
    m = T.shape[1]
    out = {}
    lo, hi, _ = eigen_series(terms) if m == 2 else (None, None, None)
    for L in Ls:
        if 2 * L + 1 > len(terms):
            break
        M = np.array([[pade_at(T[:, i, j], L) for j in range(m)] for i in range(m)])
        row = dict(matrix=M, eig_matrix=np.sort(np.linalg.eigvals(M)))
        if m == 2:
            row['eig_scalar'] = np.array([pade_at(lo, L), pade_at(hi, L)])
        out[L] = row
    return out
