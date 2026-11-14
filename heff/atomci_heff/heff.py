"""Exact effective Hamiltonians and Epstein-Nesbet QDPT series for a CI matrix.

Notation: P is a list of model-space indices and Q the rest. Eigenvectors
are the columns of C, normalized over the full space.
"""
import numpy as np


def split(n, P):
    P = list(P)
    Q = [i for i in range(n) if i not in P]
    return P, Q


def spectrum(H, P):
    """Eigenvalues, eigenvectors, total P weight and per-P-CSF weights for every state."""
    e, C = np.linalg.eigh(H)
    wP = (C[P, :] ** 2).sum(axis=0)
    return e, C, wP, C[P, :] ** 2


def select_states(wP, m):
    """The m states with the largest P weight (index order) plus an ambiguity measure.

    ambiguity = (weight of the best excluded state) / (smallest included weight).
    Values near 1 mean the choice is ambiguous.
    """
    order = np.argsort(-wP)
    chosen = np.sort(order[:m])
    amb = wP[order[m]] / wP[order[m - 1]] if len(wP) > m else 0.0
    return chosen, amb


def exact_heff(H, P, states):
    """Exact Bloch and des Cloizeaux effective Hamiltonians for the given eigenstates.

    Returns a dict with X (P components, columns = states), E, the Bloch
    H_eff = X E X^-1, the Hermitian des Cloizeaux form
    S^-1/2 X E X^T S^-1/2 with S = X X^T, cond(X), chi = Y X^-1 (the Q
    components of the wave operator), and the 2-norm and Frobenius norm of
    chi.
    """
    e, C = np.linalg.eigh(H)
    P, Q = split(H.shape[0], P)
    X = C[np.ix_(P, states)]
    Y = C[np.ix_(Q, states)]
    E = np.diag(e[states])
    Xi = np.linalg.inv(X)
    hb = X @ E @ Xi
    S = X @ X.T
    w, U = np.linalg.eigh(S)
    Sm = U @ np.diag(w ** -0.5) @ U.T
    hdc = Sm @ X @ E @ X.T @ Sm
    chi = Y @ Xi
    return dict(X=X, E=e[states], bloch=hb, hermitian=hdc, condX=np.linalg.cond(X),
                chi=chi, chi_norm2=np.linalg.norm(chi, 2), chi_normF=np.linalg.norm(chi))


def en_series(H, P, nmax):
    """Epstein-Nesbet Rayleigh-Schroedinger QDPT from the generalized Bloch equation.

    H0 = diag(H), V = H - H0. Solving
        [Omega, H0] P = Q V Omega P - chi P V Omega P,   Omega = P + chi,
    order by order gives
        chi^(1)_qp (E_p - E_q) = V_qp
        chi^(n)_qp (E_p - E_q) = [V_QQ chi^(n-1) - chi^(n-1) V_PP
                                  - sum_{k=1}^{n-2} chi^(k) V_PQ chi^(n-1-k)]_qp
    with H_eff^(0) = H0_PP, H_eff^(1) = V_PP and H_eff^(n) = V_PQ chi^(n-1)
    for n >= 2.
    Returns the list of H_eff^(n), n = 0..nmax, and the list of chi^(n).
    """
    P, Q = split(H.shape[0], P)
    d = np.diag(H).copy()
    V = H - np.diag(d)
    VPP, VPQ, VQP, VQQ = (V[np.ix_(P, P)], V[np.ix_(P, Q)], V[np.ix_(Q, P)], V[np.ix_(Q, Q)])
    R = 1.0 / (d[P][None, :] - d[Q][:, None])
    chis = [None, R * VQP]
    terms = [np.diag(d[P]), VPP]
    for n in range(2, nmax + 1):
        terms.append(VPQ @ chis[n - 1])
        if n == nmax:
            break
        rhs = VQQ @ chis[n - 1] - chis[n - 1] @ VPP
        for k in range(1, n - 1):
            rhs -= chis[k] @ VPQ @ chis[n - 1 - k]
        chis.append(R * rhs)
    return terms, chis


def partial_sums(terms):
    out, acc = [], np.zeros_like(terms[0])
    for t in terms:
        acc = acc + t
        out.append(acc.copy())
    return out


def sorted_eigs(M):
    w = np.linalg.eigvals(M)
    return w[np.lexsort((w.imag, w.real))]


def intruders(H, P, window=1.0):
    """Q CSFs whose diagonal energy lies within `window` of the span of the P diagonals.

    Returns (q, H_qq, distance to the nearest P diagonal, |V_qP| per p,
    max |V_qp/(H_pp - H_qq)|) sorted by distance; distance 0 means the CSF
    lies between the two P diagonals.
    """
    P, Q = split(H.shape[0], P)
    d = np.diag(H)
    lo, hi = d[P].min(), d[P].max()
    rows = []
    for q in Q:
        dist = 0.0 if lo <= d[q] <= hi else min(abs(d[q] - lo), abs(d[q] - hi))
        if dist <= window:
            v = H[q, P]
            ratio = np.max(np.abs(v / (d[P] - d[q])))
            rows.append((q, d[q], dist, v, ratio))
    return sorted(rows, key=lambda r: r[2])


def z_scan(H, P, zs, m=2):
    """Eigenvalues of H(z) = H0 + z V on a real grid, with the m most P-like states per z."""
    d = np.diag(H)
    V = H - np.diag(d)
    E, W, sel = [], [], []
    for z in zs:
        e, C = np.linalg.eigh(np.diag(d) + z * V)
        wP = (C[P, :] ** 2).sum(axis=0)
        E.append(e)
        W.append(wP)
        sel.append(np.sort(np.argsort(-wP)[:m]))
    return np.array(E), np.array(W), np.array(sel)


def avoided_crossings(zs, E, W, sel, gap_max=0.05):
    """Local minima of adjacent-level gaps that involve a P-selected state.

    Reports (z, i, i+1, gap, P weight of i and of i+1 at the minimum).
    Only minima with gap < gap_max where the two levels exchange P weight
    across the minimum are kept.
    """
    out = []
    nz, n = E.shape
    for i in range(n - 1):
        g = E[:, i + 1] - E[:, i]
        for k in range(1, nz - 1):
            if g[k] <= g[k - 1] and g[k] <= g[k + 1] and g[k] < gap_max:
                involved = (i in sel[k]) or (i + 1 in sel[k])
                if not involved:
                    continue
                a, b = max(k - 20, 0), min(k + 20, nz - 1)
                swap = (W[a, i] - W[a, i + 1]) * (W[b, i] - W[b, i + 1]) < 0
                if swap:
                    out.append((zs[k], i, i + 1, g[k], W[k, i], W[k, i + 1]))
    return out
