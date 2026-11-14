"""Iterative H_eff levels for one term block, without forming the dense H (Phase 15o).

Same quantities as phase15j.heff_levels (dense), from H kept as SymLower (strict lower triangle + diagonal):
  exact      des Cloizeaux H_eff of the m states with the largest P weight. The lowest k roots come from block
             Davidson on H - c (c = mean P diagonal, so that E - c is O(1) and no digits are lost to the total
             energy); k grows until the P weight left outside the computed roots, R = m - sum_k wP, is below the
             smallest selected wP, which proves that no uncomputed state could be selected (sum of wP over all
             states = m). below, gap and amb are then those of the dense selection (amb exact when the best
             excluded computed weight >= R, otherwise an upper bound max(best, R) / min selected).
  sylvester  X Hpp - Hqq X = V^T column by column in the Hpp eigenbasis, (Hqq - p_j) y_j = -V^T a_j by MINRES
             (Jacobi preconditioner |d_q - p_j|^-1, true residual checked), Hqq applied through H on [0; x]
             (P first, so no Q copy of H is made).
  closure    unchanged (needs only V and the configuration averages).
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, minres

from .sparse import SymLower, davidson

HART_EV = 27.211386          # as scripts/phase13d.py


def shifted(op, c):
    """op - c (shares L and L^T, no copy)."""
    s = SymLower.__new__(SymLower)
    s.L, s.LT, s.d, s.n = op.L, op.LT, op.d - c, op.n
    return s


def p_states(op, P, m, extra=8, tol=1e-10, dense_below=40):
    """Lowest roots of op until the m largest-P-weight states are provably among them.
    Returns E (shifted by c), C, wP, selected indices, R, c."""
    P = np.asarray(P)
    c = float(op.d[P].mean())
    sh = shifted(op, c)
    if op.n <= dense_below:
        e, C = np.linalg.eigh(sh.dense())
        wP = (C[P] ** 2).sum(axis=0)
        st = np.sort(np.argsort(-wP)[:m])
        return e, C, wP, st, 0.0, c
    X0 = None
    while True:
        k = min(m + extra, op.n - 1)
        if X0 is None or X0.shape[1] < k:
            # P unit vectors, then the lowest Q diagonals
            q = [i for i in np.argsort(sh.d) if i not in set(P.tolist())][:k - m]
            X0n = np.zeros((op.n, k))
            X0n[np.r_[P, q], np.arange(k)] = 1.0
            if X0 is not None:
                X0n[:, :X0.shape[1]] = X0
            X0 = X0n
        E, C, rn, it, nmv = davidson(sh, k, X0=X0, tol=tol, maxit=2000, max_sub=max(6 * k, 60))
        assert rn.max() < tol, ('Davidson not converged', rn.max(), it)
        wP = (C[P] ** 2).sum(axis=0)
        st = np.sort(np.argsort(-wP)[:m])
        R = m - wP.sum()
        if R < wP[st].min() or k == op.n - 1:
            return E, C, wP, st, max(R, 0.0), c
        X0, extra = C, 2 * extra


def heff_block(b, op, tol=1e-10):
    """Iterative counterpart of phase13f.heff_block: (U^T H_dC U, info)."""
    P, m, U = b['P'], b['m'], b['U']
    if not b['Q']:
        Hpp = op.sub(np.asarray(P)).dense()
        return U.T @ Hpp @ U, dict(wP=1.0, amb=0.0, below=0, gap=np.inf)
    e, C, wP, st, R, c = p_states(op, P, m, tol=tol)
    other = [k for k in range(len(e)) if k not in st]
    X, E = C[np.ix_(P, st)], e[st]
    w, Us = np.linalg.eigh(X @ X.T)
    Sm = Us @ np.diag(w ** -0.5) @ Us.T
    hdc = Sm @ X @ np.diag(E) @ X.T @ Sm + c * np.eye(m)
    best = max([wP[k] for k in other], default=0.0)
    return U.T @ hdc @ U, dict(wP=float(wP[st].min()), amb=float(max(best, R) / wP[st].min()),
                               amb_exact=bool(best >= R), R=float(R), nroots=len(e),
                               below=int(sum(1 for k in other if e[k] < e[st].max())),
                               gap=float((e[other[0]] - e[st].max()) * HART_EV) if other else np.inf)


def sylvester(op, m, Hpp, V, rtol=1e-13):
    """Hpp + (V X + X^T V^T)/2 with X Hpp - Hqq X = V^T (P = first m rows)."""
    n = op.n
    nq = n - m
    dq = op.d[m:]
    p, A = np.linalg.eigh(Hpp)

    def hqq(x):
        z = np.zeros(n)
        z[m:] = x
        return op.matvec(z[:, None])[m:, 0]
    Y = np.zeros((nq, m))
    info = []
    for j in range(m):
        rhs = -(V.T @ A[:, j])
        Aop = LinearOperator((nq, nq), matvec=lambda x, pj=p[j]: hqq(x) - pj * x, dtype=float)
        pre = 1.0 / np.maximum(np.abs(dq - p[j]), 1e-6)
        M = LinearOperator((nq, nq), matvec=lambda x, pr=pre: pr * x, dtype=float)
        y, ok = minres(Aop, rhs, M=M, rtol=rtol, maxiter=20000)
        res = np.linalg.norm(Aop.matvec(y) - rhs) / np.linalg.norm(rhs)
        assert ok == 0 and res < 1e-10, ('MINRES', ok, res)
        Y[:, j] = y
        info.append(float(res))
    X = Y @ A.T
    return Hpp + 0.5 * (V @ X + X.T @ V.T), info


def heff_levels(b, tol=1e-10):
    """Iterative counterpart of phase15j.heff_levels: b['op'] is the SymLower H of the block (P first)."""
    op, P, Q, U = b['op'], b['P'], b['Q'], b['U']
    m = len(P)
    assert list(P) == list(range(m))
    Hpp = op.sub(np.asarray(P)).dense()
    V = op.block(np.asarray(P), np.asarray(Q)).toarray()
    ex, inf = heff_block(b, op, tol=tol)
    syl, rs = sylvester(op, m, Hpp, V)
    inf['minres_res'] = max(rs)
    clo = Hpp.copy()
    for c, dl in b['delta'].items():
        idx = [k for k, i in enumerate(Q) if b['lab'][i] == c]
        clo -= V[:, idx] @ V[:, idx].T / dl
    return {'exact': ex, 'sylvester': U.T @ syl @ U, 'closure': U.T @ clo @ U, 'php': U.T @ Hpp @ U}, inf
