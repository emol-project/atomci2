"""Sparse symmetric H from a lower-triangle export, and a block Davidson solver.

The full-CI exports of Phase 6 are too large to densify (82,429 CSFs would
need 54 GB). H is kept as its strict lower triangle L (CSR) plus the
diagonal d, and applied as H x = L x + L^T x + d x, so the symmetric matrix
is never formed.
"""
import numpy as np
import scipy.sparse as sp


class SymLower:
    def __init__(self, L, d):
        self.L = L.tocsr()
        self.LT = self.L.T.tocsr()
        self.d = np.asarray(d, dtype=float)
        self.n = len(self.d)

    @classmethod
    def from_npz(cls, npz):
        """npz with H (dense) or H_rows/H_cols/H_vals/N (lower triangle incl. diagonal)."""
        if 'H' in npz.files:
            H = npz['H']
            return cls(sp.csr_matrix(np.tril(H, -1)), np.diag(H))
        n = int(npz['N'])
        r, c, v = npz['H_rows'], npz['H_cols'], npz['H_vals']
        dg = r == c
        d = np.zeros(n)
        d[r[dg]] = v[dg]
        off = ~dg
        L = sp.csr_matrix((v[off], (r[off], c[off])), shape=(n, n))
        return cls(L, d)

    def matvec(self, X):
        return self.L @ X + self.LT @ X + (self.d * X.T).T

    def sub(self, idx):
        """Principal submatrix on idx (sorted ascending, so strict lower stays strict lower)."""
        idx = np.asarray(idx)
        assert np.all(np.diff(idx) > 0)
        return SymLower(self.L[idx][:, idx], self.d[idx])

    def block(self, rows, cols):
        """Dense or sparse H[rows, cols] for disjoint index sets (CSR)."""
        return (self.L[rows][:, cols] + self.LT[rows][:, cols]).tocsr()

    def dense(self):
        A = self.L.toarray()
        return A + A.T + np.diag(self.d)


def davidson(op, k, X0=None, tol=1e-9, maxit=200, max_sub=None, log=None, time_limit=None):
    """Lowest k eigenpairs of op (SymLower-like: .matvec, .d, .n) by block Davidson.

    Diagonal (Davidson) preconditioner; restarts at max_sub vectors keeping
    2k Ritz vectors. Converged when every residual norm < tol (|E| errors are
    ~ residual^2). Stops early after time_limit seconds (unconverged; the
    caller checks the residuals). Returns E, C, residual norms, iterations,
    matvec count.
    """
    import time
    t0 = time.time()
    n = op.n
    max_sub = max_sub or max(8 * k, 40)
    if X0 is None:
        X0 = np.zeros((n, k))
        X0[np.argsort(op.d)[:k], np.arange(k)] = 1.0
    V, _ = np.linalg.qr(X0)
    AV = op.matvec(V)
    nmv = V.shape[1]
    for it in range(1, maxit + 1):
        T = V.T @ AV
        w, S = np.linalg.eigh((T + T.T) / 2)
        E, Y = w[:k], S[:, :k]
        X = V @ Y
        R = AV @ Y - X * E
        rn = np.linalg.norm(R, axis=0)
        if log:
            log(it, E, rn)
        if rn.max() < tol or (time_limit and time.time() - t0 > time_limit):
            return E, X, rn, it, nmv
        act = np.nonzero(rn >= tol)[0]
        den = op.d[:, None] - E[act][None, :]
        den[np.abs(den) < 1e-8] = 1e-8
        Tn = R[:, act] / den
        if V.shape[1] + len(act) > max_sub:
            keep = S[:, :2 * k]
            V = V @ keep        # orthonormal: V and the columns of S are
            AV = AV @ keep
        for _ in range(2):
            Tn -= V @ (V.T @ Tn)
        Tn, _ = np.linalg.qr(Tn)
        nrm = np.linalg.norm(Tn - V @ (V.T @ Tn), axis=0)
        Tn = Tn[:, nrm > 1e-6]
        if Tn.shape[1] == 0:
            return E, X, rn, it, nmv
        V = np.hstack([V, Tn])
        AV = np.hstack([AV, op.matvec(Tn)])
        nmv += Tn.shape[1]
    return E, X, rn, maxit, nmv
