"""Direct (non-series) solvers of the Bloch equation for chi = Q Omega P.

    G(chi) = H_QP + H_QQ chi - chi H_PP - chi H_PQ chi = 0
    H_eff  = H_PP + H_PQ chi

Each solution corresponds to a choice of len(P) eigenstates whose P
components are linearly independent.

All solvers work on (H, P), with P a list of indices. For the series-form
iterations, H0 = diag(H) in whatever basis H is given; rotate first with
partition.transform to use another block-diagonal H0.
"""
import numpy as np

from . import heff


def blocks(H, P):
    P, Q = heff.split(H.shape[0], P)
    return (H[np.ix_(P, P)], H[np.ix_(P, Q)], H[np.ix_(Q, P)], H[np.ix_(Q, Q)]), P, Q


def residual(H, P, chi):
    (HPP, HPQ, HQP, HQQ), _, _ = blocks(H, P)
    return HQP + HQQ @ chi - chi @ HPP - chi @ HPQ @ chi


def denominators(H, P):
    P, Q = heff.split(H.shape[0], P)
    d = np.diag(H)
    return 1.0 / (d[P][None, :] - d[Q][:, None])


def chi_series(H, P, order):
    """Sum of the EN series chi^(1) + ... + chi^(order)."""
    _, chis = heff.en_series(H, P, order + 1)
    return sum(chis[1:order + 1])


def _result(H, P, chi, it, converged, history):
    return dict(chi=chi, iterations=it, converged=converged,
                residual=np.abs(residual(H, P, chi)).max() if np.all(np.isfinite(chi)) else np.inf,
                history=history)


def fixed_point(H, P, chi0, alpha=1.0, maxit=500, tol=1e-10, diis=0):
    """chi <- chi + alpha * R o G(chi), the series-form map (alpha=1) with optional damping and DIIS.

    diis > 0 keeps that many vectors (Pulay extrapolation on the
    preconditioned residual R o G).
    """
    R = denominators(H, P)
    chi = chi0.copy()
    hist, vecs, errs = [], [], []
    for it in range(1, maxit + 1):
        G = residual(H, P, chi)
        r = R * G
        nr = np.abs(G).max()
        hist.append(nr)
        if not np.isfinite(nr) or nr > 1e8:
            return _result(H, P, chi, it, False, hist)
        if nr < tol:
            return _result(H, P, chi, it, True, hist)
        new = chi + alpha * r
        if diis:
            vecs.append(new.copy())
            errs.append(r.copy())
            vecs, errs = vecs[-diis:], errs[-diis:]
            k = len(vecs)
            if k >= 2:
                B = -np.ones((k + 1, k + 1))
                B[k, k] = 0
                for i in range(k):
                    for j in range(k):
                        B[i, j] = np.vdot(errs[i], errs[j])
                rhs = np.zeros(k + 1)
                rhs[k] = -1
                try:
                    cvec = np.linalg.solve(B, rhs)[:k]
                    new = sum(c * v for c, v in zip(cvec, vecs))
                except np.linalg.LinAlgError:
                    pass
        chi = new
    return _result(H, P, chi, maxit, False, hist)


def newton(H, P, chi0, maxit=100, tol=1e-12):
    """Newton-Raphson on G(chi) = 0, with the Jacobian as a Kronecker sum (column-major vec)."""
    (HPP, HPQ, HQP, HQQ), Pi, Qi = blocks(H, P)
    m, nq = len(Pi), len(Qi)
    chi = chi0.copy()
    hist = []
    for it in range(1, maxit + 1):
        G = residual(H, P, chi)
        nr = np.abs(G).max()
        hist.append(nr)
        if not np.isfinite(nr) or nr > 1e8:
            return _result(H, P, chi, it, False, hist)
        if nr < tol:
            return _result(H, P, chi, it, True, hist)
        # dG = (HQQ - chi HPQ) dchi - dchi (HPP + HPQ chi)
        J = np.kron(np.eye(m), HQQ - chi @ HPQ) - np.kron((HPP + HPQ @ chi).T, np.eye(nq))
        try:
            dvec = np.linalg.solve(J, -G.reshape(-1, order='F'))
        except np.linalg.LinAlgError:
            return _result(H, P, chi, it, False, hist)
        chi = chi + dvec.reshape(nq, m, order='F')
    return _result(H, P, chi, maxit, False, hist)


def bw_iteration(H, P, chi0, maxit=500, tol=1e-10, alpha=1.0):
    """Partitioning (Brillouin-Wigner-like) iteration with the P energies updated each cycle.

    H_eff(chi) = H_PP + H_PQ chi = U diag(eps) U^-1. For each eps_k, the Q
    part of the eigenvector is y_k = (eps_k - H_QQ)^-1 H_QP u_k (exact
    partitioning), and chi_new = [y_1 .. y_m] U^-1. A fixed point is an
    exact solution of the Bloch equation.
    """
    (HPP, HPQ, HQP, HQQ), Pi, Qi = blocks(H, P)
    nq = len(Qi)
    chi = chi0.copy().astype(complex)
    hist = []
    for it in range(1, maxit + 1):
        G = residual(H, P, chi)
        nr = np.abs(G).max()
        hist.append(nr)
        if not np.isfinite(nr) or nr > 1e8:
            return _result(H, P, chi.real, it, False, hist)
        if nr < tol:
            return _result(H, P, chi.real if np.abs(chi.imag).max() < 1e-10 else chi, it, True, hist)
        eps, U = np.linalg.eig(HPP + HPQ @ chi)
        Y = np.column_stack([np.linalg.solve(e * np.eye(nq) - HQQ, HQP @ U[:, k])
                             for k, e in enumerate(eps)])
        new = Y @ np.linalg.inv(U)
        chi = (1 - alpha) * chi + alpha * new
    return _result(H, P, chi.real if np.abs(chi.imag).max() < 1e-10 else chi, maxit, False, hist)


def identify(H, P, chi):
    """Map a Bloch solution to exact eigenstates.

    Returns the eigenvalues of H_eff and, for each, (index of the exact
    state with the largest overlap, |overlap|) for the normalized vector
    Omega u_k.
    """
    (HPP, HPQ, _, _), Pi, Qi = blocks(H, P)
    eps, U = np.linalg.eig(HPP + HPQ @ chi)
    e, C = np.linalg.eigh(H)
    out = []
    for k in np.argsort(eps.real):
        v = np.zeros(H.shape[0], complex)
        v[Pi] = U[:, k]
        v[Qi] = chi @ U[:, k]
        v = v / np.linalg.norm(v)
        ov = np.abs(C.T @ v)
        j = int(np.argmax(ov))
        out.append((eps[k], j, ov[j], e[j]))
    return out
