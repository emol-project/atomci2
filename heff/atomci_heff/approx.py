"""Approximate effective Hamiltonians against an exact (full-CI) benchmark.

Conventions: P is a list of model-space indices. A "state set" is a set of
m approximate eigenvectors given in some subspace S of the full space
(columns of V, len(S) rows), with energies E. Its P components define
X = V[P] (columns normalized over S), from which

    Bloch           H_eff = X diag(E) X^-1
    des Cloizeaux   H_dC  = S^-1/2 X diag(E) X^T S^-1/2,   S = X X^T

which is exact when V and E are exact eigenpairs.
"""
import collections

import numpy as np

from . import heff, solvers

# ---------------------------------------------------------------- labels


def parse_label(label):
    """'11p1 1s2 2p1' -> {'11p': 1, '1s': 2, '2p': 1}."""
    occ = {}
    for tok in label.split():
        i = len(tok) - 1
        while tok[i].isdigit():
            i -= 1
        occ[tok[:i + 1]] = int(tok[i + 1:])
    return occ


def classify(labels, refs, core='1s', ncore=2):
    """Per CSF: excitation level (min over refs), number of core holes, configuration key."""
    nel = sum(refs[0].values())
    lev, holes, conf = [], [], []
    for lab in labels:
        occ = parse_label(lab)
        lev.append(min(nel - sum(min(occ.get(o, 0), r[o]) for o in r) for r in refs))
        holes.append(ncore - occ.get(core, 0))
        conf.append(' '.join(sorted(lab.split())))
    return np.array(lev), np.array(holes), conf


def groups_from_keys(idx, keys):
    """Partition the index list idx into groups with equal keys[i]."""
    g = collections.OrderedDict()
    for i in idx:
        g.setdefault(keys[i], []).append(i)
    return list(g.values())


# ---------------------------------------------------------------- H_eff from states


def heff_from_states(V, E, Ploc):
    """Bloch and des Cloizeaux H_eff from m approximate states (columns of V, full-space normalized)."""
    Vn = V / np.linalg.norm(V, axis=0)
    X = Vn[Ploc, :]
    E = np.asarray(E, dtype=float)
    bloch = X @ np.diag(E) @ np.linalg.inv(X)
    S = X @ X.T
    w, U = np.linalg.eigh(S)
    Sm = U @ np.diag(w ** -0.5) @ U.T
    dc = Sm @ X @ np.diag(E) @ X.T @ Sm
    return dict(E=np.sort(E), bloch=bloch, dc=dc, condX=np.linalg.cond(X))


def heff_from_chi(HPP, HPQ, chi, W=None):
    """States implied by a wave operator chi: H_eff(chi) = W (default H_PP + H_PQ chi)."""
    W = HPP + HPQ @ chi if W is None else W
    eps, U = np.linalg.eig(W)
    eps, U = eps.real, U.real
    V = np.vstack([U, chi @ U])
    return heff_from_states(V, eps, list(range(U.shape[0])))


def metrics(res, bench):
    """Errors in mEh: the two eigenvalues, the excitation energy, and the max |H_dC - H_dC(exact)| element."""
    e = np.sort(res['E'])
    eb = np.sort(bench['E'])
    de = (e - eb) * 1e3
    return dict(E=e, dE1=de[0], dE2=de[1], dExc=de[1] - de[0],
                dDC=np.abs(res['dc'] - bench['dc']).max() * 1e3)


def exact(H, P, states):
    e, C = np.linalg.eigh(H)
    return heff_from_states(C[:, states], e[states], P)


def target_states(H, P, m=2):
    e, C, wP, wPi = heff.spectrum(H, P)
    st, amb = heff.select_states(wP, m)
    return st, amb, wP, wPi, e, C


# ---------------------------------------------------------------- A: g-scaled Bloch equation


def g_residual(HPP, HPQ, HQP, HQQ, chi, g):
    W = HPP + g * HPQ @ chi
    return HQP + HQQ @ chi - chi @ W


def solve_g(HPP, HPQ, HQP, HQQ, g, chi0, maxit=100, tol=1e-12):
    """Newton for  H_QP + H_QQ chi - chi (H_PP + g H_PQ chi) = 0  (g=1: Bloch; g=0: linear)."""
    m, nq = HPP.shape[0], HQQ.shape[0]
    chi = chi0.copy()
    for it in range(1, maxit + 1):
        G = g_residual(HPP, HPQ, HQP, HQQ, chi, g)
        nr = np.abs(G).max()
        if nr < tol:
            return chi, it, True
        if not np.isfinite(nr) or nr > 1e6:
            return chi, it, False
        W = HPP + g * HPQ @ chi
        J = np.kron(np.eye(m), HQQ - g * chi @ HPQ) - np.kron(W.T, np.eye(nq))
        d = np.linalg.solve(J, -G.reshape(-1, order='F'))
        chi = chi + d.reshape(nq, m, order='F')
    return chi, maxit, False


def sylvester_linear(HPP, HQP, HQQ):
    """g = 0: H_QQ chi - chi H_PP = -H_QP, solved directly. Returns chi, cond(operator), separation."""
    m, nq = HPP.shape[0], HQQ.shape[0]
    L = np.kron(np.eye(m), HQQ) - np.kron(HPP.T, np.eye(nq))
    chi = np.linalg.solve(L, -HQP.reshape(-1, order='F')).reshape(nq, m, order='F')
    sv = np.linalg.svd(L, compute_uv=False)
    sep = np.min(np.abs(np.linalg.eigvalsh(HQQ)[:, None] - np.linalg.eigvalsh(HPP)[None, :]))
    return chi, sv[0] / sv[-1], sv[-1], sep


def davidson_q(e_ci, c2_ref, e_ref):
    """Multireference Davidson correction: E + (1 - c_ref^2)(E - E_ref)."""
    return e_ci + (1.0 - c2_ref) * (e_ci - e_ref)


# ---------------------------------------------------------------- B: blocked QHQ


def blocked(H, Q, groups):
    """Copy of H with the Q-Q block kept only inside the given groups (lists of Q indices)."""
    Ht = H.copy()
    Qa = np.array(Q)
    Ht[np.ix_(Qa, Qa)] = 0.0
    for g in groups:
        ga = np.array(g)
        Ht[np.ix_(ga, ga)] = H[np.ix_(ga, ga)]
    return Ht


def bw_solve(H, P, maxit=300, tol=1e-10):
    """BW-like partitioning iteration from chi = 0 on H (Phase 3 solver), plus the pair it lands on."""
    n = H.shape[0]
    res = solvers.bw_iteration(H, P, np.zeros((n - len(P), len(P))), maxit=maxit, tol=tol)
    ids = solvers.identify(H, P, res['chi']) if res['converged'] else None
    return res, ids


# ---------------------------------------------------------------- C: hierarchical M/R


def hierarchical(H, M, R, target_vecs_M, mode='diag', groups=None, self_consistent=True,
                 maxit=200, tol=1e-12):
    """Energy-dependent downfolding of R onto M for each target state.

    H_M(E) = H_MM + H_MR (E - D_RR)^-1 H_RM with D = diag(H_RR) ('diag'),
    H_RR blocked by `groups` ('block'), or the full H_RR ('full').
    target_vecs_M: M-space start vectors (columns), one per target, used to
    track the state by overlap. Returns per target (E, full-space vector on
    M+R, iterations).
    """
    HMM = H[np.ix_(M, M)]
    HMR = H[np.ix_(M, R)]
    HRR = H[np.ix_(R, R)]
    if mode == 'diag':
        D = np.diag(np.diag(HRR))
    elif mode == 'block':
        pos = {r: k for k, r in enumerate(R)}
        D = np.zeros_like(HRR)
        for g in groups:
            ix = [pos[r] for r in g]
            D[np.ix_(ix, ix)] = HRR[np.ix_(ix, ix)]
    elif mode == 'full':
        D = HRR
    else:
        raise ValueError(mode)
    nR = len(R)
    out = []
    for k in range(target_vecs_M.shape[1]):
        cprev = target_vecs_M[:, k]
        w0, U0 = np.linalg.eigh(HMM)
        E = w0[np.argmax(np.abs(U0.T @ cprev))]
        it = 0
        for it in range(1, maxit + 1):
            G = np.linalg.solve(E * np.eye(nR) - D, HMR.T) if nR else np.zeros((0, len(M)))
            HM = HMM + HMR @ G
            w, U = np.linalg.eigh((HM + HM.T) / 2)
            j = int(np.argmax(np.abs(U.T @ cprev)))
            Enew, c = w[j], U[:, j]
            cprev = c
            if not self_consistent:
                E = Enew
                break
            if abs(Enew - E) < tol:
                E = Enew
                break
            E = Enew
        cR = np.linalg.solve(E * np.eye(nR) - D, HMR.T @ c) if nR else np.zeros(0)
        out.append((E, np.concatenate([c, cR]), it))
    return out
