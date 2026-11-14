#!/usr/bin/env python3
"""Phase 14a: average-of-configuration HF for V [Ar] 3d^N (Z = 23) in the QZP basis, outside ATOMCI.

ATOMCI's MCSCF optimizes one (L, S) symmetry per run, so neither a configuration average over all terms of 3d^N nor
an ensemble over N is available there. Both are one functional here:
  E = sum_a q_a h_aa + sum_{a<b} q_a q_b [F0(ab) - 1/2 sum_k w(l_a k l_b) G^k(ab)]
      + sum_a pair_a [F0(aa) - (2l+1)/(4l+1) sum_{k>0} w(l k l) F^k(aa)],      w(l k l') = (l k l'; 0 0 0)^2
with closed shells 1s2 2s2 2p6 3s2 3p6 (pair = q(q-1)/2) and the 3d shell with q = <N> and pair = <C(N,2)> over the
ensemble (N = 3: 3, 3; d4: 4, 6; d3-d4: 3.5, 4.5; d3-d5: 4, 19/3; d5: 5, 10). The core is re-optimized.
Fock operators: closed l-blocks F_l = h + sum_b q_b (J_b - 1/2 sum_k w K^k_b) (RHF form, self term included);
3d: F_d = h + sum_{b closed} q_b (J_b - 1/2 sum_k w K^k_b) + (2 pair / q)(J_3d - 5/9 sum_{k=2,4} w J^k_3d).
Radial integrals on the log grid of atomci_heff.integrals (Simpson), one-electron matrices analytic (gto.matrices).
"""
import os
import sys

import numpy as np
from scipy.integrate import cumulative_simpson
from sympy.physics.wigner import wigner_3j

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # heff/
from atomci_heff import config as _C  # noqa: E402
ROOT = _C.root()  # HEFF_WORK (the research layout: results/, runs/, inputs/)
_C.setup_path()
from atomci_heff import gto  # noqa: E402
from atomci_heff import integrals as I  # noqa: E402
from atomci_heff.reader import read_vectors  # noqa: E402
import gen_v  # noqa: E402

Z = 23
NBF = list(gen_v.nbf('vq'))                     # s p d f g h = 10 8 6 4 3 2
CLOSED = [('1s', 0, 0), ('2s', 0, 1), ('3s', 0, 2), ('2p', 1, 0), ('3p', 1, 1)]
_W = {}


def w3(l1, k, l2):
    if (l1, k, l2) not in _W:
        _W[(l1, k, l2)] = float(wigner_3j(l1, k, l2, 0, 0, 0)) ** 2
    return _W[(l1, k, l2)]


class Basis:
    def __init__(self):
        self.fns = gto.read_bas(os.path.join(_C.BASIS, gen_v.BASES['vq']))
        self.mat = {l: gto.matrices(self.fns, l, Z) for l in range(len(NBF))}
        self.h = {l: self.mat[l]['T'] + self.mat[l]['V'] for l in self.mat}
        self.S = {l: self.mat[l]['S'] for l in self.mat}
        self.x, self.r = I.radial_grid()
        n = len(self.x)
        assert n % 2 == 1
        dx = self.x[1] - self.x[0]
        w = np.full(n, 2.0)
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        self.w = w * dx / 3 * self.r                  # dr = r dx
        self.chi = I.radial_functions(self.fns, {l: np.eye(NBF[l]) for l in range(len(NBF))}, self.r)  # P = r R

    def Y(self, rho, k):
        """Y^k(r) of a product density rho(r) (rows allowed): int rho(s) r<^k / r>^(k+1) ds, times r (as I.slater)."""
        r, x = self.r, self.x
        inner = cumulative_simpson(rho * r ** k * r, x=x, initial=0.0, axis=-1)
        c = cumulative_simpson(rho * r ** (-k - 1) * r, x=x, initial=0.0, axis=-1)
        outer = c[..., -1:] - c
        return inner * r ** (-k - 1) + outer * r ** k

    def P(self, l, c):
        return c @ self.chi[l]

    def J(self, l, Pb, k):
        """Matrix of R^k(mu nu; b b) over basis functions of symmetry l."""
        Yb = self.Y(Pb * Pb, k)
        X = self.chi[l]
        return (X * (Yb * self.w)) @ X.T

    def K(self, l, Pb, k):
        """Matrix of R^k(mu b; b nu)."""
        X = self.chi[l]
        Yn = self.Y(Pb[None, :] * X, k)                # Y^k of (b nu) for each nu
        return (X * Pb * self.w) @ Yn.T

    def R(self, Pa, Pb, Pc, Pd, k):
        return float(np.sum(Pa * Pb * self.Y(Pc * Pd, k) * self.w))


def exch_ks(la, lb):
    return [k for k in range(abs(la - lb), la + lb + 1, 2)]


def scf(bs, q, pair, guess=None, tol=1e-10, itmax=300, damp=0.5):
    """Average-of-configuration HF; returns dict(C={l: C}, E, eps, ...). C columns: occupied first, then the
    remaining eigenvectors of the same Fock (closed l: F_l; d: F_d)."""
    C = guess or {l: gen_eig(bs.h[l], bs.S[l])[1] for l in range(3)}
    Fold = None
    Eold = 0.0
    for it in range(itmax):
        F, E, eps = fock(bs, C, q, pair)
        if Fold is not None:
            F = {l: (1 - damp) * F[l] + damp * Fold[l] for l in F}
        Fold = F
        Cn = {}
        for l in range(3):
            e, v = gen_eig(F[l], bs.S[l])
            Cn[l] = v
        C = Cn
        if abs(E - Eold) < tol and it > 5:
            break
        Eold = E
    F, E, eps = fock(bs, C, q, pair)
    # final canonical form (no damping)
    for l in range(3):
        e, v = gen_eig(F[l], bs.S[l])
        C[l] = v
        eps[l] = e
    F, E, _ = fock(bs, C, q, pair)
    return dict(C=C, E=E, eps=eps, F=F, it=it)


def gen_eig(F, S):
    s, U = np.linalg.eigh(S)
    X = U / np.sqrt(s)
    e, v = np.linalg.eigh(X.T @ F @ X)
    return e, X @ v


def shells(C, bs, q):
    sh = [(nm, l, bs.P(l, C[l][:, i]), 4 * l + 2) for nm, l, i in CLOSED]
    sh.append(('3d', 2, bs.P(2, C[2][:, 0]), q))
    return sh


def fock(bs, C, q, pair):
    sh = shells(C, bs, q)
    Fcl = {l: bs.h[l].copy() for l in range(3)}
    Fd = bs.h[2].copy()
    for nm, lb, Pb, qb in sh:
        for l in range(3):
            G = bs.J(l, Pb, 0) - 0.5 * sum(w3(l, k, lb) * bs.K(l, Pb, k) for k in exch_ks(l, lb))
            Fcl[l] += qb * G
            if nm != '3d':
                if l == 2:
                    Fd += qb * G
    P3 = sh[-1][2]
    if q > 0:
        Fd += (2 * pair / q) * (bs.J(2, P3, 0) - 5 / 9 * sum(w3(2, k, 2) * bs.J(2, P3, k) for k in (2, 4)))
    F = {0: Fcl[0], 1: Fcl[1], 2: Fd}
    E = energy(bs, C, q, pair)
    return F, E, {}


def energy(bs, C, q, pair):
    sh = shells(C, bs, q)
    E = 0.0
    for i, (na, la, Pa, qa) in enumerate(sh):
        c = C[la][:, 0 if na == '3d' else dict((n, j) for n, l, j in CLOSED)[na]]
        E += qa * c @ bs.h[la] @ c
        pr = pair if na == '3d' else qa * (qa - 1) / 2
        E += pr * (bs.R(Pa, Pa, Pa, Pa, 0) - (2 * la + 1) / (4 * la + 1) *
                   sum(w3(la, k, la) * bs.R(Pa, Pa, Pa, Pa, k) for k in range(2, 2 * la + 1, 2)))
        for nb, lb, Pb, qb in sh[i + 1:]:
            E += qa * qb * (bs.R(Pa, Pa, Pb, Pb, 0) -
                            0.5 * sum(w3(la, k, lb) * bs.R(Pa, Pb, Pa, Pb, k) for k in exch_ks(la, lb)))
    return E


def load_original():
    """Set 2 of runs/vq_orb/be.NO (V2+ 3d3 4F MCSCF core + 3d, IVO s virtuals), all l."""
    return read_vectors(os.path.join(ROOT, 'runs', 'vq_orb', 'be.NO'), NBF)[1]
