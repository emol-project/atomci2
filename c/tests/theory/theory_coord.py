# theory_coord.py -- coordinate-space reference for theory_check_bp2e, _props,
# _props2e and _multipole.
#
# The technical note (Secs. 4.2 and 5) writes each Breit-Pauli / property operator as
#   sum over ordered shells of Theta x (coupled second-quantized operator).
# Here the same operator is built a second way, without any of that tensor
# algebra: m-resolved orbitals phi = exp(-zeta r^2) r^l Y_lm on a product grid
# (Gauss-Legendre in cos(theta), uniform in phi, Simpson in r), derivatives as
# Cartesian finite differences of the solid harmonics (exact for polynomials of
# degree <= 4), electron-electron kernels through Coulomb potentials of
# electron-2 functions (Laplace expansion with numerical projection onto Y_kq,
# cumulative radial integrals). The resulting spin-orbital integrals
# <pq|o|rs> give the operator sum_{i!=j} o(i, j) = sum <pq|o|rs> a+_p a+_q a_s a_r
# in pyref's determinant space, to be compared with the note's formula.
# Radial functions are not normalized (the identities do not need it).
import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import sph_harm_y

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "python"))
sys.path.insert(0, _here)
from pyref import Space, cg2, threej2, two_body_normal, one_body_tensor  # noqa: E402,F401


# ---------------------------------------------------------------- grid
class Grid:
    def __init__(self, nr=600, rmax=9.0, nth=12, nph=24):
        self.h = rmax / nr
        self.r0 = np.arange(nr + 1) * self.h                  # with r = 0 (cumulative integrals)
        self.r = self.r0[1:]                                  # evaluation points
        w = np.ones(nr + 1)
        w[1:-1:2], w[2:-1:2] = 4.0, 2.0
        self.wr = (w * self.h / 3.0)[1:]                      # Simpson; the r = 0 term vanishes
        x, wx = np.polynomial.legendre.leggauss(nth)
        th = np.arccos(x)
        ph = 2 * np.pi * np.arange(nph) / nph
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        self.th, self.ph = TH.ravel(), PH.ravel()
        self.wang = (np.repeat(wx, nph) * 2 * np.pi / nph)
        self.u = np.stack([np.sin(self.th) * np.cos(self.ph), np.sin(self.th) * np.sin(self.ph),
                           np.cos(self.th)], axis=-1)        # (nang, 3)
        self.nr, self.nang = len(self.r), len(self.th)
        self.W = (self.wr * self.r ** 2)[:, None] * self.wang[None, :]   # volume weights (nr, nang)
        self.X = self.r[:, None, None] * self.u[None, :, :]               # points (nr, nang, 3)
        self._Y, self._gS = {}, {}

    def Y(self, k, q):
        if (k, q) not in self._Y:
            self._Y[(k, q)] = sph_harm_y(k, q, self.th, self.ph)
        return self._Y[(k, q)]

    def gradS(self, k, q):
        """grad of the solid harmonic S_kq = r^k Y_kq at the unit vectors (nang, 3)."""
        if (k, q) not in self._gS:
            self._gS[(k, q)] = fd_grad(lambda P: solid(k, q, P), self.u)
        return self._gS[(k, q)]


def solid(l, m, P):
    """r^l Y_lm at Cartesian points P (..., 3)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    th = np.arccos(np.clip(z / np.where(r > 0, r, 1.0), -1, 1))
    ph = np.arctan2(y, x)
    return r ** l * sph_harm_y(l, m, th, ph)


_FD = 0.05
_C1 = [(-2, 1 / 12), (-1, -8 / 12), (1, 8 / 12), (2, -1 / 12)]                  # 4th-order first derivative
_C2 = [(-2, -1 / 12), (-1, 16 / 12), (0, -30 / 12), (1, 16 / 12), (2, -1 / 12)]  # 4th-order second derivative


def fd_grad(f, P):
    out = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = _FD
        out.append(sum(c * f(P + s * e) for s, c in _C1) / _FD)
    return np.stack(out, axis=-1)


def fd_hess(f, P):
    """Hessian (..., 3, 3) of f at P, exact for polynomials of degree <= 4."""
    H = np.zeros(P.shape[:-1] + (3, 3), dtype=complex)
    for i in range(3):
        ei = np.zeros(3)
        ei[i] = _FD
        H[..., i, i] = sum(c * f(P + s * ei) for s, c in _C2) / _FD ** 2
        for j in range(i + 1, 3):
            ej = np.zeros(3)
            ej[j] = _FD
            v = sum(ci * cj * f(P + si * ei + sj * ej) for si, ci in _C1 for sj, cj in _C1) / _FD ** 2
            H[..., i, j] = H[..., j, i] = v
    return H


# ---------------------------------------------------------------- orbitals
class Orb:
    """phi = exp(-zeta r^2) r^l Y_lm with value, gradient, Hessian on the grid."""

    def __init__(self, g, l, m, zeta):
        self.l, self.m, self.zeta = l, m, zeta
        X = g.X
        r2 = (X ** 2).sum(-1)
        e = np.exp(-zeta * r2)
        S = solid(l, m, X)
        gS = fd_grad(lambda P: solid(l, m, P), X)
        hS = fd_hess(lambda P: solid(l, m, P), X)
        ge = -2 * zeta * X * e[..., None]                                       # grad exp
        he = (4 * zeta ** 2 * X[..., :, None] * X[..., None, :] - 2 * zeta * np.eye(3)) * e[..., None, None]
        self.v = e * S
        self.g = ge * S[..., None] + e[..., None] * gS
        self.H = (he * S[..., None, None] + ge[..., :, None] * gS[..., None, :]
                  + gS[..., :, None] * ge[..., None, :] + e[..., None, None] * hS)
        self.lap = np.trace(self.H, axis1=-2, axis2=-1)


def radial(l, zeta):
    """R(r) = r^l exp(-zeta r^2) and R', R'' as functions."""
    R = lambda r: r ** l * np.exp(-zeta * r * r)
    dR = lambda r: (l * r ** (l - 1) if l else 0.0 * r) * np.exp(-zeta * r * r) - 2 * zeta * r ** (l + 1) * np.exp(-zeta * r * r)

    def d2R(r):
        e = np.exp(-zeta * r * r)
        t = -2 * zeta * (2 * l + 1) * r ** l + 4 * zeta ** 2 * r ** (l + 2)
        if l >= 2:
            t = t + l * (l - 1) * r ** (l - 2)
        return t * e
    return R, dR, d2R


# ---------------------------------------------------------------- Coulomb potentials
def _cumint(r0, y):
    """Cumulative integral from 0 of the columns of y (complex) on r0 (cubic spline)."""
    re = CubicSpline(r0, y.real, axis=0).antiderivative()(r0)
    im = CubicSpline(r0, y.imag, axis=0).antiderivative()(r0)
    return re + 1j * im


def _tables(g, kmax):
    key = ("tab", kmax)
    if key not in g._Y:
        kq = [(k, q) for k in range(kmax + 1) for q in range(-k, k + 1)]
        Ym = np.stack([g.Y(k, q) for k, q in kq])                                 # (nkq, nang)
        gS = np.stack([g.gradS(k, q) for k, q in kq])                             # (nkq, nang, 3)
        kk = np.array([k for k, _ in kq], dtype=float)
        g._Y[key] = (Ym, gS, kk)
    return g._Y[key]


def potential(g, f, kmax, grad=False):
    """V(r1) = int f(2)/r12 d2 on the grid (and grad V (nr, nang, 3) if asked).
    f: values (nr, nang) on the grid."""
    Ym, gS, kk = _tables(g, kmax)
    r0, r = g.r0, g.r[:, None]
    fk = (f * g.wang[None, :]) @ np.conj(Ym).T                                  # (nr, nkq)
    z = np.zeros((1, len(kk)), dtype=complex)
    A = _cumint(r0, np.vstack([z, fk * r ** (kk + 2)]))
    Bc = _cumint(r0, np.vstack([z, fk * r ** (1 - kk)]))
    B = (Bc[-1:] - Bc)[1:]
    A = A[1:]
    c = 4 * np.pi / (2 * kk + 1)
    v = c * (r ** (-kk - 1) * A + r ** kk * B)                                  # (nr, nkq)
    V = v @ Ym
    G = None
    if grad:
        dv = c * (-(kk + 1) * r ** (-kk - 2) * A + kk * r ** (kk - 1) * B)
        radf = dv - kk * v / r                                                   # (nr, nkq)
        G = np.einsum("rk,ka,ai->rai", radf, Ym, g.u) + np.einsum("rk,kai->rai", v / r, gS)
    return V, G


# ---------------------------------------------------------------- spin and assembly
SPIN = {  # <mu|s_i|mu'> for tmu = +1, -1 (index 0, 1), i = x, y, z
    0: 0.5 * np.array([[0, 1], [1, 0]], dtype=complex),
    1: 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex),
    2: 0.5 * np.array([[1, 0], [0, -1]], dtype=complex),
}
ID2 = np.eye(2, dtype=complex)
TMU = (1, -1)


def assemble(space, orbs, I, tol=1e-13):
    """Matrix of sum_{pqrs} I[p,q,r,s] a+_p a+_q a_s a_r; spin orbitals p = (orbital, spin index).
    orbs: list of (shell, m) per orbital index. I: array (no, 2, no, 2, no, 2, no, 2)."""
    bit = {(o, s): space.orb[(orbs[o][0], orbs[o][1], TMU[s])] for o in range(len(orbs)) for s in range(2)}
    terms = []
    for idx in zip(*np.nonzero(np.abs(I) > tol)):
        p, q, r, s = (idx[0], idx[1]), (idx[2], idx[3]), (idx[4], idx[5]), (idx[6], idx[7])
        terms.append((I[idx], [(True, bit[p]), (True, bit[q]), (False, bit[s]), (False, bit[r])]))
    return complex_matrix(space, terms)


def assemble1(space, orbs, I, tol=1e-13):
    """Matrix of sum_{pr} I[p,r] a+_p a_r; I: array (no, 2, no, 2)."""
    bit = {(o, s): space.orb[(orbs[o][0], orbs[o][1], TMU[s])] for o in range(len(orbs)) for s in range(2)}
    terms = []
    for idx in zip(*np.nonzero(np.abs(I) > tol)):
        terms.append((I[idx], [(True, bit[(idx[0], idx[1])]), (False, bit[(idx[2], idx[3])])]))
    return complex_matrix(space, terms)


def complex_matrix(space, terms):
    M = np.zeros((space.dim, space.dim), dtype=complex)
    for coef, ops in terms:
        for j, det in enumerate(space.dets):
            s, d = 1, det
            for cre, b in reversed(ops):
                rr = space._create(d, b) if cre else space._annihilate(d, b)
                if rr is None:
                    break
                s, d = s * rr[0], rr[1]
            else:
                i = space.index.get(d)
                if i is not None:
                    M[i, j] += s * coef
    return M


# ---------------------------------------------------------------- helpers for the note's formulas
def rmeC(l, k, lp):
    """<l||C^k||l'> (Edmonds)."""
    return (-1) ** l * math.sqrt((2 * l + 1) * (2 * lp + 1)) * threej2(2 * l, 0, 2 * k, 0, 2 * lp, 0)


_TBN = {}


def tbn(space, a, b, c, d, p, q, tot):
    key = (a, b, c, d, p, q, tot)
    if key not in _TBN:
        _TBN[key] = two_body_normal(space, a, b, c, d, p, q, tot)
    return _TBN[key]


def jscalar(W, kappa):
    """(W^(kappa kappa))^(0) = sum_nu <kappa nu kappa -nu|0 0> W_{nu,-nu}; W keyed (2 nu, q)."""
    return sum(cg2(2 * kappa, 2 * nu, 2 * kappa, -2 * nu, 0, 0) * W[(2 * nu, -nu)] for nu in range(-kappa, kappa + 1))


def jvector_z(W, ks, kl):
    """Component Q = 0 of [W^(ks kl)]^(1): sum <ks nu kl -nu|1 0> W_{nu,-nu}."""
    return sum(cg2(2 * ks, 2 * nu, 2 * kl, -2 * nu, 2, 0) * W.get((2 * nu, -nu), 0.0)
               for nu in range(-ks, ks + 1) if abs(nu) <= kl)


def radint(f, g):
    """int_0^rmax f(r) dr on the grid (Simpson); f given on g.r."""
    return float((f * g.wr).sum())


def Yk_radial(g, Rc, Rd, k):
    """Y^k(cd; r) of the note on g.r."""
    r0, r = g.r0, g.r
    rho = Rc(r) * Rd(r)
    ia = np.concatenate([[0.0], r ** (k + 2) * rho])
    ib = np.concatenate([[0.0], rho * r ** (1 - k)])
    A = CubicSpline(r0, ia).antiderivative()(r0)[1:]
    Bc = CubicSpline(r0, ib).antiderivative()(r0)
    B = (Bc[-1] - Bc)[1:]
    return A / r ** k + r ** (k + 1) * B


def Yk_of(g, rho, k):
    """r int r_<^k / r_>^(k+1) rho(r') r'^2 dr' for rho given on g.r."""
    r0, r = g.r0, g.r
    ia = np.concatenate([[0.0], r ** (k + 2) * rho])
    ib = np.concatenate([[0.0], rho * r ** (1 - k)])
    A = CubicSpline(r0, ia).antiderivative()(r0)[1:]
    Bc = CubicSpline(r0, ib).antiderivative()(r0)
    B = (Bc[-1] - Bc)[1:]
    return A / r ** k + r ** (k + 1) * B


def Wk_of(g, rho, k):
    """int W_k(r, r') rho(r') r'^2 dr' with W_k = r_<^(k+2)/((2k+3) r_>^(k+1)) - r_<^k/((2k-1) r_>^(k-1))."""
    r0, r = g.r0, g.r

    def cum(p):
        y = np.concatenate([[0.0], rho * r ** p])
        return CubicSpline(r0, y).antiderivative()(r0)

    def inner(p):
        return cum(p)[1:]

    def outer(p):
        c = cum(p)
        return (c[-1] - c)[1:]
    t1 = (r ** (-k - 1) * inner(k + 4) + r ** (k + 2) * outer(1 - k)) / (2 * k + 3)
    t2 = (r ** (-k + 1) * inner(k + 2) + r ** k * outer(3 - k)) / (2 * k - 1)
    return t1 - t2


def compare(name, Mref, Mnote):
    scale = np.abs(Mref).max()
    diff = np.abs(Mref - Mnote).max()
    ok = diff <= 1e-6 * max(scale, 1e-30)
    print(f"   {'ok  ' if ok else 'BAD '} {name:44s} max|ref| {scale:.6e}  max|ref - note|  {diff:.2e}  rel {diff / scale:.1e}")
    return ok
