"""Orbit-orbit Zeeman term of g for Gaussian orbitals, in momentum space (S2-M7, Z2).

  O_z = -(alpha^2/2) sum_{i != j} [r_i x (K_ij p_j)]_z,  K = delta/r + r r/r^3,

(cf. K. Pachucki, Phys. Rev. A 69, 052502 (2004), Eq. (32)),

for orbitals s = N_s exp(-a r^2) and p_m = N_p (Condon-Shortley) r Y-like
polynomials times exp(-b r^2): p_0 = N_p z e, p_+1 = -N_p (x + i y)/sqrt2 e,
p_-1 = N_p (x - i y)/sqrt2 e. Electron 1 (bra a, ket b) carries the position,
electron 2 (bra c, ket d) the momentum:

  G(a, b; c, d) = (i alpha^2/2) int int X_ab(1) . K . A_cd(2),
  X_ab = phi_a* phi_b (z^ x r),  A_cd = phi_c* grad phi_d   (as it stands: the
  longitudinal part of A is removed by K itself, nothing is projected here),
  = (i alpha^2/2) int d3q/(2 pi)^3 X~(q) . (8 pi/q^2)(1 - q^ q^) . A~(-q).

Fourier transforms of polynomial x Gaussian are analytic (recursion in
x^n); the q integral uses Gauss-Legendre in q, cos(theta) and a uniform phi.
None of this uses multipole expansions, the transversality split
K = 2 delta/r - grad grad r, or the tensor algebra of the C code.

usage: ozref.py a b    (prints <O_z> of |1s 2p_+1| 3P and 1P (M_L = 1) and of
                        |2p_+1 2p_0| 3P (M_L = 1))
"""
import math
import sys

import numpy as np

ALPHA = 1 / 137.036


class GPoly:
    """sum_t coef_t x^nx y^ny z^nz exp(-c r^2) (one exponent c)."""

    def __init__(self, terms, c):
        self.t = {}
        for k, v in terms.items():
            if v != 0:
                self.t[k] = self.t.get(k, 0) + v
        self.c = c

    def conj(self):
        return GPoly({k: np.conj(v) for k, v in self.t.items()}, self.c)

    def __mul__(self, o):
        out = {}
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                k = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2])
                out[k] = out.get(k, 0) + v1 * v2
        return GPoly(out, self.c + o.c)

    def scale(self, s):
        return GPoly({k: v * s for k, v in self.t.items()}, self.c)

    def times_coord(self, ax):
        out = {}
        for k, v in self.t.items():
            kk = list(k)
            kk[ax] += 1
            out[tuple(kk)] = out.get(tuple(kk), 0) + v
        return GPoly(out, self.c)

    def grad(self, ax):
        out = {}
        for k, v in self.t.items():
            if k[ax]:
                kk = list(k)
                kk[ax] -= 1
                out[tuple(kk)] = out.get(tuple(kk), 0) + v * k[ax]
            kk = list(k)
            kk[ax] += 1
            out[tuple(kk)] = out.get(tuple(kk), 0) - 2 * self.c * v
        return GPoly(out, self.c)

    def ft(self, q):
        """int f(r) exp(-i q.r) d3r at q (n, 3)."""
        nmax = max(max(k) for k in self.t) + 1
        c = self.c
        I = []
        for ax in range(3):
            x = q[:, ax]
            row = [np.sqrt(np.pi / c) * np.exp(-x * x / (4 * c))]
            row.append(-1j * x / (2 * c) * row[0])
            for n in range(1, nmax):
                row.append((n * row[n - 1] - 1j * x * row[n]) / (2 * c))
            I.append(row)
        out = np.zeros(len(q), complex)
        for k, v in self.t.items():
            out += v * I[0][k[0]] * I[1][k[1]] * I[2][k[2]]
        return out


def orbitals(a, b):
    Ns = (2 * a / math.pi) ** 0.75
    Np = (2 * b / math.pi) ** 0.75 * 2 * math.sqrt(b)
    s = GPoly({(0, 0, 0): Ns}, a)
    r2 = 1 / math.sqrt(2)
    p = {0: GPoly({(0, 0, 1): Np}, b),
         1: GPoly({(1, 0, 0): -Np * r2, (0, 1, 0): -1j * Np * r2}, b),
         -1: GPoly({(1, 0, 0): Np * r2, (0, 1, 0): -1j * Np * r2}, b)}
    return s, p


def qgrid(nq=160, nt=48, qmax=40.0):
    xq, wq = np.polynomial.legendre.leggauss(nq)
    q = (xq + 1) * qmax / 2
    wq = wq * qmax / 2
    xt, wt = np.polynomial.legendre.leggauss(nt)
    nph = 2 * nt
    ph = np.arange(nph) * 2 * np.pi / nph
    Q, CT, PH = np.meshgrid(q, xt, ph, indexing="ij")
    W = (wq[:, None, None] * wt[None, :, None] * (2 * np.pi / nph))
    W = np.broadcast_to(W, Q.shape)
    st = np.sqrt(1 - CT ** 2)
    pts = np.stack([Q * st * np.cos(PH), Q * st * np.sin(PH), Q * CT], axis=-1).reshape(-1, 3)
    return pts, (W * Q ** 2).reshape(-1)


def G(fa, fb, fc, fd, pts, w):
    """(i alpha^2/2) int int X_ab . K . A_cd."""
    f = fa.conj() * fb
    X = [f.times_coord(1).scale(-1), f.times_coord(0)]  # f (z^ x r) = f (-y, x, 0)
    A = [fc.conj() * fd.grad(ax) for ax in range(3)]
    Xq = np.array([X[0].ft(pts), X[1].ft(pts), np.zeros(len(pts))])
    Aq = np.array([A[ax].ft(-pts) for ax in range(3)])
    q2 = np.sum(pts * pts, axis=1)
    qX = np.einsum("ig,gi->g", Xq, pts)
    qA = np.einsum("ig,gi->g", Aq, pts)
    integrand = 8 * np.pi / q2 * (np.einsum("ig,ig->g", Xq, Aq) - qX * qA / q2)
    return 0.5j * ALPHA ** 2 * np.sum(w * integrand) / (2 * np.pi) ** 3


def main():
    a, b = float(sys.argv[1]), float(sys.argv[2])
    s, p = orbitals(a, b)
    pts, w = qgrid()
    pp = p[1]
    D = G(s, s, pp, pp, pts, w) + G(pp, pp, s, s, pts, w)
    X = G(s, pp, pp, s, pts, w) + G(pp, s, s, pp, pts, w)
    print(f"a={a} b={b}")
    print(f"1s2p 3P M_L=1: <O_z> = {(D - X).real:+.12e}  (imag {(D - X).imag:+.1e})")
    print(f"1s2p 1P M_L=1: <O_z> = {(D + X).real:+.12e}  (imag {(D + X).imag:+.1e})")
    p0 = p[0]
    D2 = G(pp, pp, p0, p0, pts, w) + G(p0, p0, pp, pp, pts, w)
    X2 = G(pp, p0, p0, pp, pts, w) + G(p0, pp, pp, p0, pts, w)
    print(f"2p2 3P M_L=1:  <O_z> = {(D2 - X2).real:+.12e}  (imag {(D2 - X2).imag:+.1e})")


if __name__ == "__main__":
    main()
