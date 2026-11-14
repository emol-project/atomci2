"""Two-electron spin-orbit operator by an independent route (S2-M5).

H_SO2 = -(alpha^2/2) sum_{i != j} r_ij^-3 (r_ij x p_i).(s_i + 2 s_j)
      =  (alpha^2/2) sum_{i != j} O_ij.(s_i + 2 s_j),  O_12 = grad_1(1/r12) x p_1
(H. A. Bethe and E. E. Salpeter, Quantum Mechanics of One- and Two-Electron
Atoms, Springer (1957), section 39).

Integrating by parts in electron 1 (grad x grad = 0),

  <a c|O_q|b d> = i int int (1/r12) [grad phi_a(1)* x grad phi_b(1)]_q phi_c(2)* phi_d(2),

so 1/r12 is not differentiated. With phi = R(r) Y_lm,
  grad phi = R' Y r^ + (R/r) G,  G = surface gradient of Y,
the cross product splits into R_a' R_b / r Y_a* (r^ x G_b), R_a R_b' / r (G_a* x r^) Y_b
and R_a R_b / r^2 (G_a* x G_b). 1/r12 = sum_k r<^k/r>^(k+1) 4pi/(2k+1) sum_mu
Y_kmu*(1) Y_kmu(2). Angular integrals: Gauss-Legendre x trapezoid on the
sphere (exact for these polynomials); radial: Y^k potentials on a ln r grid
with a 6-point cumulative rule (as ac_radial).

Spherical components of a Cartesian vector A: A_{+1} = -(A_x + i A_y)/sqrt2,
A_0 = A_z, A_{-1} = (A_x - i A_y)/sqrt2; O.s = sum_q (-1)^q O_q s_-q.
"""
import numpy as np
from scipy import special

ALPHA = 1 / 137.035999084


# ---------------------------------------------------------------- spherical harmonics
def ylm(l, m, th, ph):
    """Condon-Shortley Y_lm."""
    return special.sph_harm_y(l, m, th, ph) if hasattr(special, "sph_harm_y") \
        else special.sph_harm(m, l, ph, th)


def ylm_grad(l, m, th, ph):
    """Surface gradient of Y_lm as a Cartesian vector (3, ...):
    theta^ dY/dtheta + phi^ (1/sin) dY/dphi,
    dY/dtheta = m cot(theta) Y_lm + sqrt((l-m)(l+m+1)) e^(-i phi) Y_l,m+1."""
    y = ylm(l, m, th, ph)
    dth = m / np.tan(th) * y
    if m < l:
        dth = dth + np.sqrt((l - m) * (l + m + 1)) * np.exp(-1j * ph) * ylm(l, m + 1, th, ph)
    dph = 1j * m * y / np.sin(th)
    thv = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    phv = np.array([-np.sin(ph), np.cos(ph), 0 * th])
    return thv * dth + phv * dph


def spherical(v):
    """(3, ...) Cartesian -> dict q -> component."""
    return {1: -(v[0] + 1j * v[1]) / np.sqrt(2), 0: v[2], -1: (v[0] - 1j * v[1]) / np.sqrt(2)}


class Sphere:
    def __init__(self, n=40):
        x, w = np.polynomial.legendre.leggauss(n)
        nph = 2 * n
        ph = np.arange(nph) * 2 * np.pi / nph
        self.th = np.arccos(x)[:, None] * np.ones(nph)[None, :]
        self.ph = np.ones(n)[:, None] * ph[None, :]
        self.w = w[:, None] * np.ones(nph)[None, :] * (2 * np.pi / nph)
        self.rhat = np.array([np.sin(self.th) * np.cos(self.ph), np.sin(self.th) * np.sin(self.ph),
                              np.cos(self.th)])

    def integrate(self, f):
        return np.sum(self.w * f)


# ---------------------------------------------------------------- radial grid
class Grid:
    def __init__(self, rmin=1e-12, rmax=60.0, h=0.004):
        self.h = h
        self.x = np.arange(np.log(rmin), np.log(rmax) + h, h)
        self.r = np.exp(self.x)
        self.w = h * self.r
        # 6-point cumulative weights per interval (as ac_radial.c)
        self.wt = []
        for p in range(5):
            A = np.array([[j ** m for j in range(6)] for m in range(6)], float)
            rhs = np.array([((p + 1) ** (m + 1) - p ** (m + 1)) / (m + 1) for m in range(6)])
            self.wt.append(np.linalg.solve(A, rhs))

    def steps(self, f):
        """integral of f dx over each interval [x_i, x_i+1]"""
        n = len(f)
        out = np.empty(n - 1)
        for i in range(n - 1):
            j0 = min(max(i - 2, 0), n - 6)
            out[i] = self.h * np.dot(self.wt[i - j0], f[j0:j0 + 6])
        return out

    def yk(self, k, rho):
        """Y^k(r) = r^-(k+1) int_0^r rho s^k ds + r^k int_r^inf rho s^-(k+1) ds,
        rho given on the grid as a function of r (density times r^2)."""
        r = self.r
        a = np.concatenate([[0.0], np.cumsum(self.steps(rho * r ** (k + 1)))])
        st = self.steps(rho * r ** (-k))
        b = np.concatenate([np.cumsum(st[::-1])[::-1], [0.0]])
        return a * r ** (-(k + 1)) + b * r ** k


# ---------------------------------------------------------------- operator
def so2_orbital_integrals(orbs, grid, sphere, kmax):
    """orbs: list of (l, m, R, dR) with R, dR on grid.r. Returns O[q][a, c, b, d]
    = <a c|O_q|b d> for q in (-1, 0, 1)."""
    n = len(orbs)
    th, ph, rh = sphere.th, sphere.ph, sphere.rhat
    Y = [ylm(l, m, th, ph) for l, m, _, _ in orbs]
    G = [ylm_grad(l, m, th, ph) for l, m, _, _ in orbs]
    r = grid.r
    O = {q: np.zeros((n, n, n, n), complex) for q in (-1, 0, 1)}
    # electron-2 Gaunt-type integrals E2[k][mu][c, d] = int Y_c* Y_d Y_kmu
    E2, E1 = {}, {}
    for k in range(kmax + 1):
        for mu in range(-k, k + 1):
            yk = ylm(k, mu, th, ph)
            E2[k, mu] = np.array([[sphere.integrate(np.conj(Y[c]) * Y[d] * yk) for d in range(n)]
                                  for c in range(n)])
            ykc = np.conj(yk) * 4 * np.pi / (2 * k + 1)
            for a in range(n):
                for b in range(n):
                    t1 = np.conj(Y[a]) * np.cross(rh, G[b], axis=0)
                    t2 = np.cross(np.conj(G[a]), rh, axis=0) * Y[b]
                    t3 = np.cross(np.conj(G[a]), G[b], axis=0)
                    for nt, t in enumerate((t1, t2, t3)):
                        s = spherical(t)
                        for q in (-1, 0, 1):
                            E1[k, mu, a, b, nt, q] = sphere.integrate(s[q] * ykc)
    # radial: int r1^2 f_n(r1) Y^k_cd(r1) dr1, cached per radial function (by id)
    rid = [id(o[2]) for o in orbs]
    for k in range(kmax + 1):
        yk, radc = {}, {}
        for c in range(n):
            for d in range(n):
                if (rid[c], rid[d]) not in yk:
                    yk[rid[c], rid[d]] = grid.yk(k, orbs[c][2] * orbs[d][2] * r ** 2)
        for a in range(n):
            la, _, Ra, dRa = orbs[a]
            for b in range(n):
                lb, _, Rb, dRb = orbs[b]
                f = (dRa * Rb / r, Ra * dRb / r, Ra * Rb / r ** 2)
                for c in range(n):
                    for d in range(n):
                        key = (rid[a], rid[b], rid[c], rid[d])
                        if key not in radc:
                            radc[key] = [np.sum(grid.w * r ** 2 * fn * yk[rid[c], rid[d]])
                                         for fn in f]
                        rad = radc[key]
                        for mu in range(-k, k + 1):
                            e2 = E2[k, mu][c, d]
                            if abs(e2) < 1e-14:
                                continue
                            for q in (-1, 0, 1):
                                v = sum(E1[k, mu, a, b, nt, q] * rad[nt] for nt in range(3))
                                O[q][a, c, b, d] += 1j * v * e2
    return O


def so2_spin_orbital(O):
    """W[P,Q,R,S] (H_SO2 = sum W a+P a+R a_S a_Q, electron 1 on P, Q), spin orbitals
    P = 2 p + sigma, sigma = 0 for alpha."""
    n = O[0].shape[0]
    s = {1: np.array([[0, -1 / np.sqrt(2)], [0, 0]]), 0: np.array([[0.5, 0], [0, -0.5]]),
         -1: np.array([[0, 0], [1 / np.sqrt(2), 0]])}
    nso = 2 * n
    sp, sg = np.arange(nso) // 2, np.arange(nso) % 2
    d = np.eye(2)
    W = np.zeros((nso,) * 4, complex)
    for q in (-1, 0, 1):
        Oq = O[q][np.ix_(sp, sp, sp, sp)]  # [P(a), R(c), Q(b), S(d)]
        Oq = Oq.transpose(0, 2, 1, 3)       # [P, Q, R, S]
        sm = s[-q][sg][:, sg]
        W += (-1) ** q * Oq * (sm[:, :, None, None] * d[sg][:, sg][None, None, :, :] +
                               2 * d[sg][:, sg][:, :, None, None] * sm[None, None, :, :])
    return 0.5 * ALPHA ** 2 * W
