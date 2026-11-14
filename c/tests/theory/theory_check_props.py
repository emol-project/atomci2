# props: the coefficients of the technical note, Sec. 5, that are written as explicit
# formulas, against operators built in coordinate space (theory_coord.py):
#   g     (Sec. 5.1) rec2 (Theta_rec), GC1e (2/3 s_z and (s.Q) = -sqrt(5/3)[s x Q2]^(1), Q2 = sqrt(2/3) C2),
#         H6 rank 2 (kappa alpha^2 sqrt(5/3) [s x [p x p]^(2)]^(1))
#   hfs   (Sec. 5.2) T^(1) (orbital, spin-dipole, contact) and T^(2); the A_J, B_J reductions
#   trans (Sec. 5.3) E1 length and velocity, E2, M1 (L + g_s S); the one-electron M1 corrections (ss, orb, ten)
#         with the [p x p]^(2) formula
#   is    (Sec. 5.5) Theta_SMS
# One-electron operators are compared in the one-electron space, two-electron ones in the
# two-electron space; for vector operators the z (Q = 0) component, for rank 2 the Q = 0 component.
# alpha^2 = 1, kappa = 1, g_s = 2 (the factors are bookkeeping and checked elsewhere).
import itertools
import math
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import theory_coord as tc  # noqa: E402
from theory_wig import sixj2  # noqa: E402

which = sys.argv[1:] or ["g", "hfs", "trans", "is"]
NEG = "neg" in which


def neg(x):
    return (-1) ** x if NEG else 1


SH = [(0, 1.1), (1, 0.9), (1, 0.5), (2, 0.7)]
g = tc.Grid(nr=400, rmax=8.0, nth=12, nph=24)
shells = [l for l, _ in SH]
nmax = [2 * (2 * l + 1) for l in shells]
sp1 = tc.Space(shells, [o for o in itertools.product(*[range(n + 1) for n in nmax]) if sum(o) == 1])
sp2 = tc.Space(shells, [o for o in itertools.product(*[range(n + 1) for n in nmax]) if sum(o) == 2])
orbs = [(s, m) for s, (l, _) in enumerate(SH) for m in range(-l, l + 1)]
no = len(orbs)
O = [tc.Orb(g, SH[s][0], m, SH[s][1]) for (s, m) in orbs]
RAD = [tc.radial(l, z) for l, z in SH]
r = g.r
ns = len(SH)
print(f"one-electron dets {sp1.dim}, two-electron dets {sp2.dim}")
ok_all = True
Zn = 3.0      # nuclear charge in GC1e and the M1 corrections
kk = 0.7      # transition energy k in the M1 corrections


def C(l, k, lp):
    return tc.rmeC(l, k, lp)


def Dr(lam, l, s):
    """<lam||grad||l> R_s as a radial function (the gradient formula)."""
    R, dR, _ = RAD[s]
    if lam == l + 1:
        return math.sqrt(l + 1) * (dR(r) - l * R(r) / r)
    if lam == l - 1:
        return -math.sqrt(l) * (dR(r) + (l + 1) * R(r) / r)
    return 0.0 * r


def gradme(a, b):
    """<a||grad||b> = int R_a (<la||grad||lb> R_b) r^2 dr."""
    return tc.radint(RAD[a][0](r) * Dr(shells[a], shells[b], b) * r ** 2, g)


def rint(a, b, f):
    return tc.radint(RAD[a][0](r) * RAD[b][0](r) * f * r ** 2, g)


def ob(space, a, b, ts, kl):
    return tc.one_body_tensor(space, a, b, ts, kl)


def onebody_note(space, coef, ts, kl, Q=None):
    """sum_ab coef(a, b) (b_a b~_b)^(ts/2, kl); with Q: the component [ ]^(1)_0 (Q = 'z') or the (0, 0) / (0, q) one."""
    M = np.zeros((space.dim, space.dim), dtype=complex)
    for a in range(ns):
        for b in range(ns):
            c = coef(a, b)
            if abs(c) < 1e-15:
                continue
            T = ob(space, a, b, ts, kl)
            if Q == "z":
                M += c * tc.jvector_z(T, ts // 2, kl)
            else:
                M += c * T[Q]
    return M


def onebody_ref(space, fn):
    """sum_i o(i) with <a s|o|b s'> = fn(a, b) (2x2 spin matrix)."""
    I = np.zeros((no, 2, no, 2), dtype=complex)
    for a in range(no):
        for b in range(no):
            I[a, :, b, :] = fn(a, b)
    return tc.assemble1(space, orbs, I)


def gint(f):
    return (g.W * f).sum()


def phi0(o):
    """phi(0): only s orbitals, Y00 = 1/sqrt(4 pi)."""
    return 1 / math.sqrt(4 * math.pi) if O[o].l == 0 else 0.0


def sph_from_cart(V):
    """spherical components {q: ...} of a Cartesian vector (V_x, V_y, V_z)."""
    return {1: -(V[0] + 1j * V[1]) / math.sqrt(2), 0: V[2], -1: (V[0] - 1j * V[1]) / math.sqrt(2)}


def pp_rme(a, b):
    """<a||[p x p]^(2)||b> of Sec. 5.3 (Edmonds (7.1.1) through l'' = l +- 1, integrated by parts)."""
    la, lb = shells[a], shells[b]
    s = 0.0
    for l2 in (la - 1, la + 1):
        if l2 < 0 or abs(l2 - lb) != 1:
            continue
        sj = sixj2(2, 2, 4, 2 * lb, 2 * la, 2 * l2)
        s += sj * C(l2, 1, la) * C(l2, 1, lb) * tc.radint(Dfun(l2, a) * Dfun(l2, b) * r ** 2, g)
    return -math.sqrt(5) * (-1) ** (la + lb) * s


def Dfun(l2, s):
    """the radial operator D (l -> l''): d/dr - l/r (l'' = l + 1) or d/dr + (l + 1)/r (l'' = l - 1) on R_s."""
    R, dR, _ = RAD[s]
    l = shells[s]
    if l2 == l + 1:
        return dR(r) - l * R(r) / r
    return dR(r) + (l + 1) * R(r) / r


S = tc.SPIN
SZ, ID2 = S[2], tc.ID2

# ======================================================================== g (Sec. 5.1)
if "g" in which:
    print("== g (Sec. 5.1)")
    # rec2: O_z = sum_{i!=j} (x_i p_jy - y_i p_jx), p = -i grad
    pm = np.array([[[gint(np.conj(O[a].v) * (-1j) * O[b].g[..., i]) for b in range(no)] for a in range(no)] for i in range(3)])
    xm = np.stack([np.array([[gint(np.conj(O[a].v) * g.X[..., i] * O[b].v) for b in range(no)] for a in range(no)])
                   for i in range(3)])
    oz = np.einsum("ab,cd->abcd", xm[0], pm[1]) - np.einsum("ab,cd->abcd", xm[1], pm[0])
    I = np.einsum("abcd,ij,kl->aickbjdl", oz, ID2, ID2)
    Mref = tc.assemble(sp2, orbs, I)
    Mn = np.zeros_like(Mref)
    for a, b, c, d in itertools.product(range(ns), repeat=4):
        th = -math.sqrt(2) * (2 / 3) * C(shells[a], 1, shells[b]) * rint(a, b, r) * gradme(c, d) * neg(shells[a])
        if abs(th) > 1e-15:
            Mn += th * tc.tbn(sp2, a, b, c, d, (0, 1), (0, 1), (0, 1))[(0, 0)]
    ok_all &= tc.compare("rec2  Theta_rec (z)", Mref, Mn)

    # GC1e: O_z = (Z/2) sum (1/r)[s_z - (s.rhat) rhat_z]
    def gc1(a, b):
        out = np.zeros((2, 2), dtype=complex)
        f = np.conj(O[a].v) * O[b].v / g.r[:, None]
        out += 0.5 * Zn * gint(f) * SZ
        for i in range(3):
            out -= 0.5 * Zn * gint(f * g.u[None, :, i] * g.u[None, :, 2]) * S[i]
        return out
    Mref = onebody_ref(sp1, gc1)
    # note: (Z/2)/r [ (2/3) s_z + (sqrt(10)/3) [s x C2]^(1)_z ];  one-body rule
    rs = math.sqrt(1.5)
    Mn = onebody_note(sp1, lambda a, b: 0.5 * Zn * (2 / 3) * rint(a, b, 1 / r) * rs * (1.0 if shells[a] == shells[b] else 0.0)
                      * math.sqrt(2 * shells[a] + 1) / (-math.sqrt(3)), 2, 0, Q="z")
    Mn += onebody_note(sp1, lambda a, b: 0.5 * Zn * (math.sqrt(10) / 3) * rint(a, b, 1 / r) * rs * C(shells[a], 2, shells[b])
                       / (-math.sqrt(3 * 5)) * neg(1), 2, 2, Q="z")
    ok_all &= tc.compare("GC1e  2/3 s_z + (s.Q) decomposition (z)", Mref, Mn)

    # H6 rank 2: -sum_j s_j T_jz, T_jz = p_j p_z - delta_jz p^2/3  vs  sqrt(5/3) [s x [p x p]^(2)]^(1)_z
    def ppm(a, b, i, j):
        return -gint(np.conj(O[a].v) * O[b].H[..., i, j])

    def h6(a, b):
        out = np.zeros((2, 2), dtype=complex)
        p2 = sum(ppm(a, b, i, i) for i in range(3))
        for j in range(3):
            T = ppm(a, b, j, 2) - (p2 / 3 if j == 2 else 0.0)
            out -= T * S[j]
        return out
    Mref = onebody_ref(sp1, h6)
    Mn = onebody_note(sp1, lambda a, b: math.sqrt(5 / 3) * rs * pp_rme(a, b) / (-math.sqrt(3 * 5)) * neg(shells[a]), 2, 2, Q="z")
    ok_all &= tc.compare("H6 rank 2  with the [p x p]^(2) formula (z)", Mref, Mn)

# ======================================================================== hfs (Sec. 5.2)
if "hfs" in which:
    print("== hfs (Sec. 5.2)")
    rs = math.sqrt(1.5)

    def t1(a, b):
        out = np.zeros((2, 2), dtype=complex)
        f3 = np.conj(O[a].v) / g.r[:, None] ** 3
        # orbital l_z / r^3, l_z = -i (x d_y - y d_x)
        lz = -1j * (g.X[..., 0] * O[b].g[..., 1] - g.X[..., 1] * O[b].g[..., 0])
        out += gint(f3 * lz) * ID2
        # spin dipole (g_s/2 = 1): (3 (s.rhat) rhat_z - s_z)/r^3
        fb = f3 * O[b].v
        for i in range(3):
            out += 3 * gint(fb * g.u[None, :, i] * g.u[None, :, 2]) * S[i]
        out -= gint(fb) * SZ
        # contact (8 pi/3) delta(r) s_z
        out += (8 * math.pi / 3) * np.conj(phi0(a)) * phi0(b) * SZ
        return out
    Mref = onebody_ref(sp1, t1)

    def lrme(l):
        return math.sqrt(l * (l + 1) * (2 * l + 1))
    Mn = onebody_note(sp1, lambda a, b: (rint(a, b, r ** -3) * math.sqrt(2) * lrme(shells[a]) / (-math.sqrt(3)) * neg(shells[a])
                                        if shells[a] == shells[b] else 0.0), 0, 1, Q="z")
    Mn += onebody_note(sp1, lambda a, b: -math.sqrt(10) * rint(a, b, r ** -3) * rs * C(shells[a], 2, shells[b])
                       / (-math.sqrt(15)), 2, 2, Q="z")
    Mn += onebody_note(sp1, lambda a, b: ((2 / 3) * RAD[a][0](0.0) * RAD[b][0](0.0) * rs * 1.0 / (-math.sqrt(3))
                                         if shells[a] == shells[b] == 0 else 0.0), 2, 0, Q="z")
    ok_all &= tc.compare("hfs T^(1) = orb + sd + contact (z)", Mref, Mn)

    # T^(2)_0 = C^(2)_0 / r^3 = (3 cos^2 - 1)/2 / r^3
    def t2(a, b):
        return gint(np.conj(O[a].v) * O[b].v * (1.5 * g.u[None, :, 2] ** 2 - 0.5) / g.r[:, None] ** 3) * ID2
    Mref = onebody_ref(sp1, t2)
    Mn = onebody_note(sp1, lambda a, b: rint(a, b, r ** -3) * math.sqrt(2) * C(shells[a], 2, shells[b]) / (-math.sqrt(5)) * neg(shells[a]),
                      0, 2, Q=(0, 0))
    ok_all &= tc.compare("hfs T^(2) (Q = 0)", Mref, Mn)
    # A_J and B_J reductions: <JJ|T_0^(1)|JJ> = J <||T1||>/sqrt(J(J+1)(2J+1)),
    # <JJ|T^(2)_0|JJ> = sqrt(J(2J-1)/((J+1)(2J+1)(2J+3))) <||T2||>  (Wigner-Eckart)
    bad = 0
    for tj in range(1, 16):
        J = tj / 2
        w1 = tc.threej2(tj, -tj, 2, 0, tj, tj)          # (-1)^(J-J) = 1
        bad += abs(w1 - J / math.sqrt(J * (J + 1) * (2 * J + 1))) > 1e-13
        if tj >= 2:
            w2 = tc.threej2(tj, -tj, 4, 0, tj, tj)
            bad += abs(w2 - math.sqrt(J * (2 * J - 1) / ((J + 1) * (2 * J + 1) * (2 * J + 3)))) > 1e-13
    print(f"   {'ok  ' if not bad else 'BAD '} A_J, B_J reductions, J = 1/2 .. 15/2: {bad} bad")
    ok_all &= not bad

# ======================================================================== trans (Sec. 5.3)
if "trans" in which:
    print("== trans (Sec. 5.3)")
    # E1 length z, velocity d_z, E2 (3z^2 - r^2)/2
    Mref = onebody_ref(sp1, lambda a, b: gint(np.conj(O[a].v) * g.X[..., 2] * O[b].v) * ID2)
    Mn = onebody_note(sp1, lambda a, b: rint(a, b, r) * math.sqrt(2) * C(shells[a], 1, shells[b]) / (-math.sqrt(3)) * neg(shells[a]),
                      0, 1, Q=(0, 0))
    ok_all &= tc.compare("E1 length (Q = 0)", Mref, Mn)
    Mref = onebody_ref(sp1, lambda a, b: gint(np.conj(O[a].v) * O[b].g[..., 2]) * ID2)
    Mn = onebody_note(sp1, lambda a, b: gradme(a, b) * math.sqrt(2) / (-math.sqrt(3)) * neg(shells[a]), 0, 1, Q=(0, 0))
    ok_all &= tc.compare("E1 velocity (Q = 0)", Mref, Mn)
    Mref = onebody_ref(sp1, lambda a, b: gint(np.conj(O[a].v) * (1.5 * g.X[..., 2] ** 2 - 0.5 * (g.X ** 2).sum(-1)) * O[b].v) * ID2)
    Mn = onebody_note(sp1, lambda a, b: rint(a, b, r ** 2) * math.sqrt(2) * C(shells[a], 2, shells[b]) / (-math.sqrt(5)) * neg(shells[a]),
                      0, 2, Q=(0, 0))
    ok_all &= tc.compare("E2 (Q = 0)", Mref, Mn)

    # one-electron M1 corrections (z): ss, orb, ten
    def p2m(a, b):
        return sum(gint(np.conj(O[a].g[..., i]) * O[b].g[..., i]) for i in range(3))

    def scal(a, b, c1, c2, c3):
        """<a| c1 k^2 r^2 + c2 p^2 + c3 Z/r |b>"""
        return (c1 * kk ** 2 * gint(np.conj(O[a].v) * (g.X ** 2).sum(-1) * O[b].v) + c2 * p2m(a, b)
                + c3 * Zn * gint(np.conj(O[a].v) * O[b].v / g.r[:, None]))

    def lzop(a, b, c1, c2):
        """<a| l_z (c1 k^2 r^2 + c2 p^2) |b> = sum over the l_z eigenvalue (orbitals have definite m)."""
        return O[a].m * scal(a, b, c1, c2, 0.0) if O[a].m == O[b].m else 0.0

    def ppm(a, b, i, j):
        return -gint(np.conj(O[a].v) * O[b].H[..., i, j])

    def m1(a, b):
        out = -4 * scal(a, b, 1 / 12, 1 / 3, -1 / 6) * SZ          # ss
        out = out - lzop(a, b, 1 / 10, 1 / 2) * ID2                 # orb
        # ten: [s x X]^(1)_z with X^(2)_q built from Cartesian forms
        X = {}
        for q in (-2, -1, 0, 1, 2):
            Y2 = g.Y(2, q)
            c2q = math.sqrt(4 * math.pi / 5) * Y2                   # C^(2)_q
            v = (-(math.sqrt(10) / 15) * kk ** 2 * gint(np.conj(O[a].v) * (g.X ** 2).sum(-1) * c2q[None, :] * O[b].v)
                 + (math.sqrt(10) / 3) * Zn * gint(np.conj(O[a].v) * c2q[None, :] * O[b].v / g.r[:, None]))
            X[q] = v
        # [p x p]^(2)_q from p_mu p_nu
        comp = {1: {0: -1 / math.sqrt(2), 1: -1j / math.sqrt(2)}, 0: {2: 1.0}, -1: {0: 1 / math.sqrt(2), 1: -1j / math.sqrt(2)}}
        for q in (-2, -1, 0, 1, 2):
            pp = 0.0
            for mu in (-1, 0, 1):
                nu = q - mu
                if abs(nu) > 1:
                    continue
                cg = tc.cg2(2, 2 * mu, 2, 2 * nu, 4, 2 * q)
                for i, ci in comp[mu].items():
                    for j, cj in comp[nu].items():
                        pp += cg * ci * cj * ppm(a, b, i, j)
            X[q] += -(math.sqrt(15) / 6) * pp
        sph = {1: -(S[0] + 1j * S[1]) / math.sqrt(2), 0: S[2], -1: (S[0] - 1j * S[1]) / math.sqrt(2)}
        for nu in (-1, 0, 1):
            out = out + tc.cg2(2, 2 * nu, 4, -2 * nu, 2, 0) * X[-nu] * sph[nu]
        return out
    Mref = onebody_ref(sp1, m1)
    rs = math.sqrt(1.5)

    def same(a, b):
        return 1.0 if shells[a] == shells[b] else 0.0

    def lrme(l):
        return math.sqrt(l * (l + 1) * (2 * l + 1))

    def scal_r(a, b, c1, c2, c3):
        """radial <a| c1 k^2 r^2 + c2 p^2 + c3 Z/r |b> for la = lb (p^2 = int R_a' R_b' + l(l+1) R_a R_b / r^2)."""
        Ra, dRa, _ = RAD[a]
        Rb, dRb, _ = RAD[b]
        l = shells[a]
        p2 = tc.radint((dRa(r) * dRb(r) + l * (l + 1) * Ra(r) * Rb(r) / r ** 2) * r ** 2, g)
        return c1 * kk ** 2 * rint(a, b, r ** 2) + c2 * p2 + c3 * Zn * rint(a, b, 1 / r)
    Mn = onebody_note(sp1, lambda a, b: -4 * same(a, b) * scal_r(a, b, 1 / 12, 1 / 3, -1 / 6) * rs * math.sqrt(2 * shells[a] + 1)
                      / (-math.sqrt(3)), 2, 0, Q="z")
    Mn += onebody_note(sp1, lambda a, b: -same(a, b) * scal_r(a, b, 1 / 10, 1 / 2, 0.0) * math.sqrt(2) * lrme(shells[a])
                       / (-math.sqrt(3)) * neg(shells[a]), 0, 1, Q="z")
    Mn += onebody_note(sp1, lambda a, b: ((-(math.sqrt(10) / 15) * kk ** 2 * rint(a, b, r ** 2)
                                           + (math.sqrt(10) / 3) * Zn * rint(a, b, 1 / r)) * C(shells[a], 2, shells[b])
                                          - (math.sqrt(15) / 6) * pp_rme(a, b)) * rs / (-math.sqrt(15)), 2, 2, Q="z")
    ok_all &= tc.compare("M1 corrections ss + orb + ten (z)", Mref, Mn)

# ======================================================================== is (Sec. 5.5)
if "is" in which:
    print("== is (Sec. 5.5)")
    pm = np.stack([np.array([[gint(np.conj(O[a].v) * O[b].g[..., i]) for b in range(no)] for a in range(no)])
                   for i in range(3)])                         # <a|d_i|b>
    pp = -np.einsum("iab,icd->abcd", pm, pm)                   # <ac|p1.p2|bd>
    I = 0.5 * np.einsum("abcd,ij,kl->aickbjdl", pp, ID2, ID2)
    Mref = tc.assemble(sp2, orbs, I)
    Mn = np.zeros_like(Mref)
    for a, b, c, d in itertools.product(range(ns), repeat=4):
        th = gradme(a, b) * gradme(c, d) / math.sqrt(3) * neg(shells[a])
        if abs(th) > 1e-15:
            Mn += th * tc.tbn(sp2, a, b, c, d, (0, 1), (0, 1), (0, 0))[(0, 0)]
    ok_all &= tc.compare("SMS  Theta_SMS", Mref, Mn)

print("props:", "all passed" if ok_all else "FAILED")
