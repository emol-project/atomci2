# M2 and E3 one-body operators: tensor forms against coordinate space (theory_coord.py),
# in the one-electron space of shells s, p, p', d, f (Gaussian radial functions).
#   E3:  sum r^3 C^(3)                                        (Q = 0: r^3 C^(3)_0)
#   M2:  sum grad(r^2 C^(2)) . ((2/3) l + g_s s)              (mu_B; Q = 0: grad((3z^2 - r^2)/2) = (-x, -y, 2z))
#        = sqrt(6) r [C^(1) x ((2/3) l + g_s s)]^(2)          (grad(r^k C^(k)) = sqrt(k(2k-1)) r^(k-1)
#                                                              [C^(k-1) x e]^(k); orbital (0,2) via Edmonds
#                                                              (7.1.1), spin (1,1) coupled to J rank 2)
import itertools
import math
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import theory_coord as tc  # noqa: E402
from theory_wig import sixj2  # noqa: E402

gs = 2.00231930436256
SH = [(0, 1.1), (1, 0.9), (1, 0.5), (2, 0.7), (3, 0.6)]
g = tc.Grid(nr=400, rmax=8.0, nth=14, nph=28)
shells = [l for l, _ in SH]
nmax = [2 * (2 * l + 1) for l in shells]
sp1 = tc.Space(shells, [o for o in itertools.product(*[range(n + 1) for n in nmax]) if sum(o) == 1])
orbs = [(s, m) for s, (l, _) in enumerate(SH) for m in range(-l, l + 1)]
no = len(orbs)
O = [tc.Orb(g, SH[s][0], m, SH[s][1]) for (s, m) in orbs]
RAD = [tc.radial(l, z) for l, z in SH]
r = g.r
ns = len(SH)
S, ID2 = tc.SPIN, tc.ID2
ok = True


def C(l, k, lp):
    return tc.rmeC(l, k, lp)


def rint(a, b, f):
    return tc.radint(RAD[a][0](r) * RAD[b][0](r) * f * r ** 2, g)


def gint(f):
    return (g.W * f).sum()


def ref(fn):
    I = np.zeros((no, 2, no, 2), dtype=complex)
    for a in range(no):
        for b in range(no):
            I[a, :, b, :] = fn(a, b)
    return tc.assemble1(sp1, orbs, I)


def note(coef, ts, kl, tK):
    """sum_ab coef(a, b) [ (b_a b~_b)^(ts/2, kl) ]^(tK/2)_0"""
    M = np.zeros((sp1.dim, sp1.dim), dtype=complex)
    for a in range(ns):
        for b in range(ns):
            c = coef(a, b)
            if abs(c) < 1e-15:
                continue
            T = tc.one_body_tensor(sp1, a, b, ts, kl)
            K = tK // 2
            M += c * sum(tc.cg2(ts, 2 * nu, 2 * kl, -2 * nu, tK, 0) * T[(2 * nu, -nu)]
                         for nu in range(-(ts // 2), ts // 2 + 1) if abs(nu) <= kl)
    return M


# E3
Y30 = g.Y(3, 0) * math.sqrt(4 * math.pi / 7)
Mref = ref(lambda a, b: gint(np.conj(O[a].v) * (g.r[:, None] ** 3 * Y30[None, :]) * O[b].v) * ID2)
Mn = note(lambda a, b: rint(a, b, r ** 3) * math.sqrt(2) * C(shells[a], 3, shells[b]) / (-math.sqrt(7)), 0, 3, 6)
ok &= tc.compare("E3 = r^3 C^(3) (Q = 0)", Mref, Mn)


# M2 reference: (-x, -y, 2z) . ((2/3) l + g_s s), l = -i r x grad
def m2(a, b):
    X = g.X
    G = np.stack([-X[..., 0], -X[..., 1], 2 * X[..., 2]], axis=-1)
    gb = O[b].g
    lb = np.stack([-1j * (X[..., 1] * gb[..., 2] - X[..., 2] * gb[..., 1]),
                   -1j * (X[..., 2] * gb[..., 0] - X[..., 0] * gb[..., 2]),
                   -1j * (X[..., 0] * gb[..., 1] - X[..., 1] * gb[..., 0])], axis=-1)
    out = (2 / 3) * sum(gint(np.conj(O[a].v) * G[..., i] * lb[..., i]) for i in range(3)) * ID2
    for i in range(3):
        out = out + gs * gint(np.conj(O[a].v) * G[..., i] * O[b].v) * S[i]
    return out


Mref = ref(m2)


def orb_rme(la, lb):
    """<la||[C^(1) x l]^(2)||lb> = (-1)^(la+2+lb) sqrt5 {1 1 2; lb la lb} <la||C^1||lb> <lb||l||lb>"""
    if lb == 0:
        return 0.0
    return ((-1) ** (la + lb) * math.sqrt(5) * sixj2(2, 2, 4, 2 * lb, 2 * la, 2 * lb) * C(la, 1, lb)
            * math.sqrt(lb * (lb + 1) * (2 * lb + 1)))


rs = math.sqrt(1.5)
Mn = note(lambda a, b: math.sqrt(6) * (2 / 3) * rint(a, b, r) * math.sqrt(2) * orb_rme(shells[a], shells[b])
          / (-math.sqrt(5)), 0, 2, 4)
Mn += note(lambda a, b: math.sqrt(6) * gs * rint(a, b, r) * rs * C(shells[a], 1, shells[b]) / (-math.sqrt(9)),
           2, 1, 4)
ok &= tc.compare("M2 = sqrt6 r [C^(1) x ((2/3) l + g_s s)]^(2) (Q = 0)", Mref, Mn)


# M3 (2026-11-02): grad(r^3 C^(3)_0) = grad(z^3 - (3/2) z (x^2 + y^2)) = (-3xz, -3yz, 3z^2 - (3/2)(x^2 + y^2)),
# times ((1/2) l + g_s s), against sqrt15 r^2 [C^(2) x ((1/2) l + g_s s)]^(3); the unit tensors
# (b_a b~_b)^(sigma k) have <a||.||b> = -sqrt([sigma][k]) (hence the divisors -sqrt7, -sqrt15)
def m3(a, b):
    X = g.X
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    G = np.stack([-3 * x * z, -3 * y * z, 3 * z ** 2 - 1.5 * (x ** 2 + y ** 2)], axis=-1)
    gb = O[b].g
    lb = np.stack([-1j * (X[..., 1] * gb[..., 2] - X[..., 2] * gb[..., 1]),
                   -1j * (X[..., 2] * gb[..., 0] - X[..., 0] * gb[..., 2]),
                   -1j * (X[..., 0] * gb[..., 1] - X[..., 1] * gb[..., 0])], axis=-1)
    out = 0.5 * sum(gint(np.conj(O[a].v) * G[..., i] * lb[..., i]) for i in range(3)) * ID2
    for i in range(3):
        out = out + gs * gint(np.conj(O[a].v) * G[..., i] * O[b].v) * S[i]
    return out


def orb_rme3(la, lb):
    """<la||[C^(2) x l]^(3)||lb> = (-1)^(la+3+lb) sqrt7 {2 1 3; lb la lb} <la||C^2||lb> <lb||l||lb>"""
    if lb == 0:
        return 0.0
    return ((-1) ** (la + lb + 3) * math.sqrt(7) * sixj2(4, 2, 6, 2 * lb, 2 * la, 2 * lb) * C(la, 2, lb)
            * math.sqrt(lb * (lb + 1) * (2 * lb + 1)))


Mref = ref(m3)
Mn = note(lambda a, b: math.sqrt(15) * 0.5 * rint(a, b, r ** 2) * math.sqrt(2) * orb_rme3(shells[a], shells[b])
          / (-math.sqrt(7)), 0, 3, 6)
Mn += note(lambda a, b: math.sqrt(15) * gs * rint(a, b, r ** 2) * rs * C(shells[a], 2, shells[b])
           / (-math.sqrt(15)), 2, 2, 6)
ok &= tc.compare("M3 = sqrt15 r^2 [C^(2) x ((1/2) l + g_s s)]^(3) (Q = 0)", Mref, Mn)
print("multipole:", "all passed" if ok else "FAILED")
