# conv: numerical checks of formulas of the technical note (Secs. 3.1 and 3.2).
#   Sec. 3.1: the reduced elements of the parts of a coupled product and of a scalar
#         product against Edmonds (7.1.5) with the 9j with a zero argument, and the
#         9j sum formula against a brute-force CG sum.
#   Sec. 3.2: the contraction (product = normal product + contraction term) and the
#         one-electron reduced elements, in the determinant space of pyref.py.
import itertools
import math
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "python"))
from pyref import Space, cg2, one_body_tensor, two_body_normal  # noqa: E402

F = math.factorial


def tri2(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def delta2(a, b, c):
    return math.sqrt(F((a + b - c) // 2) * F((a - b + c) // 2) * F((-a + b + c) // 2) / F((a + b + c) // 2 + 1))


def sixj2(a, b, c, d, e, f):
    """6j with doubled arguments (Racah formula)."""
    if not (tri2(a, b, c) and tri2(a, e, f) and tri2(d, b, f) and tri2(d, e, c)):
        return 0.0
    pre = delta2(a, b, c) * delta2(a, e, f) * delta2(d, b, f) * delta2(d, e, c)
    s = 0.0
    lo = max(a + b + c, a + e + f, d + b + f, d + e + c) // 2
    hi = min(a + b + d + e, b + c + e + f, a + c + d + f) // 2
    for t in range(lo, hi + 1):
        s += (-1) ** t * F(t + 1) / (F(t - (a + b + c) // 2) * F(t - (a + e + f) // 2) * F(t - (d + b + f) // 2)
                                     * F(t - (d + e + c) // 2) * F((a + b + d + e) // 2 - t)
                                     * F((b + c + e + f) // 2 - t) * F((a + c + d + f) // 2 - t))
    return pre * s


def ninej2(a, b, c, d, e, f, g, h, i):
    """9j (doubled) by the 6j sum of Sec. 3.1."""
    s = 0.0
    for x in range(0, 40):
        s += (-1) ** x * (x + 1) * sixj2(a, d, g, h, i, x) * sixj2(b, e, h, d, x, f) * sixj2(c, f, i, x, a, b)
    return s


def ninej2_brute(j11, j12, j13, j21, j22, j23, j31, j32, j33):
    """9j from its definition as a recoupling overlap, via CG sums."""
    s = 0.0
    for m11 in range(-j11, j11 + 1, 2):
        for m12 in range(-j12, j12 + 1, 2):
            for m21 in range(-j21, j21 + 1, 2):
                for m22 in range(-j22, j22 + 1, 2):
                    m13, m23, m31, m32 = m11 + m12, m21 + m22, m11 + m21, m12 + m22
                    m33 = m13 + m23
                    if abs(m33) > j33 or m33 != j33:
                        continue
                    s += (cg2(j11, m11, j12, m12, j13, m13) * cg2(j21, m21, j22, m22, j23, m23)
                          * cg2(j13, m13, j23, m23, j33, m33) * cg2(j11, m11, j21, m21, j31, m31)
                          * cg2(j12, m12, j22, m22, j32, m32) * cg2(j31, m31, j32, m32, j33, m33))
    return s / math.sqrt((j13 + 1) * (j23 + 1) * (j31 + 1) * (j32 + 1))


def ph(x2):
    """(-1)^(x2/2) for an even doubled exponent."""
    assert x2 % 2 == 0
    return -1.0 if (x2 // 2) % 2 else 1.0


bad = 0
n = 0
# 9j sum formula against the brute-force definition
for args in [(1, 1, 2, 1, 1, 0, 2, 2, 2), (2, 1, 3, 2, 1, 1, 4, 2, 4), (2, 2, 4, 2, 2, 2, 2, 2, 2),
             (1, 2, 3, 3, 2, 1, 2, 2, 2), (4, 2, 2, 2, 2, 4, 2, 4, 6), (1, 1, 0, 1, 1, 2, 2, 2, 2)]:
    n += 1
    if abs(ninej2(*args) - ninej2_brute(*args)) > 1e-12:
        bad += 1
        print("9j", args, ninej2(*args), ninej2_brute(*args))

J = range(0, 5)
for j1, j1p, j2, j2p, k in itertools.product(J, J, J, J, (0, 2, 4)):
    for Jt in J:
        for Jp in J:
            if tri2(j1, j2, Jt) and tri2(j1p, j2, Jp) and tri2(Jt, Jp, k) and tri2(j1, j1p, k):
                g = math.sqrt((Jt + 1) * (Jp + 1) * (k + 1)) * ninej2(j1, j1p, k, j2, j2, 0, Jt, Jp, k) * math.sqrt(j2 + 1)
                f = ph(j1 + j2 + Jp + k) * math.sqrt((Jt + 1) * (Jp + 1)) * sixj2(j1, Jt, j2, Jp, j1p, k)
                n += 1
                if abs(g - f) > 1e-12:
                    bad += 1
                    print("part1", j1, j1p, j2, Jt, Jp, k, g, f)
            if j2p != j2 and tri2(j1, j2, Jt) and tri2(j1, j2p, Jp) and tri2(Jt, Jp, k) and tri2(j2, j2p, k):
                pass
            if tri2(j1, j2, Jt) and tri2(j1, j2p, Jp) and tri2(Jt, Jp, k) and tri2(j2, j2p, k):
                g = math.sqrt((Jt + 1) * (Jp + 1) * (k + 1)) * ninej2(j1, j1, 0, j2, j2p, k, Jt, Jp, k) * math.sqrt(j1 + 1)
                f = ph(j1 + j2p + Jt + k) * math.sqrt((Jt + 1) * (Jp + 1)) * sixj2(j2, Jt, j1, Jp, j2p, k)
                n += 1
                if abs(g - f) > 1e-12:
                    bad += 1
                    print("part2", j1, j2, j2p, Jt, Jp, k, g, f)
        if tri2(j1, j2, Jt) and tri2(j1p, j2p, Jt) and tri2(j1, j1p, k) and tri2(j2, j2p, k):
            rme0 = (Jt + 1) * ninej2(j1, j1p, k, j2, j2p, k, Jt, Jt, 0)
            g = ph(k) * math.sqrt(k + 1) * rme0 / math.sqrt(Jt + 1)
            f = ph(j1p + j2 + Jt) * sixj2(Jt, j2, j1, k, j1p, j2p)
            n += 1
            if abs(g - f) > 1e-12:
                bad += 1
                print("scalar", j1, j1p, j2, j2p, Jt, k, g, f)
print("Sec. 3.1: tests", n, "bad", bad)

# ---------------------------------------------------------------- Sec. 3.2
bad2 = 0
n2 = 0


def couple(A, B, p, q, tot):
    """(A B)^tot from component dicts {(tnu, q): matrix}; p, q, tot = (2 sigma, k)."""
    out = {}
    for tnu in range(-tot[0], tot[0] + 1, 2):
        for mq in range(-tot[1], tot[1] + 1):
            M = 0.0
            for (t1, q1), A1 in A.items():
                t2, q2 = tnu - t1, mq - q1
                if (t2, q2) not in B:
                    continue
                c = cg2(p[0], t1, q[0], t2, tot[0], tnu) * cg2(2 * p[1], 2 * q1, 2 * q[1], 2 * q2, 2 * tot[1], 2 * mq)
                if c:
                    M = M + c * (A1 @ B[(t2, q2)])
            out[(tnu, mq)] = M
    return out


for shells, N in (([1, 1], 2), ([1, 2], 3), ([0, 1, 2], 2)):
    configs = [occ for occ in itertools.product(*[range(0, 4 * l + 3) for l in shells]) if sum(occ) == N]
    sp = Space(shells, configs)
    ns = len(shells)
    n2_start = n2
    ranks = [(ts, kl) for ts in (0, 2) for kl in range(0, 4)]
    for a, b, c, d in itertools.product(range(ns), repeat=4):
        la, lb, lc, ld = (shells[x] for x in (a, b, c, d))
        for p in ranks:
            if not tri2(2 * la, 2 * lb, 2 * p[1]):
                continue
            for q in ranks:
                if not tri2(2 * lc, 2 * ld, 2 * q[1]):
                    continue
                for tot in [(ts, kl) for ts in range(abs(p[0] - q[0]), p[0] + q[0] + 1, 4) for kl in range(abs(p[1] - q[1]), p[1] + q[1] + 1)]:
                    if n2 - n2_start > 600:
                        break
                    A = one_body_tensor(sp, a, b, *p)
                    B = one_body_tensor(sp, c, d, *q)
                    prod = couple(A, B, p, q, tot)
                    norm = two_body_normal(sp, a, b, c, d, p, q, tot)
                    if b == c and tri2(2 * la, 2 * ld, 2 * tot[1]):
                        one = one_body_tensor(sp, a, d, *tot)
                        # (-1)^(a+d+K): a, d = (1/2, l); K = (sigma, k)
                        e2 = 1 + 1 + 2 * la + 2 * ld + tot[0] + 2 * tot[1]
                        coef = -ph(e2) * math.sqrt((p[0] + 1) * (2 * p[1] + 1) * (q[0] + 1) * (2 * q[1] + 1)) \
                            * sixj2(p[0], q[0], tot[0], 1, 1, 1) * sixj2(2 * p[1], 2 * q[1], 2 * tot[1], 2 * ld, 2 * la, 2 * lb)
                    else:
                        one, coef = None, 0.0
                    err = 0.0
                    for key in prod:
                        diff = prod[key] - norm[key]
                        if one is not None:
                            diff = diff - coef * one[key]
                        err = max(err, float(np.max(np.abs(diff))) if np.ndim(diff) else abs(diff))
                    n2 += 1
                    if err > 1e-12:
                        bad2 += 1
                        if bad2 < 10:
                            print("contraction", shells, (a, b, c, d), p, q, tot, err)
print("Sec. 3.2 contraction: tests", n2, "bad", bad2)

# one-electron reduced element of (b_a b~_b)^K: -sqrt[K]
sp = Space([1, 2], [(1, 0), (0, 1)])
bad3 = 0
for ts in (0, 2):
    for kl in (1, 2, 3):
        T = one_body_tensor(sp, 0, 1, ts, kl)
        # <a alpha| T_{nu q} |b beta> with a = p (l=1), b = d (l=2)
        for (tnu, q), M in T.items():
            for m1 in range(-1, 2):
                for t1 in (1, -1):
                    for m2 in range(-2, 3):
                        for t2 in (1, -1):
                            i = sp.index[1 << sp.orb[(0, m1, t1)]]
                            j = sp.index[1 << sp.orb[(1, m2, t2)]]
                            w = (ph(1 - t1) * (-1) ** (1 - m1)
                                 * cg2(1, -t1, ts, tnu, 1, -t2) * 0 + 1)  # placeholder, replaced below
                            # Wigner-Eckart, spin and orbit: (-1)^(j-m) (j k j'; -m q m')
                            from pyref import threej2
                            we = (ph(1 - t1) * threej2(1, -t1, ts, tnu, 1, t2)
                                  * (-1) ** (1 - m1) * threej2(2, -2 * m1, 2 * kl, 2 * q, 4, 2 * m2))
                            rme = -math.sqrt((ts + 1) * (2 * kl + 1))
                            if abs(M[i, j] - we * rme) > 1e-12:
                                bad3 += 1
print("Sec. 3.2 one-electron rme: bad", bad3)
