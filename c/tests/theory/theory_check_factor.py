# factor (technical note, Secs. 3.4 and 3.5), in the determinant space of two shells (pyref.Space):
#  (a) invariant <F~ O G> = (-1)^(F+O-G) <F||O||G> (Edmonds), and Wigner-Eckart for every component;
#  (b) factorization <F~_2 H_2 G_2> = L_1 L_2 with MOTECC (51) as printed,
#      L_k = (-1)^(n(g_k) n(H_{k-1})) sqrt([H_k][G_k][F_k]) 9j{h_k H_{k-1} H_k; g_k G_{k-1} G_k; f_k F_{k-1} F_k} <f~_k h_k g_k>;
#  (c) MOTECC (48) with a complete intermediate set: <f~ (h1 h2)^h g> = sqrt[h] sum_k (-1)^(h1+h2+f+g) {h1 h2 h; g f k} <f~ h1 k><k~ h2 g>.
# Ranks are (2S, L) pairs. Tensors are dicts {(2M_S, M_L): matrix}.
import itertools
import math
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "python"))
from pyref import Space, cg2, threej2  # noqa: E402

F_ = math.factorial


def _tri2(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def _delta2(a, b, c):
    return math.sqrt(F_((a + b - c) // 2) * F_((a - b + c) // 2) * F_((-a + b + c) // 2) / F_((a + b + c) // 2 + 1))


def sixj2(a, b, c, d, e, f):
    """6j with doubled arguments (Racah formula)."""
    if not (_tri2(a, b, c) and _tri2(a, e, f) and _tri2(d, b, f) and _tri2(d, e, c)):
        return 0.0
    pre = _delta2(a, b, c) * _delta2(a, e, f) * _delta2(d, b, f) * _delta2(d, e, c)
    s = 0.0
    lo = max(a + b + c, a + e + f, d + b + f, d + e + c) // 2
    hi = min(a + b + d + e, b + c + e + f, a + c + d + f) // 2
    for t in range(lo, hi + 1):
        s += (-1) ** t * F_(t + 1) / (F_(t - (a + b + c) // 2) * F_(t - (a + e + f) // 2) * F_(t - (d + b + f) // 2)
                                      * F_(t - (d + e + c) // 2) * F_((a + b + d + e) // 2 - t)
                                      * F_((b + c + e + f) // 2 - t) * F_((a + c + d + f) // 2 - t))
    return pre * s


def ninej2(a, b, c, d, e, f, g, h, i):
    """9j (doubled) as a sum of 6j products (checked in theory_check_conv.py)."""
    s = 0.0
    for x in range(0, 40):
        s += (-1) ** x * (x + 1) * sixj2(a, d, g, h, i, x) * sixj2(b, e, h, d, x, f) * sixj2(c, f, i, x, a, b)
    return s


def sgn(x2):
    """(-1)^(x2/2), x2 even."""
    assert x2 % 2 == 0, x2
    return -1.0 if (x2 // 2) % 2 else 1.0


def comps(r):
    return [(tm, ml) for tm in range(-r[0], r[0] + 1, 2) for ml in range(-r[1], r[1] + 1)]


def tri(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def rtri(A, B, C):
    return tri(A[0], B[0], C[0]) and tri(2 * A[1], 2 * B[1], 2 * C[1])


def couple(A, rA, B, rB, rC):
    out = {}
    for (tm, ml) in comps(rC):
        M = None
        for (ta, ma) in comps(rA):
            tb, mb = tm - ta, ml - ma
            if abs(tb) > rB[0] or abs(mb) > rB[1]:
                continue
            c = cg2(rA[0], ta, rB[0], tb, rC[0], tm) * cg2(2 * rA[1], 2 * ma, 2 * rB[1], 2 * mb, 2 * rC[1], 2 * ml)
            if c:
                t = c * (A[(ta, ma)] @ B[(tb, mb)])
                M = t if M is None else M + t
        out[(tm, ml)] = M if M is not None else 0.0 * next(iter(A.values()))
    return out


def adjoint(A, r):
    return {(tm, ml): sgn(r[0] - tm) * (-1) ** (r[1] - ml) * A[(-tm, -ml)].T for (tm, ml) in comps(r)}


def dimr(r):
    return (r[0] + 1) * (2 * r[1] + 1)


def sixjr(a, b, c, d, e, f):
    return sixj2(a[0], b[0], c[0], d[0], e[0], f[0]) * sixj2(2 * a[1], 2 * b[1], 2 * c[1], 2 * d[1], 2 * e[1], 2 * f[1])


def ninejr(*x):
    return ninej2(*[y[0] for y in x]) * ninej2(*[2 * y[1] for y in x])


def ph3(A, B, C):
    """(-1)^(A+B-C) for rank pairs."""
    return sgn(A[0] + B[0] - C[0]) * (-1) ** (A[1] + B[1] - C[1])


def invariant(F, rF, H, rH, G, rG):
    if not rtri(rH, rG, rF):
        return 0.0
    HG = couple(H, rH, G, rG, rF)
    X = couple(adjoint(F, rF), rF, HG, rF, (0, 0))
    return X[(0, 0)][0, 0]


def edmonds(F, rF, O, rO, G, rG):
    """<F||O||G> from all components; returns (value, max inconsistency)."""
    vals = []
    for (tf, mf) in comps(rF):
        for (to, mo) in comps(rO):
            for (tg, mg) in comps(rG):
                w = (sgn(rF[0] - tf) * threej2(rF[0], -tf, rO[0], to, rG[0], tg)
                     * (-1) ** (rF[1] - mf) * threej2(2 * rF[1], -2 * mf, 2 * rO[1], 2 * mo, 2 * rG[1], 2 * mg))
                me = (F[(tf, mf)].T @ O[(to, mo)] @ G[(tg, mg)])[0, 0]
                if abs(w) > 1e-9:
                    vals.append(me / w)
                elif abs(me) > 1e-12:
                    return None, 1.0
    if not vals:
        return 0.0, 0.0
    return vals[0], max(abs(v - vals[0]) for v in vals)


# ---------------------------------------------------------------- space
shells = [1, 0]          # coupling shell 1 = p (index 0), shell 2 = s (index 1)
sp = Space(shells, [occ for occ in itertools.product(range(7), range(3))])
I = np.eye(sp.dim)


def bten(k):
    l = shells[k]
    return {(t, m): sp.matrix([(1.0, [(True, sp.orb[(k, m, t)])])]) for (t, m) in comps((1, l))}


B = [bten(k) for k in range(2)]
RB = [(1, shells[k]) for k in range(2)]
BT = [adjoint(B[k], RB[k]) for k in range(2)]
ONE = {(0, 0): I}
R0 = (0, 0)


def shell_states(k):
    """[(tensor, rank, N)] of shell k for N = 0, 1, 2 (all two-electron terms)."""
    out = [(ONE, R0, 0), (B[k], RB[k], 1)]
    l = shells[k]
    for ts in (0, 2):
        for L in range(0, 2 * l + 1):
            if (ts // 2 + L) % 2 == 0:
                T = couple(B[k], RB[k], B[k], RB[k], (ts, L))
                out.append(({c: v / math.sqrt(2) for c, v in T.items()}, (ts, L), 2))
    return out


def shell_ops(k):
    """[(tensor, rank, n_created)]: 1, b, b~, (b b~)^x."""
    out = [(ONE, R0, 0), (B[k], RB[k], 1), (BT[k], RB[k], -1)]
    l = shells[k]
    for ts in (0, 2):
        for L in range(0, 2 * l + 1):
            out.append((couple(B[k], RB[k], BT[k], RB[k], (ts, L)), (ts, L), 0))
    return out


def ranks_between(A, Bq):
    return [(ts, L) for ts in range(abs(A[0] - Bq[0]), A[0] + Bq[0] + 1, 2)
            for L in range(abs(A[1] - Bq[1]), A[1] + Bq[1] + 1)]


# ---------------------------------------------------------------- (a) and (b)
bad_a = bad_b = n_a = n_b = 0
S1, S2 = shell_states(0), shell_states(1)
O1, O2 = shell_ops(0), shell_ops(1)
csfs = []
for (f1, r1, n1) in S1:
    for (f2, r2, n2) in S2:
        for rF in ranks_between(r2, r1):
            if n1 + n2 > 0 and (rF[0] % 2) != ((n1 + n2) % 2):
                continue
            csfs.append((couple(f2, r2, f1, r1, rF), rF, (f1, r1, n1), (f2, r2, n2)))
print("CSFs", len(csfs))
import random  # noqa: E402
random.seed(1)
for (h1, rh1, c1) in O1:
    for (h2, rh2, c2) in O2:
        for rH in ranks_between(rh2, rh1):
            H = couple(h2, rh2, h1, rh1, rH)
            for (F, rF, s1f, s2f) in random.sample(csfs, 12):
                for (G, rG, s1g, s2g) in csfs:
                    if s1f[2] != s1g[2] + c1 or s2f[2] != s2g[2] + c2:
                        continue
                    if not rtri(rH, rG, rF):
                        continue
                    inv = invariant(F, rF, H, rH, G, rG)
                    ed, incons = edmonds(F, rF, H, rH, G, rG)
                    n_a += 1
                    if ed is None or incons > 1e-10 or abs(inv - ph3(rF, rH, rG) * ed) > 1e-10:
                        bad_a += 1
                        if bad_a < 5:
                            print("(a)", rF, rH, rG, inv, ed, incons)
                    # (b) factorization, k = 1 (shell index 0), k = 2 (shell index 1)
                    (f1, r1, n1), (f2, r2, n2) = s1f, s2f
                    (g1, q1, m1), (g2, q2, m2) = s1g, s2g
                    L1 = (math.sqrt(dimr(rh1) * dimr(q1) * dimr(r1))
                          * ninejr(rh1, R0, rh1, q1, R0, q1, r1, R0, r1) * invariant(f1, r1, h1, rh1, g1, q1))
                    L2 = (sgn(2 * m2 * c1) * math.sqrt(dimr(rH) * dimr(rG) * dimr(rF))
                          * ninejr(rh2, rh1, rH, q2, q1, rG, r2, r1, rF) * invariant(f2, r2, h2, rh2, g2, q2))
                    n_b += 1
                    if abs(inv - L1 * L2) > 1e-10:
                        bad_b += 1
                        if bad_b < 5:
                            print("(b)", rF, rH, rG, inv, L1 * L2)
print("(a) invariant vs Edmonds: tests", n_a, "bad", bad_a)
print("(b) factorization (51): tests", n_b, "bad", bad_b)

# ---------------------------------------------------------------- (c) MOTECC (48) in the p shell
# <f~ (h1 h2)^h g> with f, g two-electron states, h1 = b, h2 = b~: intermediates = one-electron states {b}.
bad_c = n_c = 0
two = [s for s in S1 if s[2] == 2]
rb = RB[0]
for (f, rf, _) in two:
    for (g, rg, _) in two:
        for rh in ranks_between(rb, rb):
            if not rtri(rh, rg, rf):
                continue
            lhs = invariant(f, rf, couple(B[0], rb, BT[0], rb, rh), rh, g, rg)
            rhs = (math.sqrt(dimr(rh)) * sgn(rb[0] + rb[0] + rf[0] + rg[0])
                   * (-1) ** (rb[1] + rb[1] + rf[1] + rg[1])
                   * sixjr(rb, rb, rh, rg, rf, rb) * invariant(f, rf, B[0], rb, B[0], rb) * invariant(B[0], rb, BT[0], rb, g, rg))
            n_c += 1
            if abs(lhs - rhs) > 1e-10:
                bad_c += 1
                if bad_c < 5:
                    print("(c)", rf, rh, rg, lhs, rhs)
print("(c) MOTECC (48): tests", n_c, "bad", bad_c)
