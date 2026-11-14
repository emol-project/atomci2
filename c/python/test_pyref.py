"""test_pyref.py -- self-tests of pyref.py and the conventions they pin down.

Run with a Python that has numpy:
    python c/python/test_pyref.py
Prints each check and the constants of the technical note, Sec. 3.1;
exits 1 if a check fails.
"""
import math
import sys

import numpy as np

import pyref as pr

TOL = 1e-12
failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures += 1


def all_ranks(ts_max=2, k_max=None):
    for ts in range(0, ts_max + 1, 2):
        for k in range(0, k_max + 1):
            yield ts, k


# ---- T1: anticommutator (METECC eq. 15) on one p shell, every rank --------
sp = pr.Space([1], [(n,) for n in range(7)])
ident = np.eye(sp.dim)
worst = 0.0
for ts, k in all_ranks(2, 2):
    for tnu in range(-ts, ts + 1, 2):
        for q in range(-k, k + 1):
            # (b~ b)^(x) and (b b~)^(x), component (nu, q)
            tb, bb = [], []
            for (m1, t1) in pr._components(sp, 0):
                for (m2, t2) in pr._components(sp, 0):
                    c = pr.cg2(2, 2 * m1, 2, 2 * m2, 2 * k, 2 * q) * pr.cg2(1, t1, 1, t2, ts, tnu)
                    if c == 0.0:
                        continue
                    (ca, oa), = sp.bt(0, m1, t1)
                    (cb, ob), = sp.b(0, m2, t2)
                    tb.append((c * ca * cb, [oa, ob]))
                    (cb, ob), = sp.b(0, m1, t1)
                    (ca, oa), = sp.bt(0, m2, t2)
                    bb.append((c * cb * ca, [ob, oa]))
            # (-1)^(2b - x) with 2b = 2(1/2 + l) = 3 and x = sigma + k
            sign = (-1) ** ((2 * 3 - ts - 2 * k) // 2)
            lhs = sp.matrix(tb) + sign * sp.matrix(bb)
            rhs = math.sqrt(6) * ident if (ts, k) == (0, 0) else 0 * ident
            worst = max(worst, np.abs(lhs - rhs).max())
check("T1 anticommutator (b~ b)^x + (-1)^(2b-x) (b b~)^x = d_x0 sqrt[b], p shell", worst < TOL,
      f"max {worst:.1e}")

# ---- T2: multiplets of p^2 and d^2 ---------------------------------------
for l, expect in ((1, {(2, 1), (0, 2), (0, 0)}), (2, {(2, 3), (2, 1), (0, 4), (0, 2), (0, 0)})):
    mu = pr.Multiplets(pr.Space([l], [(2,)]))
    got = {key for key in mu.blocks}
    check(f"T2 terms of l={l}^2", got == expect, f"{sorted(got)}")

# ---- T3: reduced matrix elements of S and L themselves -------------------
space = pr.Space([1, 2], [(2, 1)])  # p^2 d
mu = pr.Multiplets(space)
S_op = {(2 * q, 0): mu.S[q] for q in (-1, 0, 1)}
L_op = {(0, q): mu.L[q] for q in (-1, 0, 1)}
worst = 0.0
for key, st in mu.blocks.items():
    tS, L = key
    S, n = tS / 2, st[key].shape[1]
    rS = mu.reduced(S_op, key, key, 2, 0)
    rL = mu.reduced(L_op, key, key, 0, 1)
    eS = math.sqrt(S * (S + 1) * (2 * S + 1)) * math.sqrt(2 * L + 1)
    eL = math.sqrt(L * (L + 1) * (2 * L + 1)) * math.sqrt(2 * S + 1)
    if rS is not None:
        worst = max(worst, np.abs(rS - eS * np.eye(n)).max())
    if rL is not None:
        worst = max(worst, np.abs(rL - eL * np.eye(n)).max())
check("T3 <SL||S||SL> = sqrt(S(S+1)[S][L]), <SL||L||SL> = sqrt(L(L+1)[L][S]), p^2 d",
      worst < TOL, f"max {worst:.1e}")

# ---- T4: sum_k w_k (b_k b~_k)^(1,0) and ^(0,1) against S and L -----------
opS, opL = {}, {}
for k, l in enumerate(space.shells):
    tS = pr.one_body_tensor(space, k, k, 2, 0)
    tL = pr.one_body_tensor(space, k, k, 0, 1)
    for c, v in tS.items():
        opS[c] = opS.get(c, 0) + math.sqrt(2 * l + 1) * v
    for c, v in tL.items():
        opL[c] = opL.get(c, 0) + math.sqrt(l * (l + 1) * (2 * l + 1)) * v
# component-wise ratio op = c * S
num = sum(np.sum(opS[(2 * q, 0)] * mu.S[q]) for q in (-1, 0, 1))
den = sum(np.sum(mu.S[q] * mu.S[q]) for q in (-1, 0, 1))
c_S = num / den
res_S = max(np.abs(opS[(2 * q, 0)] - c_S * mu.S[q]).max() for q in (-1, 0, 1))
num = sum(np.sum(opL[(0, q)] * mu.L[q]) for q in (-1, 0, 1))
den = sum(np.sum(mu.L[q] * mu.L[q]) for q in (-1, 0, 1))
c_L = num / den
res_L = max(np.abs(opL[(0, q)] - c_L * mu.L[q]).max() for q in (-1, 0, 1))
check("T4 sum_k sqrt[l_k] (b b~)^(1,0) = c_S S", res_S < TOL,
      f"c_S = {c_S:.15f} (-sqrt2 = {-math.sqrt(2):.15f}), residual {res_S:.1e}")
check("T4 sum_k sqrt(l(l+1)[l]) (b b~)^(0,1) = c_L L", res_L < TOL,
      f"c_L = {c_L:.15f} (-sqrt(3/2) = {-math.sqrt(1.5):.15f}), residual {res_L:.1e}")
check("T4 constants c_S = -sqrt2, c_L = -sqrt(3/2)",
      abs(c_S + math.sqrt(2)) < TOL and abs(c_L + math.sqrt(1.5)) < TOL)
check("T4 C invariant / Edmonds reduced = -1 (S and L sums, diagonal)",
      abs(math.sqrt(2) / c_S + 1) < TOL and abs(math.sqrt(1.5) / c_L + 1) < TOL)
# The C engine gives invariant = sqrt2 * sqrt(S(S+1)[S][L]) for the same S sum
# and sqrt(3/2) * sqrt(L(L+1)[S][L]) for the L sum (c/tests/probe_rank.c).
print(f"     => C invariant / Edmonds reduced: S sum {math.sqrt(2) / c_S:.15f}, "
      f"L sum {math.sqrt(1.5) / c_L:.15f}")

# ---- T5: one-electron spin-orbit l.s against the tensor form -------------
# l.s = sum_q (-1)^q l_q s_-q, built from one-electron matrices
def ls_operator(space):
    H = np.zeros((space.dim, space.dim))
    for k, l in enumerate(space.shells):
        for (m1, t1) in pr._components(space, k):
            for (m2, t2) in pr._components(space, k):
                v = 0.0
                for q in (-1, 0, 1):
                    # <l m1|l_q|l m2> = <l m2 1 q|l m1> sqrt(l(l+1))  (Edmonds)
                    lq = pr.cg2(2 * l, 2 * m2, 2, 2 * q, 2 * l, 2 * m1) * math.sqrt(l * (l + 1))
                    sq = pr.cg2(1, t2, 2, -2 * q, 1, t1) * math.sqrt(0.75)
                    v += (-1) ** q * lq * sq
                if v != 0.0:
                    H += space.matrix([(v, [(True, space.orb[(k, m1, t1)]),
                                            (False, space.orb[(k, m2, t2)])])])
    return H


def so_tensor(space, k):
    """J-scalar of (b_k b~_k)^(1,1): sum_{nu q} <1 nu 1 q|0 0> W_(nu q),
    spin rank coupled first."""
    W = pr.one_body_tensor(space, k, k, 2, 1)
    X = 0.0
    for nu in (-1, 0, 1):
        X = X + pr.cg2(2, 2 * nu, 2, -2 * nu, 0, 0) * W[(2 * nu, -nu)]
    return X


space5 = pr.Space([1, 2], [(2, 1), (1, 2)])
H = ls_operator(space5)
X = [so_tensor(space5, k) for k in range(2)]
# fit H = sum_k c_k X_k shell by shell (they act on different shells)
coefs = []
for k, l in enumerate(space5.shells):
    Hk = ls_operator(pr.Space([l], [(1,)]))
    Xk = so_tensor(pr.Space([l], [(1,)]), 0)
    c = np.sum(Hk * Xk) / np.sum(Xk * Xk)
    coefs.append(c)
resid = np.abs(H - sum(c * x for c, x in zip(coefs, X))).max()
check("T5 sum_i l_i.s_i = sum_k c_l J-scalar[(b_k b~_k)^(1,1)]", resid < TOL,
      f"residual {resid:.1e}")
worst = 0.0
for l in range(1, 5):
    Hk = ls_operator(pr.Space([l], [(1,)]))
    Xk = so_tensor(pr.Space([l], [(1,)]), 0)
    c = np.sum(Hk * Xk) / np.sum(Xk * Xk)
    expect = math.sqrt(l * (l + 1) * (2 * l + 1) / 2)
    worst = max(worst, abs(c - expect), np.abs(Hk - c * Xk).max())
    print(f"     l={l}: c_l = {c:.15f}   sqrt(l(l+1)(2l+1)/2) = {expect:.15f}")
check("T5 c_l = sqrt(l(l+1)(2l+1)/2) for l = 1..4", worst < TOL, f"max {worst:.1e}")

# ---- T6: LSJ matrix elements of the J-scalar from the double-tensor reduced
#          elements, all in pyref's own basis (signs included) ------------
def sixj2(a, b, c, d, e, f):
    """6j {a b c; d e f} from Racah's formula, doubled arguments."""
    def tri(x, y, z):
        return (x + y + z) % 2 == 0 and abs(x - y) <= z <= x + y

    if not (tri(a, b, c) and tri(a, e, f) and tri(d, b, f) and tri(d, e, c)):
        return 0.0

    def delta(x, y, z):
        return math.sqrt(pr._fact((x + y - z) // 2) * pr._fact((x - y + z) // 2)
                         * pr._fact((-x + y + z) // 2) / pr._fact((x + y + z) // 2 + 1))

    al = [(a + b + c) // 2, (a + e + f) // 2, (d + b + f) // 2, (d + e + c) // 2]
    be = [(a + b + d + e) // 2, (b + c + e + f) // 2, (c + a + f + d) // 2]
    s = 0.0
    for t in range(max(al), min(be) + 1):
        den = 1.0
        for x in al:
            den *= pr._fact(t - x)
        for x in be:
            den *= pr._fact(x - t)
        s += (-1) ** t * pr._fact(t + 1) / den
    return s * delta(a, b, c) * delta(a, e, f) * delta(d, b, f) * delta(d, e, c)


def lsj_states(mu, key, tJ):
    """|alpha S L J, M = J> (columns) of block key = (2S, L), spin first."""
    tS, L = key
    st = mu.blocks[key]
    vec = None
    for (tms, ml), v in st.items():
        if tms + 2 * ml != tJ:
            continue
        c = pr.cg2(tS, tms, 2 * L, 2 * ml, tJ, tJ)
        if c != 0.0:
            vec = c * v if vec is None else vec + c * v
    return vec


space6 = pr.Space([1, 2], [(2, 1)])  # p^2 d: many (S, L) and J
mu6 = pr.Multiplets(space6)
kap = 1
H6 = ls_operator(space6)
W6 = [pr.one_body_tensor(space6, k, k, 2, 1) for k in range(2)]
cl = [math.sqrt(l * (l + 1) * (2 * l + 1) / 2) for l in space6.shells]
worst, nblk = 0.0, 0
for bra in mu6.blocks:
    for ket in mu6.blocks:
        red = None
        for k in range(2):
            r = mu6.reduced(W6[k], bra, ket, 2, 1)
            if r is not None:
                red = cl[k] * r if red is None else red + cl[k] * r
        (tS, L), (tSp, Lp) = bra, ket
        for tJ in range(abs(tS - 2 * L) if tS < 2 * L else tS - 2 * L, tS + 2 * L + 1, 2):
            if not (abs(tSp - 2 * Lp) <= tJ <= tSp + 2 * Lp):
                continue
            vb, vk = lsj_states(mu6, bra, tJ), lsj_states(mu6, ket, tJ)
            if vb is None or vk is None:
                continue
            brute = vb.T @ H6 @ vk
            ph = (-1) ** ((tSp + 2 * L + tJ) // 2) * (-1) ** kap / math.sqrt(2 * kap + 1)
            six = sixj2(tJ, 2 * L, tS, 2 * kap, tSp, 2 * Lp)
            pred = ph * six * red if red is not None else 0 * brute
            worst = max(worst, np.abs(brute - pred).max())
            nblk += 1
check("T6 <aSLJ|sum l.s|a'S'L'J> = (-1)^k/sqrt[k] (-1)^(S'+L+J) {J L S; k S' L'} <aSL||W||a'S'L'>",
      worst < TOL, f"{nblk} (bra, ket, J) blocks, max {worst:.1e}")

print("pyref self-tests", "FAILED" if failures else "passed")
sys.exit(1 if failures else 0)
