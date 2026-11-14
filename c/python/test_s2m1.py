"""test_s2m1.py -- S2-M1: the C general-form engine against PythonRef.

usage: python test_s2m1.py path/to/acmat CFPTBL

Part A -- every tensor type, configuration-changing operators included.
  The C engine (acmat) gives invariants <F~ O G> between its CSFs, and
  PythonRef gives Edmonds reduced matrices between its own multiplets. The
  two bases differ by orthogonal transformations inside each (S, L) block,
  so we compare quantities that do not depend on them:
    * Tr R, Tr R^2, Tr R^3 of the reduced matrix inside one block (the odd
      powers are sign-sensitive and fix the overall sign of each operator;
      traces are used instead of eigenvalues, which these often defective,
      non-symmetric matrices only give to about sqrt(machine epsilon));
    * singular values between two blocks;
    * traces of products of three reduced matrices around three blocks
      (sensitive to any CSF-pair-dependent sign error).
  C reduced = (-1)^(sigma_T + k_T) x invariant (technical note,
  Sec. 3.1). A factor that depends only on the block labels, such as
  (-1)^(G - F), is a phase convention of the states and drops out of all
  three comparisons.

Part B -- the non-relativistic Hamiltonian with explicit closed shells.
  PythonRef builds h1 + h2 from Condon-Shortley c^k coefficients with random
  radial integrals. C builds METECC eq. (44) and eq. (45):
    h1 = - sum sqrt[2l] h(a,b) (b_a b~_b)^(1S),
    h2 = sum_k sum_abcd H^k(la lb lc ld) R^k(ab;cd) :((b_a b~_b)^(1k)(b_c b~_d)^(1k))^(1S):,
  with H^k from F(a,b,c). The two are compared through the eigenvalues of
  every (S, L) block.
"""
import itertools
import math
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np

import pyref as pr

TOL = 1e-10
failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures += 1


def run_acmat(acmat, cfptbl, shells, configs, ops):
    """ops: list of ('one', a, b, (ts, tl)) or
    ('two', a, b, c, d, (ts, tl), (ts, tl), (ts, tl), nrm), 0-based shells,
    orbital ranks given as k (not doubled)."""
    lines = ["shells " + " ".join(map(str, shells))]
    for occ in configs:
        lines.append("config " + " ".join(map(str, occ)))
    for o in ops:
        if o[0] == "one":
            _, a, b, (ts, k) = o
            lines.append(f"one {a + 1} {b + 1} {ts} {2 * k}")
        else:
            _, a, b, c, d, p, q, tot, nrm = o
            lines.append(f"two {a + 1} {b + 1} {c + 1} {d + 1} {p[0]} {2 * p[1]} "
                         f"{q[0]} {2 * q[1]} {tot[0]} {2 * tot[1]} {nrm}")
    lines.append("end")
    out = subprocess.run([acmat, cfptbl], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout
    csfs, vals = [], defaultdict(dict)
    for line in out.splitlines():
        f = line.split()
        if f[0] == "csf":
            csfs.append((int(f[3]), int(f[4]) // 2))  # (2S, L)
        elif f[0] == "v":
            vals[int(f[1])][(int(f[2]), int(f[3]))] = float(f[4])
    return csfs, vals


def c_blocks(csfs):
    blocks = defaultdict(list)
    for i, key in enumerate(csfs):
        blocks[key].append(i)
    return blocks


def c_reduced(vals, blocks, bra, ket, sign):
    rows, cols = blocks[bra], blocks[ket]
    R = np.zeros((len(rows), len(cols)))
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            R[a, b] = vals.get((i, j), 0.0)
    return sign * R


def total_rank(o):
    return o[3] if o[0] == "one" else o[7]


def py_operator(space, o):
    if o[0] == "one":
        _, a, b, (ts, k) = o
        return pr.one_body_tensor(space, a, b, ts, k)
    _, a, b, c, d, p, q, tot, nrm = o
    assert nrm == 1
    return pr.two_body_normal(space, a, b, c, d, p, q, tot)


# ======================================================================
# Part A
# ======================================================================
def part_a(acmat, cfptbl, shells, configs, ops, label, nloops=400, seed=1):
    space = pr.Space(shells, configs)
    mu = pr.Multiplets(space)
    csfs, vals = run_acmat(acmat, cfptbl, shells, configs, ops)
    blocks = c_blocks(csfs)
    same_blocks = {k: len(v) for k, v in blocks.items()} == \
                  {k: st[k].shape[1] for k, st in mu.blocks.items()}
    check(f"A[{label}] same (S, L) blocks and sizes in C and PythonRef", same_blocks,
          f"{len(csfs)} CSFs, {len(blocks)} blocks")
    if not same_blocks:
        return
    keys = sorted(blocks)
    Rc, Rp = [], []
    worst_eig = worst_sv = 0.0
    n_eig = n_sv = 0
    for idx, o in enumerate(ops):
        tsT, kT = total_rank(o)
        sign = (-1) ** ((tsT + 2 * kT) // 2)
        P = py_operator(space, o)
        rc, rp = {}, {}
        for bra in keys:
            for ket in keys:
                r = mu.reduced(P, bra, ket, tsT, kT)
                if r is None:
                    continue
                c = c_reduced(vals[idx], blocks, bra, ket, sign)
                rc[(bra, ket)], rp[(bra, ket)] = c, r
                if bra == ket:
                    d = max(abs(np.trace(np.linalg.matrix_power(c, p))
                                - np.trace(np.linalg.matrix_power(r, p))) for p in (1, 2, 3))
                    worst_eig, n_eig = max(worst_eig, d), n_eig + 1
                sc = np.linalg.svd(c, compute_uv=False)
                sp = np.linalg.svd(r, compute_uv=False)
                d = np.abs(sc - sp).max() if len(sc) else 0.0
                worst_sv, n_sv = max(worst_sv, d), n_sv + 1
        Rc.append(rc)
        Rp.append(rp)
    check(f"A[{label}] Tr R, Tr R^2, Tr R^3 inside blocks ({n_eig} block x operator)",
          worst_eig < TOL, f"max {worst_eig:.1e}")
    check(f"A[{label}] singular values between blocks ({n_sv} pairs x operator)",
          worst_sv < TOL, f"max {worst_sv:.1e}")

    rng = random.Random(seed)
    worst_loop, n_loop, big = 0.0, 0, 0.0
    tries = 0
    while n_loop < nloops and tries < 50 * nloops:
        tries += 1
        i, j, k = (rng.randrange(len(ops)) for _ in range(3))
        B, K, M = (rng.choice(keys) for _ in range(3))
        if (B, K) not in Rc[i] or (K, M) not in Rc[j] or (M, B) not in Rc[k]:
            continue
        tc = np.trace(Rc[i][(B, K)] @ Rc[j][(K, M)] @ Rc[k][(M, B)])
        tp = np.trace(Rp[i][(B, K)] @ Rp[j][(K, M)] @ Rp[k][(M, B)])
        worst_loop = max(worst_loop, abs(tc - tp))
        big = max(big, abs(tp))
        n_loop += 1
    check(f"A[{label}] traces around three blocks ({n_loop} loops)", worst_loop < TOL,
          f"max {worst_loop:.1e} (largest trace {big:.2f})")


def ops_one_body(shells, pairs, max_k=None):
    ops = []
    for a, b in pairs:
        la, lb = shells[a], shells[b]
        for ts in (0, 2):
            for k in range(abs(la - lb), la + lb + 1):
                ops.append(("one", a, b, (ts, k)))
    return ops


def ops_two_body(shells, tuples, types, rng, n):
    """random two-body operators of the given (sigma1, sigma2, Sigma) types"""
    ops = []
    while len(ops) < n:
        a, b, c, d = rng.choice(tuples)
        s1, s2, S = rng.choice(types)
        k1 = rng.randrange(abs(shells[a] - shells[b]), shells[a] + shells[b] + 1)
        k2 = rng.randrange(abs(shells[c] - shells[d]), shells[c] + shells[d] + 1)
        Ks = list(range(abs(k1 - k2), k1 + k2 + 1))
        K = rng.choice(Ks)
        if not (abs(s1 - s2) <= S <= s1 + s2):
            continue
        ops.append(("two", a, b, c, d, (2 * s1, k1), (2 * s2, k2), (2 * S, K), 1))
    return ops


# ======================================================================
# Part B
# ======================================================================
def ck(l1, m1, k, q, l2, m2):
    """<l1 m1|C^k_q|l2 m2> (Condon-Shortley)."""
    return ((-1) ** m1 * math.sqrt((2 * l1 + 1) * (2 * l2 + 1))
            * pr.threej2(2 * l1, 0, 2 * k, 0, 2 * l2, 0)
            * pr.threej2(2 * l1, -2 * m1, 2 * k, 2 * q, 2 * l2, 2 * m2))


def F_abc(a, b, c):
    """F(a,b,c) of METECC eq. (45)."""
    if (a + b + c) % 2 or not abs(a - b) <= c <= a + b:
        return 0.0
    g = (a + b + c) // 2
    f = math.factorial
    return (f(g) * math.sqrt(f(2 * g - 2 * a) * f(2 * g - 2 * b) * f(2 * g - 2 * c))
            / (math.sqrt(f(2 * g + 1)) * f(g - a) * f(g - b) * f(g - c)))


def H_k(k, l1, l2, l3, l4):
    e = (l1 - l2 + l3 - l4) // 2
    return ((-1) ** e * math.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) * (2 * l4 + 1)
                                  / (2 * k + 1)) * F_abc(k, l1, l2) * F_abc(k, l3, l4))


def part_b(acmat, cfptbl, shells, configs, label, seed=7):
    rng = random.Random(seed)
    n = len(shells)
    # random radial integrals with the symmetries of real orbitals
    R, h = {}, {}
    for a, b, c, d in itertools.product(range(n), repeat=4):
        for k in range(0, 2 * max(shells) + 1):
            if (shells[a] + shells[b] + k) % 2 or (shells[c] + shells[d] + k) % 2:
                continue
            key = min((a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
                      (c, d, a, b), (d, c, a, b), (c, d, b, a), (d, c, b, a))
            R.setdefault((k,) + key, rng.uniform(-1, 1))
    for a in range(n):
        for b in range(n):
            if shells[a] == shells[b]:
                key = (min(a, b), max(a, b))
                h.setdefault(key, rng.uniform(-1, 1))

    def Rk(k, a, b, c, d):
        key = min((a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
                  (c, d, a, b), (d, c, a, b), (c, d, b, a), (d, c, b, a))
        return R.get((k,) + key, 0.0)

    # PythonRef: h1 + 1/2 sum <ij|g|kl> a+i a+j a_l a_k
    space = pr.Space(shells, configs)
    orbs = [(k, m, t) for k, l in enumerate(shells) for m in range(-l, l + 1) for t in (1, -1)]
    terms = []
    for (ka, ma, ta) in orbs:
        for (kb, mb, tb) in orbs:
            if ta == tb and ma == mb and shells[ka] == shells[kb]:
                terms.append((h[(min(ka, kb), max(ka, kb))],
                              [(True, space.orb[(ka, ma, ta)]), (False, space.orb[(kb, mb, tb)])]))
    for i, j, kk, ll in itertools.product(orbs, repeat=4):
        if i[2] != kk[2] or j[2] != ll[2]:
            continue
        v = 0.0
        for k in range(0, 2 * max(shells) + 1):
            q = i[1] - kk[1]
            if abs(q) > k or j[1] - ll[1] != -q:
                continue
            r = Rk(k, i[0], kk[0], j[0], ll[0])
            if r == 0.0:
                continue
            v += r * (-1) ** q * ck(shells[i[0]], i[1], k, q, shells[kk[0]], kk[1]) \
                * ck(shells[j[0]], j[1], k, -q, shells[ll[0]], ll[1])
        if v != 0.0:
            terms.append((0.5 * v, [(True, space.orb[i]), (True, space.orb[j]),
                                    (False, space.orb[ll]), (False, space.orb[kk])]))
    Hpy = space.matrix(terms)
    mu = pr.Multiplets(space)

    # C: METECC (44) and (45)
    ops, weights = [], []
    for a in range(n):
        for b in range(n):
            if shells[a] == shells[b]:
                ops.append(("one", a, b, (0, 0)))
                weights.append(-math.sqrt(2 * (2 * shells[a] + 1)) * h[(min(a, b), max(a, b))])
    for a, b, c, d in itertools.product(range(n), repeat=4):
        la, lb, lc, ld = (shells[x] for x in (a, b, c, d))
        for k in range(0, 2 * max(shells) + 1):
            w = H_k(k, la, lb, lc, ld) * Rk(k, a, b, c, d)
            if w == 0.0:
                continue
            ops.append(("two", a, b, c, d, (0, k), (0, k), (0, 0), 1))
            weights.append(w)
    csfs, vals = run_acmat(acmat, cfptbl, shells, configs, ops)
    blocks = c_blocks(csfs)
    worst, nb = 0.0, 0
    for key, st in mu.blocks.items():
        top = st[key]
        Ep = np.sort(np.linalg.eigvalsh(top.T @ Hpy @ top))
        idx = blocks.get(key, [])
        Hc = np.zeros((len(idx), len(idx)))
        for w, opi in zip(weights, range(len(ops))):
            for a2, i in enumerate(idx):
                for b2, j in enumerate(idx):
                    Hc[a2, b2] += w * vals[opi].get((i, j), 0.0)
        tS, L = key
        Hc /= math.sqrt((tS + 1) * (2 * L + 1))  # scalar: <jm|T|jm> = <j||T||j>/sqrt[j]
        Ec = np.sort(np.linalg.eigvals(Hc).real)
        if len(Ec) != len(Ep):
            worst = float("inf")
            break
        worst = max(worst, np.abs(Ec - Ep).max())
        nb += 1
    check(f"B[{label}] h1 + h2 spectra per (S, L) block, METECC (44)+(45) vs c^k",
          worst < TOL, f"{nb} blocks, {len(ops)} operators, max {worst:.1e}")


def main():
    acmat, cfptbl = sys.argv[1], sys.argv[2]
    rng = random.Random(3)

    # p^2 d, p^3, p d^2: one-body operators of every rank, inside and across shells
    shells, configs = [1, 2], [(2, 1), (3, 0), (1, 2)]
    ops = ops_one_body(shells, [(0, 0), (1, 1), (0, 1), (1, 0)])
    part_a(acmat, cfptbl, shells, configs, ops, "one-body, p2d+p3+pd2")

    # two-body: Coulomb-like (00)0, SO2-like (10)1 and (01)1, SS-like (11)2,
    # and the other couplings (11)0, (11)1
    tuples = list(itertools.product(range(2), repeat=4))
    ops = ops_two_body(shells, tuples,
                       [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 2), (1, 1, 0), (1, 1, 1)],
                       rng, 60)
    part_a(acmat, cfptbl, shells, configs, ops, "two-body, p2d+p3+pd2")

    # three open shells, one of them s
    shells3, configs3 = [0, 1, 2], [(1, 2, 1), (2, 1, 1), (1, 1, 2)]
    ops = ops_one_body(shells3, [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)])
    ops += ops_two_body(shells3, list(itertools.product(range(3), repeat=4)),
                        [(1, 0, 1), (0, 1, 1), (1, 1, 2)], rng, 40)
    part_a(acmat, cfptbl, shells3, configs3, ops, "s p d, three configurations")

    # explicit closed shells
    part_b(acmat, cfptbl, [0, 1], [(2, 2), (1, 3), (0, 4)], "s2p2+sp3+p4, closed s explicit")
    part_b(acmat, cfptbl, [0, 0, 1], [(2, 2, 1), (2, 1, 2)], "1s2 2s2 2p + 1s2 2s 2p2")

    print("S2-M1", "FAILED" if failures else "passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
