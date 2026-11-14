"""S2-M4: Coulomb H + one-electron spin-orbit H_SO1 in the LSJ basis.

usage: python test_s2m4.py path/to/acso CFPTBL

V1 -- against PythonRef. Random h, R^k (with the symmetries of real
  orbitals) and zeta(a,b) (same l, symmetric, cross-shell included). PythonRef
  builds H = h1 + h2 (Condon-Shortley c^k) + sum_ab zeta(ab) sum l.s a+_a a_b
  on all determinants of the configurations and diagonalizes it. The C side
  (acso) builds the LSJ CI from the general-form engine. Every C level of 2J
  must appear 2J+1 times in the determinant spectrum, and the two spectra
  must be equal as multisets.

V4 -- analytic.
  (a) jj limit: zeta only, one shell l^n. The single-electron levels are
      j = l+1/2 at zeta l/2 and j = l-1/2 at -zeta(l+1)/2, so a level with
      n+ electrons in j+ has energy n+ zeta l/2 - n- zeta(l+1)/2 and
      degeneracy C(2l+2, n+) C(2l, n-).
  (b) Lande: the diagonal LSJ element of the Hund term (largest S, then L)
      of l^n is (A/2)[J(J+1) - L(L+1) - S(S+1)] with A = zeta/(2S) below
      half filling and -zeta/(2S) above. This fixes the overall sign.
"""
import itertools
import math
import random
import subprocess
import sys
from collections import Counter

import numpy as np

import pyref as pr
from test_s2m1 import ck

TOL = 1e-10
failures = 0


def check(name, ok, detail=""):
    global failures
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        failures += 1


ALPHA2 = 1.0 / 137.035999084 ** 2
GS = 2.00231930436256


def run_acso(acso, cfptbl, shells, configs, h, R, z, t=None, u=None, gout=None):
    lines = ["orbitals " + " ".join(map(str, shells))]
    lines += ["config " + " ".join(map(str, c)) for c in configs]
    lines += [f"h {a + 1} {b + 1} {v!r}" for (a, b), v in h.items()]
    lines += [f"r {k} {a + 1} {b + 1} {c + 1} {d + 1} {v!r}" for (k, a, b, c, d), v in R.items()]
    lines += [f"z {a + 1} {b + 1} {v!r}" for (a, b), v in z.items()]
    lines += [f"t {a + 1} {b + 1} {v!r}" for (a, b), v in (t or {}).items()]
    lines += [f"u {a + 1} {b + 1} {v!r}" for (a, b), v in (u or {}).items()]
    lines.append("end")
    out = subprocess.run([acso, cfptbl], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout
    csfs, levels, diag, asym = {}, [], {}, 0.0
    for line in out.splitlines():
        w = line.split()
        if w[0] == "csf":
            csfs[int(w[1])] = (int(w[2]), int(w[3]))
        elif w[0] == "level":
            levels.append((int(w[1]), float(w[2])))
        elif w[0] == "diag":
            diag[(int(w[1]), int(w[2]))] = float(w[3])
        elif w[0] == "asym":
            asym = max(asym, float(w[2]))
        elif w[0] == "g" and gout is not None:
            gout.append((int(w[1]),) + tuple(map(float, w[2:])))
    return csfs, levels, diag, asym


def random_integrals(shells, rng, with_coulomb=True):
    n = len(shells)
    h, R, z = {}, {}, {}
    if with_coulomb:
        Rsym = {}
        for a, b, c, d in itertools.product(range(n), repeat=4):
            for k in range(0, 2 * max(shells) + 1):
                if (shells[a] + shells[b] + k) % 2 or (shells[c] + shells[d] + k) % 2:
                    continue
                if not (abs(shells[a] - shells[b]) <= k <= shells[a] + shells[b]
                        and abs(shells[c] - shells[d]) <= k <= shells[c] + shells[d]):
                    continue
                key = min((a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
                          (c, d, a, b), (d, c, a, b), (c, d, b, a), (d, c, b, a))
                R[(k, a, b, c, d)] = Rsym.setdefault((k,) + key, rng.uniform(-1, 1))
        for a in range(n):
            for b in range(a, n):
                if shells[a] == shells[b]:
                    h[(a, b)] = rng.uniform(-1, 1)
    for a in range(n):
        for b in range(a, n):
            if shells[a] == shells[b] and shells[a] > 0:
                z[(a, b)] = rng.uniform(-1, 1)
    return h, R, z


def py_hamiltonian(shells, configs, h, R, z):
    space = pr.Space(shells, configs)
    orbs = [(k, m, t) for k, l in enumerate(shells) for m in range(-l, l + 1) for t in (1, -1)]
    terms = []
    for (ka, ma, ta) in orbs:
        for (kb, mb, tb) in orbs:
            if ta == tb and ma == mb and shells[ka] == shells[kb]:
                v = h.get((min(ka, kb), max(ka, kb)), 0.0)
                terms.append((v, [(True, space.orb[(ka, ma, ta)]), (False, space.orb[(kb, mb, tb)])]))
    for i, j, kk, ll in itertools.product(orbs, repeat=4):
        if i[2] != kk[2] or j[2] != ll[2]:
            continue
        v = 0.0
        for k in range(0, 2 * max(shells) + 1):
            q = i[1] - kk[1]
            if abs(q) > k or j[1] - ll[1] != -q:
                continue
            r = R.get((k, i[0], kk[0], j[0], ll[0]), 0.0)
            if r == 0.0:
                continue
            v += r * (-1) ** q * ck(shells[i[0]], i[1], k, q, shells[kk[0]], kk[1]) \
                * ck(shells[j[0]], j[1], k, -q, shells[ll[0]], ll[1])
        if v != 0.0:
            terms.append((0.5 * v, [(True, space.orb[i]), (True, space.orb[j]),
                                    (False, space.orb[ll]), (False, space.orb[kk])]))
    # spin-orbit: zeta(ab) sum_q (-1)^q <a m1|l_q|b m2> <t1|s_-q|t2> a+ b
    for (ka, ma, ta) in orbs:
        for (kb, mb, tb) in orbs:
            l = shells[ka]
            if shells[kb] != l or l == 0:
                continue
            zz = z.get((min(ka, kb), max(ka, kb)), 0.0)
            if zz == 0.0:
                continue
            v = 0.0
            for q in (-1, 0, 1):
                lq = pr.cg2(2 * l, 2 * mb, 2, 2 * q, 2 * l, 2 * ma) * math.sqrt(l * (l + 1))
                sq = pr.cg2(1, tb, 2, -2 * q, 1, ta) * math.sqrt(0.75)
                v += (-1) ** q * lq * sq
            if v != 0.0:
                terms.append((zz * v, [(True, space.orb[(ka, ma, ta)]),
                                       (False, space.orb[(kb, mb, tb)])]))
    return space.matrix(terms)


def py_zeeman(shells, configs, t, u):
    """z components, on the determinants, of L, S and the operators of the
    RMC and gauge corrections:
      -2 alpha^2 sum_i T_i s_z,i,
      (alpha^2/2) sum_i Z r_i^-1 [s_z,i - (s_i . r^_i) z^_i].
    The angular factor of the gauge term uses <l m|C^1_p C^1_0|l m'> summed
    over the intermediate l'' = l +- 1 (closure, exact)."""
    space = pr.Space(shells, configs)
    orbs = [(k, m, s) for k, l in enumerate(shells) for m in range(-l, l + 1) for s in (1, -1)]

    def sq(q, t1, t2):  # <1/2 t1/2|s_q|1/2 t2/2>
        return pr.cg2(1, t2, 2, 2 * q, 1, t1) * math.sqrt(0.75)

    def a2(l, m, p, m2):  # <l m|C^1_p C^1_0|l m2>
        v = 0.0
        for l3 in (l - 1, l + 1):
            if l3 < 0:
                continue
            for m3 in range(-l3, l3 + 1):
                v += ck(l, m, 1, p, l3, m3) * ck(l3, m3, 1, 0, l, m2)
        return v

    Lz, Sz, Rm, Gc, Ro = [], [], [], [], []
    for (ka, ma, ta) in orbs:
        for (kb, mb, tb) in orbs:
            l = shells[ka]
            if shells[kb] != l:
                continue
            op = [(True, space.orb[(ka, ma, ta)]), (False, space.orb[(kb, mb, tb)])]
            same = ka == kb
            if same and ma == mb and ta == tb:
                Lz.append((float(ma), op))
                Sz.append((ta / 2, op))
            key = (min(ka, kb), max(ka, kb))
            if ma == mb and ta == tb:
                Rm.append((-2 * ALPHA2 * t.get(key, 0.0) * ta / 2, op))
                Ro.append((-ALPHA2 * t.get(key, 0.0) * ma, op))
            ang = (sq(0, ta, tb) if ma == mb else 0.0) - sum(
                (-1) ** q * sq(q, ta, tb) * a2(l, ma, -q, mb) for q in (-1, 0, 1))
            if ang != 0.0:
                Gc.append((0.5 * ALPHA2 * u.get(key, 0.0) * ang, op))
    return [space.matrix(x) for x in (Lz, Sz, Rm, Gc, Ro)], space


def v1_g(acso, cfptbl, shells, configs, label, seed):
    """Zeeman factors of every level with J > 0 against PythonRef at M = J."""
    rng = random.Random(seed)
    h, R, z = random_integrals(shells, rng)
    t = {k: rng.uniform(0.5, 3) for k in h}
    u = {k: rng.uniform(0.5, 3) for k in h}
    H = py_hamiltonian(shells, configs, h, R, z)
    (Lz, Sz, Rm, Gc, Ro), space = py_zeeman(shells, configs, t, u)
    Mtot = np.array([sum(m / 1 + s / 2 for (k, m, s), bit in space.orb.items() if d >> bit & 1)
                     for d in space.dets])
    gs = []
    run_acso(acso, cfptbl, shells, configs, h, R, z, t, u, gs)
    worst, n = 0.0, 0
    for tJ, E, gL, gS, rmc, gc, ormc in gs:
        J = tJ / 2
        sel = np.where(np.abs(Mtot - J) < 1e-9)[0]
        w, V = np.linalg.eigh(H[np.ix_(sel, sel)])
        hit = np.where(np.abs(w - E) < 1e-8)[0]
        if len(hit) != 1:
            worst = float("inf")
            break
        v = V[:, hit[0]]
        want = [v @ X[np.ix_(sel, sel)] @ v / J for X in (Lz, Sz, Rm, Gc, Ro)]
        got = [gL, gS, rmc, gc, ormc]
        scale = [1, 1, ALPHA2, ALPHA2, ALPHA2]
        worst = max(worst, max(abs(a - b) / s for a, b, s in zip(got, want, scale)))
        n += 1
    check(f"V1g[{label}] g_L, g_S, Delta g_RMC, ORMC, GC of every J > 0 level vs determinants",
          worst < 1e-9, f"{n} levels, max {worst:.1e} (RMC, GC in units of alpha^2)")


def v4_g(acso, cfptbl, shells, configs, label, seed):
    """With S and L good per CSF, g_S and g_L of a level are the Lande
    values averaged with the squared LSJ coefficients. Compared here level
    by level for the pure Hund terms of the jj-free case (zeta = 0)."""
    rng = random.Random(seed)
    h, R, _ = random_integrals(shells, rng)
    gs = []
    csfs, _, _, _ = run_acso(acso, cfptbl, shells, configs, h, R, {}, gout=gs)
    worst = 0.0
    # without spin-orbit every level is a pure term; find it by energy among
    # the (S, L) blocks: the Lande values then hold exactly
    for tJ, E, gL, gS, _, _, _ in gs:
        J = tJ / 2
        cand = []
        for tS, tL in set(csfs.values()):
            S, L = tS / 2, tL / 2
            if not abs(S - L) <= J <= S + L:
                continue
            x = (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
            cand.append((abs(gS - x) + abs(gL - (1 - x)), x))
        worst = max(worst, min(cand)[0])
    check(f"V4g[{label}] g_S, g_L = Lande values of pure LS levels (zeta = 0)",
          worst < 1e-12, f"{len(gs)} levels, max {worst:.1e}")


def v4_dirac_g(acso, cfptbl):
    """Hydrogen-like ions: rmc + ormc + gc must be the alpha^2 term of the
    Dirac g factor g = kappa (2 kappa E/mc^2 - 1) / (2 j (j+1)), i.e.
      1s1/2  -(2/3) (Z alpha)^2,  2p1/2  -(Z alpha)^2/6,  2p3/2  -(2/15) (Z alpha)^2,
    with the hydrogenic <T> = Z^2/(2n^2) and Z<1/r> = Z^2/n^2; to order
    alpha^2, E/mc^2 = 1 - (Z alpha)^2/(2n^2), so the alpha^2 term is
    2 kappa^2 (-(Z alpha)^2/(2n^2)) / (2 j (j+1)), for every j of each n, l."""
    worst = 0.0
    for n, l in ((1, 0), (2, 1), (3, 2), (4, 3)):
        gs = []
        run_acso(acso, cfptbl, [l], [(1,)], {}, {}, {}, {(0, 0): 1 / (2 * n * n)},
                 {(0, 0): 1 / (n * n)}, gs)
        for tJ, _, gL, gS, rmc, gc, ormc in gs:
            j = tJ / 2
            kappa = -(j + 0.5) if j == l + 0.5 else (j + 0.5)
            # alpha^2 coefficient of g: E/mc^2 = 1 - (Za)^2/(2n^2) + ...
            dg = kappa * 2 * kappa * (-1 / (2 * n * n)) / (2 * j * (j + 1)) * ALPHA2
            worst = max(worst, abs(rmc + gc + ormc - dg) / ALPHA2)
    check("V4d hydrogen-like: Delta g_RMC + Delta g_ORMC + Delta g_GC = alpha^2 term of "
          "the Dirac g (1s, 2p, 3d, 4f; Z = 1)", worst < 1e-12, f"max {worst:.1e} alpha^2")


def multiset_diff(a, b):
    """max |a_i - b_i| of two sorted lists, inf if lengths differ."""
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(sorted(a), sorted(b))), default=0.0)


def v1(acso, cfptbl, shells, configs, label, seed):
    rng = random.Random(seed)
    h, R, z = random_integrals(shells, rng)
    H = py_hamiltonian(shells, configs, h, R, z)
    asymp = np.abs(H - H.T).max()
    Ep = np.linalg.eigvalsh(0.5 * (H + H.T))
    csfs, levels, _, asym = run_acso(acso, cfptbl, shells, configs, h, R, z)
    Ec = [e for tJ, e in levels for _ in range(tJ + 1)]
    d = multiset_diff(Ec, Ep)
    check(f"V1[{label}] spectrum of Coulomb + H_SO1, LSJ CI vs determinants",
          d < TOL and asym < TOL and asymp < TOL,
          f"{len(csfs)} CSFs, {len(levels)} levels, {len(Ep)} determinants, max {d:.1e},"
          f" C asym {asym:.1e}")


def v4_jj(acso, cfptbl, l, n):
    zeta = 0.37
    _, levels, _, asym = run_acso(acso, cfptbl, [l], [(n,)], {}, {}, {(0, 0): zeta})
    got = Counter()
    for tJ, e in levels:
        got[round(e / zeta * 4)] += tJ + 1  # energies are multiples of zeta/2
    want = Counter()
    for npl in range(0, min(n, 2 * l + 2) + 1):
        nmi = n - npl
        if nmi < 0 or nmi > 2 * l:
            continue
        e2 = npl * l - nmi * (l + 1)  # energy in units of zeta/2
        want[2 * e2] += math.comb(2 * l + 2, npl) * math.comb(2 * l, nmi)
    exact = all(abs(e / zeta * 4 - round(e / zeta * 4)) < 1e-9 for _, e in levels)
    return got == want and exact and asym < TOL


def hund_term(l, n):
    """2S, L of the Hund term of l^n, and whether the shell is more than
    half filled (electrons, or holes, fill m = l, l-1, ... spin-parallel)."""
    holes = n > 2 * l + 1
    ne = 4 * l + 2 - n if holes else n
    ms = list(range(l, -l - 1, -1))
    return ne, sum(ms[:ne]), holes  # 2S, L, more than half filled


def v4_lande(acso, cfptbl, l, n):
    zeta = 0.29
    csfs, _, diag, _ = run_acso(acso, cfptbl, [l], [(n,)], {}, {}, {(0, 0): zeta})
    tS, L, holes = hund_term(l, n)
    idx = [i for i, (s, ll) in csfs.items() if s == tS and ll == 2 * L]
    if len(idx) != 1:
        return float("inf")
    S = tS / 2
    A = (-1 if holes else 1) * zeta / (2 * S) if tS else 0.0
    worst = 0.0
    for tJ in range(abs(2 * L - tS), 2 * L + tS + 1, 2):
        J = tJ / 2
        want = A / 2 * (J * (J + 1) - L * (L + 1) - S * (S + 1))
        worst = max(worst, abs(diag[(tJ, idx[0])] - want))
    return worst


def main():
    acso, cfptbl = sys.argv[1], sys.argv[2]

    v1(acso, cfptbl, [1], [(2,)], "p2", 1)
    v1(acso, cfptbl, [1], [(3,)], "p3", 2)
    v1(acso, cfptbl, [2], [(3,)], "d3", 3)
    v1(acso, cfptbl, [1, 2], [(2, 1), (3, 0), (1, 2)], "p2d+p3+pd2", 4)
    v1(acso, cfptbl, [0, 1, 1], [(2, 2, 0), (2, 1, 1), (2, 0, 2), (1, 2, 1)],
       "s2p2+s2pp'+s2p'2+sp2p', closed s, zeta(p,p')", 5)
    v1(acso, cfptbl, [0, 1, 2], [(2, 1, 1), (1, 2, 1), (2, 0, 2)], "s p d mixed", 6)

    bad = []
    for l, nmax in ((1, 5), (2, 9), (3, 3)):
        for n in range(1, nmax + 1):
            if not v4_jj(acso, cfptbl, l, n):
                bad.append(f"l={l} n={n}")
    check("V4a jj limit: levels and degeneracies of zeta l.s on l^n (p1-5, d1-9, f1-3)",
          not bad, ", ".join(bad) or "all equal")

    worst, cases = 0.0, 0
    for l, nmax in ((1, 5), (2, 9), (3, 3)):
        for n in range(1, nmax + 1):
            worst = max(worst, v4_lande(acso, cfptbl, l, n))
            cases += 1
    check("V4b Lande: Hund-term diagonal (A/2)[J(J+1)-L(L+1)-S(S+1)], A = +-zeta/(2S)",
          worst < TOL, f"{cases} configurations, max {worst:.1e}")

    v1_g(acso, cfptbl, [1], [(2,)], "p2", 11)
    v1_g(acso, cfptbl, [2], [(3,)], "d3", 12)
    v1_g(acso, cfptbl, [1, 2], [(2, 1), (1, 2)], "p2d+pd2", 13)
    v1_g(acso, cfptbl, [0, 1, 1], [(2, 2, 0), (2, 1, 1), (1, 2, 1)],
         "closed s, p and p', cross-shell T and Z/r", 14)
    v1_g(acso, cfptbl, [3], [(2,)], "f2", 15)

    v4_g(acso, cfptbl, [1, 2], [(2, 1), (1, 2)], "p2d+pd2", 21)
    v4_dirac_g(acso, cfptbl)

    print("S2-M4", "FAILED" if failures else "passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
