# bp2e: the two-electron Breit-Pauli coefficients of the technical note, Sec. 4.2,
# against the operators built in coordinate space (theory_coord.py), in the
# two-electron determinant space of shells s, p, p', d (Gaussian radial functions).
#   SO2  the Theta of H_SO2 (spin-same-orbit eta = 1 and spin-other-orbit eta = 2)
#   SS   the Theta of H_SS
#   OO   H_OO (current and divergence terms)
# alpha^2 = 1 throughout.
import itertools
import math
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import theory_coord as tc  # noqa: E402
from theory_wig import ninej2, sixj2  # noqa: E402

which = sys.argv[1:] or ["so2", "ss", "oo", "bp1"]
NEG = "neg" in which          # control: break one relative sign per operator; every comparison must then fail


def neg(x):
    return (-1) ** x if NEG else 1
SH = [(0, 1.1), (1, 0.9), (1, 0.5), (2, 0.7)]     # (l, zeta) of each shell
g = tc.Grid(nr=400, rmax=8.0, nth=12, nph=24)
shells = [l for l, _ in SH]
nmax = [2 * (2 * l + 1) for l in shells]
configs = [occ for occ in itertools.product(*[range(n + 1) for n in nmax]) if sum(occ) == 2]
space = tc.Space(shells, configs)
orbs = [(s, m) for s, (l, _) in enumerate(SH) for m in range(-l, l + 1)]
no = len(orbs)
t0 = time.time()
O = [tc.Orb(g, SH[s][0], m, SH[s][1]) for (s, m) in orbs]
RAD = [tc.radial(l, z) for l, z in SH]
print(f"dets {space.dim}, orbitals {no}, grid {g.nr}x{g.nang}, orbitals built in {time.time() - t0:.0f} s")
lmax = max(shells)


def lof(o):
    return SH[orbs[o][0]][0]


def C(l, k, lp):
    return tc.rmeC(l, k, lp)


def D(lam, l, R, dR):
    """<lam||grad||l> R as a function of r (the gradient formula)."""
    if lam == l + 1:
        return lambda r: math.sqrt(l + 1) * (dR(r) - l * R(r) / r)
    return lambda r: -math.sqrt(l) * (dR(r) + (l + 1) * R(r) / r)


def Gfun(lam, l, R, dR):
    """G_lambda of the gradient formula."""
    if lam == l + 1:
        return lambda r: -math.sqrt((l + 1) / (2 * l + 1)) * (dR(r) - l * R(r) / r)
    return lambda r: math.sqrt(l / (2 * l + 1)) * (dR(r) + (l + 1) * R(r) / r)


ok_all = True

# ======================================================================== SO2
if "so2" in which:
    print("== SO2: the Theta of H_SO2")
    t0 = time.time()
    # reference: <ac|O_x|bd> = -i int phi_a^* (grad V_cd x grad phi_b),  O_12 = grad_1(1/r12) x p_1
    P = np.stack([np.stack([g.W * np.conj(O[a].v) * O[b].g[..., z] for a in range(no) for b in range(no)])
                  for z in range(3)])                                   # (3, no*no, pts)
    Ox = np.zeros((3, no, no, no, no), dtype=complex)                   # [x, a, b, c, d]
    eps = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
    for c in range(no):
        for d in range(no):
            _, GV = tc.potential(g, np.conj(O[c].v) * O[d].v, lof(c) + lof(d), grad=True)
            for (x, y, z), e in eps.items():
                Ox[x, :, :, c, d] += -1j * e * (P[z] * GV[..., y][None]).sum((1, 2)).reshape(no, no)
    del P
    for eta, name in ((1, "spin-same-orbit"), (2, "spin-other-orbit")):
        I = np.zeros((no, 2, no, 2, no, 2, no, 2), dtype=complex)    # [a sa, c sc, b sb, d sd]
        for x in range(3):
            if eta == 1:
                sp = np.einsum("ij,kl->ikjl", tc.SPIN[x], tc.ID2)      # (sa, sc, sb, sd)
            else:
                sp = 2 * np.einsum("ij,kl->ikjl", tc.ID2, tc.SPIN[x])
            I += 0.5 * np.einsum("abcd,ikjl->aickbjdl", Ox[x], sp)
        Mref = tc.assemble(space, orbs, I)
        # the note's formula
        Mn = np.zeros_like(Mref)
        s1, s2 = (1, 0) if eta == 1 else (0, 1)
        for a, b, c, d in itertools.product(range(len(SH)), repeat=4):
            la, lb, lc, ld = shells[a], shells[b], shells[c], shells[d]
            Ra, dRa, _ = RAD[a]
            Rb, dRb, _ = RAD[b]
            for k in range(0, lc + ld + 1):
                cc = C(lc, k, ld)
                if abs(cc) < 1e-14:
                    continue
                Yk = tc.Yk_radial(g, RAD[c][0], RAD[d][0], k)
                for gg in range(max(k - 1, 0), k + 2):
                    s = 0.0
                    for lam_a in (la - 1, la + 1):
                        for lam_b in (lb - 1, lb + 1):
                            if lam_a < 0 or lam_b < 0:
                                continue
                            nj = ninej2(2 * lam_a, 2 * lam_b, 2 * k, 2, 2, 2, 2 * la, 2 * lb, 2 * gg)
                            ck = C(lam_a, k, lam_b)
                            if abs(nj * ck) < 1e-14:
                                continue
                            Ga, Gb = Gfun(lam_a, la, Ra, dRa), Gfun(lam_b, lb, Rb, dRb)
                            Rk = tc.radint(Ga(g.r) * Gb(g.r) * Yk * g.r, g)
                            s += math.sqrt(6 * (2 * la + 1) * (2 * lb + 1) * (2 * gg + 1)) * nj * ck * Rk
                    if abs(s) < 1e-14:
                        continue
                    theta = 0.5 * (-1) ** k / math.sqrt(2 * k + 1) * eta * cc * s * neg(gg - k)
                    W = tc.tbn(space, a, b, c, d, (2 * s1, gg), (2 * s2, k), (2, 1))
                    Mn += theta * tc.jscalar(W, 1)
        ok_all &= tc.compare(f"SO2 {name}", Mref, Mn)
    print(f"   ({time.time() - t0:.0f} s)")

# ======================================================================== SS
if "ss" in which:
    print("== SS: the Theta of H_SS")
    t0 = time.time()
    # reference: <ac|T_xy|bd> = int dF_x dV_y - delta_xy/3 int gradF . gradV, F = phi_a^* phi_b, V = V_cd
    dF = np.stack([np.stack([g.W * (np.conj(O[a].g[..., x]) * O[b].v + np.conj(O[a].v) * O[b].g[..., x])
                             for a in range(no) for b in range(no)]) for x in range(3)])
    T = np.zeros((3, 3, no, no, no, no), dtype=complex)
    for c in range(no):
        for d in range(no):
            _, GV = tc.potential(g, np.conj(O[c].v) * O[d].v, lof(c) + lof(d), grad=True)
            M = np.stack([np.stack([(dF[x] * GV[..., y][None]).sum((1, 2)) for y in range(3)]) for x in range(3)])
            tr = np.trace(M) / 3
            for x in range(3):
                M[x, x] -= tr
            T[:, :, :, :, c, d] = M.reshape(3, 3, no, no)
    del dF
    I = np.zeros((no, 2, no, 2, no, 2, no, 2), dtype=complex)
    for x in range(3):
        for y in range(3):
            sp = np.einsum("ij,kl->ikjl", tc.SPIN[x], tc.SPIN[y])
            I += 0.5 * np.einsum("abcd,ikjl->aickbjdl", T[x, y], sp)
    Mref = tc.assemble(space, orbs, I)
    Mn = np.zeros_like(Mref)
    for a, b, c, d in itertools.product(range(len(SH)), repeat=4):
        la, lb, lc, ld = shells[a], shells[b], shells[c], shells[d]
        Ra, Rb = RAD[a][0], RAD[b][0]
        Rc, Rd = RAD[c][0], RAD[d][0]
        r = g.r
        for k in range(0, lc + ld + 1):
            cc = C(lc, k, ld)
            if abs(cc) < 1e-14:
                continue
            # F = Y^k/r = r^-(k+1) A + r^k B and its derivatives
            rho = Rc(r) * Rd(r)
            Yk = tc.Yk_radial(g, Rc, Rd, k)
            # recover A and B from Y^k: A = r^k (Y^k - r^(k+1) B); compute B directly
            from scipy.interpolate import CubicSpline
            ib = np.concatenate([[0.0], rho * r ** (1 - k)])
            Bc = CubicSpline(g.r0, ib).antiderivative()(g.r0)
            B = (Bc[-1] - Bc)[1:]
            A = r ** k * (Yk - r ** (k + 1) * B)
            F = r ** (-k - 1) * A + r ** k * B
            F1 = -(k + 1) * r ** (-k - 2) * A + k * r ** (k - 1) * B
            F2 = (k + 1) * (k + 2) * r ** (-k - 3) * A + k * (k - 1) * r ** (k - 2) * B - (2 * k + 1) * rho
            for lam in (k - 2, k, k + 2):
                if lam < 0:
                    continue
                ca = C(la, lam, lb)
                if abs(ca) < 1e-14:
                    continue
                w = np.zeros_like(r)
                for j in (k - 1, k + 1):
                    if j < 0 or abs(lam - j) != 1:
                        continue
                    sj = sixj2(2, 2, 4, 2 * k, 2 * lam, 2 * j)
                    if abs(sj) < 1e-14:
                        continue
                    # H = <j||grad||k> F,  then <lam||grad||j> H
                    if j == k + 1:
                        cH, sH = math.sqrt(k + 1), -k            # H = cH (F' + sH F / r)
                    else:
                        cH, sH = -math.sqrt(k), k + 1
                    H = cH * (F1 + sH * F / r)
                    H1 = cH * (F2 + sH * F1 / r - sH * F / r ** 2)
                    if lam == j + 1:
                        out = math.sqrt(j + 1) * (H1 - j * H / r)
                    else:
                        out = -math.sqrt(j) * (H1 + (j + 1) * H / r)
                    w += sj * out
                w *= (-1) ** (lam + k) * math.sqrt(5)
                theta = -0.25 * (-1) ** k / (2 * k + 1) * ca * cc * tc.radint(Ra(r) * Rb(r) * w * r ** 2, g) * neg((lam - k) // 2)
                if abs(theta) < 1e-15:
                    continue
                W = tc.tbn(space, a, b, c, d, (2, lam), (2, k), (4, 2))
                Mn += theta * tc.jscalar(W, 2)
    ok_all &= tc.compare("SS", Mref, Mn)
    print(f"   ({time.time() - t0:.0f} s)")

# ======================================================================== OO
if "oo" in which:
    print("== OO: current and divergence terms")
    t0 = time.time()
    # reference: <ac|o|bd> = (1/2) int F_i(1) [int K_ij(r12) G_j(2)],  F_i = phi_a^* d_i phi_b, G_j = phi_c^* d_j phi_d,
    # int K_ij G_j = 2 V[G_i] - x_i V[div G] + V[x_i div G]   (K = delta/r + r r / r^3; r r / r^3 = -(r1 - r2)_i d_j (1/r12))
    F = np.stack([np.stack([g.W * np.conj(O[a].v) * O[b].g[..., i] for a in range(no) for b in range(no)])
                  for i in range(3)])
    oo = np.zeros((no, no, no, no), dtype=complex)
    for c in range(no):
        for d in range(no):
            kmx = lof(c) + lof(d) + 2
            Gf = [np.conj(O[c].v) * O[d].g[..., i] for i in range(3)]
            Df = sum(np.conj(O[c].g[..., i]) * O[d].g[..., i] for i in range(3)) + np.conj(O[c].v) * O[d].lap
            VD, _ = tc.potential(g, Df, kmx)
            tot = 0.0
            for i in range(3):
                VG, _ = tc.potential(g, Gf[i], kmx)
                VXD, _ = tc.potential(g, g.X[..., i] * Df, kmx)
                KG = 2 * VG - g.X[..., i] * VD + VXD
                tot = tot + (F[i] * KG[None]).sum((1, 2))
            oo[:, :, c, d] = 0.5 * tot.reshape(no, no)
    del F
    I = 0.5 * np.einsum("abcd,ij,kl->aickbjdl", oo, tc.ID2, tc.ID2)
    Mref = tc.assemble(space, orbs, I)

    def Tfun(lx, ly, k, gg, Rx, Ry, dRy):
        """<lx||[C^k x grad]^(g)||ly> as a radial function (R_x on the bra, grad on R_y), Edmonds (7.1.1)."""
        out = np.zeros_like(g.r)
        for lam in (ly - 1, ly + 1):
            if lam < 0:
                continue
            sj = sixj2(2 * k, 2, 2 * gg, 2 * ly, 2 * lx, 2 * lam)
            ck = C(lx, k, lam)
            if abs(sj * ck) < 1e-14:
                continue
            out += sj * ck * D(lam, ly, Ry, dRy)(g.r)
        return (-1) ** (lx + gg + ly) * math.sqrt(2 * gg + 1) * out * Rx(g.r)

    def jfun(x, y, k, gg):
        lx, ly = shells[x], shells[y]
        Rx, dRx, _ = RAD[x]
        Ry, dRy, _ = RAD[y]
        ph = (-1) ** (k + 1 - gg + ly - lx)
        return 0.5 * (Tfun(lx, ly, k, gg, Rx, Ry, dRy) - ph * Tfun(ly, lx, k, gg, Ry, Rx, dRx))

    def qfun(x, y):
        lx, ly = shells[x], shells[y]
        Rx, dRx, d2Rx = RAD[x]
        Ry, dRy, d2Ry = RAD[y]
        r = g.r
        Lx = d2Rx(r) + 2 * dRx(r) / r - lx * (lx + 1) * Rx(r) / r ** 2
        Ly = d2Ry(r) + 2 * dRy(r) / r - ly * (ly + 1) * Ry(r) / r ** 2
        return 0.5 * (Rx(r) * Ly - Ry(r) * Lx)

    Mn = np.zeros_like(Mref)
    for a, b, c, d in itertools.product(range(len(SH)), repeat=4):
        la, lb, lc, ld = shells[a], shells[b], shells[c], shells[d]
        for k in range(0, max(la + lb, lc + ld) + 2):
            for gg in range(max(k - 1, 0), k + 2):
                nj = ninej2(2 * k, 2 * k, 0, 2, 2, 0, 2 * gg, 2 * gg, 0)
                jab, jcd = jfun(a, b, k, gg), jfun(c, d, k, gg)
                if abs(nj) < 1e-14 or np.abs(jab).max() < 1e-14 or np.abs(jcd).max() < 1e-14:
                    continue
                I1 = tc.radint(jab * tc.Yk_of(g, jcd, k) * g.r, g)
                theta = (-1) ** k * math.sqrt(2 * k + 1) * (-math.sqrt(3)) * nj * I1 * neg(gg - k)
                if abs(theta) > 1e-15:
                    Mn += theta * tc.tbn(space, a, b, c, d, (0, gg), (0, gg), (0, 0))[(0, 0)]
            ca, cc = C(la, k, lb), C(lc, k, ld)
            if abs(ca * cc) > 1e-14:
                I2 = tc.radint(qfun(a, b) * tc.Wk_of(g, qfun(c, d), k) * g.r ** 2, g)
                theta = 0.5 * (-1) ** k / math.sqrt(2 * k + 1) * ca * cc * I2
                Mn += theta * tc.tbn(space, a, b, c, d, (0, k), (0, k), (0, 0))[(0, 0)]
    ok_all &= tc.compare("OO", Mref, Mn)
    print(f"   ({time.time() - t0:.0f} s)")

# ======================================================================== contact and MV
if "bp1" in which:
    print("== contact (through R^k) and mass velocity")
    t0 = time.time()
    # reference: the spin-dependent form -pi sum_{i<j} delta(r_ij) - (8 pi/3) sum_{i<j} s_i.s_j delta(r_ij)
    dl = np.zeros((no, no, no, no), dtype=complex)
    for a in range(no):
        for b in range(no):
            fab = g.W * np.conj(O[a].v) * O[b].v
            for c in range(no):
                for d in range(no):
                    dl[a, b, c, d] = (fab * np.conj(O[c].v) * O[d].v).sum()
    I = 0.5 * (-np.pi) * np.einsum("abcd,ij,kl->aickbjdl", dl, tc.ID2, tc.ID2)
    for x in range(3):
        I += 0.5 * (-8 * np.pi / 3) * np.einsum("abcd,ij,kl->aickbjdl", dl, tc.SPIN[x], tc.SPIN[x])
    Mref = tc.assemble(space, orbs, I)
    Mn = np.zeros_like(Mref)
    for a, b, c, d in itertools.product(range(len(SH)), repeat=4):
        la, lb, lc, ld = shells[a], shells[b], shells[c], shells[d]
        r = g.r
        R4 = tc.radint(RAD[a][0](r) * RAD[b][0](r) * RAD[c][0](r) * RAD[d][0](r) * r ** 2, g)
        for k in range(0, min(la + lb, lc + ld) + 1):
            Hk = (-1) ** k / math.sqrt(2 * k + 1) * C(la, k, lb) * C(lc, k, ld)
            if abs(Hk) < 1e-14:
                continue
            Mn += neg(k) * Hk * (2 * k + 1) * 0.25 * R4 * tc.tbn(space, a, b, c, d, (0, k), (0, k), (0, 0))[(0, 0)]
    ok_all &= tc.compare("contact D2 + SSC", Mref, Mn)
    # mass velocity: reference <a|-p^4/8|b> = -(1/8) int (lap phi_a)^* lap phi_b;  note h(a,b) = -(1/8) int (L R_a)(L R_b) r^2 dr
    I1 = np.zeros((no, 2, no, 2), dtype=complex)
    for a in range(no):
        for b in range(no):
            v = -0.125 * (g.W * np.conj(O[a].lap) * O[b].lap).sum()
            I1[a, :, b, :] = v * tc.ID2
    Mref = tc.assemble1(space, orbs, I1)
    Mn = np.zeros_like(Mref)
    for a in range(len(SH)):
        for b in range(len(SH)):
            if shells[a] != shells[b]:
                continue
            l = shells[a]
            r = g.r

            def LR(s):
                R, dR, d2R = RAD[s]
                return d2R(r) + 2 * dR(r) / r - l * (l + 1) * R(r) / r ** 2
            h = -0.125 * tc.radint(LR(a) * LR(b) * r ** 2, g) * neg(l)
            Mn += -math.sqrt(2 * (2 * l + 1)) * h * tc.one_body_tensor(space, a, b, 0, 0)[(0, 0)]
    ok_all &= tc.compare("mass velocity (one-body rule)", Mref, Mn)
    print(f"   ({time.time() - t0:.0f} s)")

print("bp2e:", "all passed" if ok_all else "FAILED")
