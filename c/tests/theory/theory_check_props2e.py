# props2e: the two-electron property operators whose radial functions the technical note
# describes in words:
#   GC2e  (Sec. 5.1)   Theta_GC2e with c'_K' and I_gc,
#   OZ2e  (Sec. 5.1)   terms 1 and 2 with I_oz,
#   m1s2  (Sec. 5.3)   the spin two-electron M1 correction, K = 0 and K = 2 (u_lambda),
# read as in the C headers (ac_radial.h: ac_radial_gcj, ac_radial_oz, ac_radial_m1w), with the
# gradient formula (Sec. 4.2), against the operators built in coordinate space (theory_coord.py).
# z (Q = 0) components, alpha^2 = 1.
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

which = sys.argv[1:] or ["gc2e", "oz2e", "m1s2"]
NEG = "neg" in which


def neg(x):
    return (-1) ** x if NEG else 1


SH = [(0, 1.1), (1, 0.9), (1, 0.5), (2, 0.7)]
g = tc.Grid(nr=400, rmax=8.0, nth=12, nph=24)
shells = [l for l, _ in SH]
nmax = [2 * (2 * l + 1) for l in shells]
space = tc.Space(shells, [o for o in itertools.product(*[range(n + 1) for n in nmax]) if sum(o) == 2])
orbs = [(s, m) for s, (l, _) in enumerate(SH) for m in range(-l, l + 1)]
no = len(orbs)
O = [tc.Orb(g, SH[s][0], m, SH[s][1]) for (s, m) in orbs]
RAD = [tc.radial(l, z) for l, z in SH]
r = g.r
ns = len(SH)
S, ID2 = tc.SPIN, tc.ID2
print(f"dets {space.dim}")
ok_all = True


def lof(o):
    return SH[orbs[o][0]][0]


def C(l, k, lp):
    return tc.rmeC(l, k, lp)


def Dop(lam, l, R, dR):
    """<lam||grad||l> R (the gradient formula) on the grid."""
    if lam == l + 1:
        return math.sqrt(l + 1) * (dR(r) - l * R(r) / r)
    if lam == l - 1:
        return -math.sqrt(l) * (dR(r) + (l + 1) * R(r) / r)
    return 0.0 * r


def gcq(k, kp, gg, x, y):
    """q_xy of ac_radial_gcj: r R_x <l_x||[C^k x [C^1 x grad]^(K')]^(g)||l_y> R_y (Edmonds (7.1.1) twice)."""
    lx, ly = shells[x], shells[y]
    Rx = RAD[x][0](r)
    Ry, dRy, _ = RAD[y]
    out = np.zeros_like(r)
    for lam in range(abs(lx - k), lx + k + 1):
        s6 = sixj2(2 * k, 2 * kp, 2 * gg, 2 * ly, 2 * lx, 2 * lam)
        ck = C(lx, k, lam)
        if abs(s6 * ck) < 1e-14:
            continue
        inner = np.zeros_like(r)
        for mu in (ly - 1, ly + 1):
            if mu < 0:
                continue
            t6 = sixj2(2, 2, 2 * kp, 2 * ly, 2 * lam, 2 * mu)
            c1 = C(lam, 1, mu)
            if abs(t6 * c1) < 1e-14:
                continue
            inner += t6 * c1 * Dop(mu, ly, Ry, dRy)
        out += s6 * ck * (-1) ** (lam + kp + ly) * math.sqrt(2 * kp + 1) * inner
    return r * Rx * (-1) ** (lx + gg + ly) * math.sqrt(2 * gg + 1) * out


def gce(k, kp, gg, x, y):
    """the full electron-1 function: q_xy + (-1)^(k-g+l_y-l_x) q_yx (gradient on f = phi_x^* phi_y)."""
    lx, ly = shells[x], shells[y]
    return gcq(k, kp, gg, x, y) + (-1) ** (k - gg + ly - lx) * gcq(k, kp, gg, y, x) * NEGT


NEGT = -1 if "negt" in which else 1     # control: flip the phase of the bra-gradient term


def ooj(k, gg, x, y):
    """the current function of H_OO (Sec. 4.2), (1/2)[T_xy - (-1)^(k+1-g+l_y-l_x) T_yx]."""
    lx, ly = shells[x], shells[y]

    def T(lx, ly, X, Y):
        Rx = RAD[X][0](r)
        Ry, dRy, _ = RAD[Y]
        out = np.zeros_like(r)
        for lam in (ly - 1, ly + 1):
            if lam < 0:
                continue
            sj = sixj2(2 * k, 2, 2 * gg, 2 * ly, 2 * lx, 2 * lam)
            ck = C(lx, k, lam)
            if abs(sj * ck) > 1e-14:
                out += sj * ck * Dop(lam, ly, Ry, dRy)
        return (-1) ** (lx + gg + ly) * math.sqrt(2 * gg + 1) * out * Rx
    return 0.5 * (T(lx, ly, x, y) - (-1) ** (k + 1 - gg + ly - lx) * T(ly, lx, y, x))


def ooq(x, y):
    lx, ly = shells[x], shells[y]
    Rx, dRx, d2Rx = RAD[x]
    Ry, dRy, d2Ry = RAD[y]
    Lx = d2Rx(r) + 2 * dRx(r) / r - lx * (lx + 1) * Rx(r) / r ** 2
    Ly = d2Ry(r) + 2 * dRy(r) / r - ly * (ly + 1) * Ry(r) / r ** 2
    return 0.5 * (Rx(r) * Ly - Ry(r) * Lx)


# ======================================================================== GC2e
if "gc2e" in which:
    print("== GC2e (Sec. 5.1)")
    t0 = time.time()
    # reference: <ac|[grad_1(1/r12) x (z x r_1)]_m|bd> = int f [grad V_cd x (z x r)]_m,  z x r = (-y, x, 0)
    zr = np.stack([-g.X[..., 1], g.X[..., 0], np.zeros_like(g.X[..., 0])], axis=-1)
    F = np.stack([g.W * np.conj(O[a].v) * O[b].v for a in range(no) for b in range(no)])
    Gm = np.zeros((3, no, no, no, no), dtype=complex)
    for c in range(no):
        for d in range(no):
            _, GV = tc.potential(g, np.conj(O[c].v) * O[d].v, lof(c) + lof(d), grad=True)
            cr = np.cross(GV, zr)
            for m in range(3):
                Gm[m, :, :, c, d] = (F * cr[..., m][None]).sum((1, 2)).reshape(no, no)
    for eta, name in ((1, "spin-same-orbit"), (2, "spin-other-orbit")):
        I = np.zeros((no, 2, no, 2, no, 2, no, 2), dtype=complex)
        for m in range(3):
            spn = (np.einsum("ij,kl->ikjl", S[m], ID2) if eta == 1 else 2 * np.einsum("ij,kl->ikjl", ID2, S[m]))
            I += 0.5 * np.einsum("abcd,ikjl->aickbjdl", Gm[m], spn)
        Mref = tc.assemble(space, orbs, I)
        Mn = np.zeros_like(Mref)
        s1, s2 = (1, 0) if eta == 1 else (0, 1)
        cp = {0: 2 / math.sqrt(3), 1: 1.0, 2: -math.sqrt(5 / 3) * (-1 if "negc" in which else 1)}
        for a, b, c, d in itertools.product(range(ns), repeat=4):
            la, lb, lc, ld = shells[a], shells[b], shells[c], shells[d]
            Rc, Rd = RAD[c][0], RAD[d][0]
            for k in range(0, lc + ld + 1):
                cc = C(lc, k, ld)
                if abs(cc) < 1e-14:
                    continue
                Yk = tc.Yk_radial(g, Rc, Rd, k)
                terms = []
                # density term: g = k, K' = 0, c' = -2, I -> <la||C^k||lb> R^k(ab; cd)
                Rk = tc.radint(RAD[a][0](r) * RAD[b][0](r) * Yk * r, g)
                terms.append((0, k, -2.0, C(la, k, lb) * Rk))
                for kp in (0, 1, 2):
                    for gg in range(abs(k - kp), k + kp + 1):
                        e = gce(k, kp, gg, a, b)
                        if np.abs(e).max() < 1e-14:
                            continue
                        terms.append((kp, gg, cp[kp], tc.radint(e * Yk * r, g)))
                for kp, gg, cpk, Igc in terms:
                    th = 0.5 * eta * cpk * (-1) ** k * cc * Igc / math.sqrt((2 * kp + 1) * (2 * k + 1)) * neg(gg - k)
                    if abs(th) < 1e-15:
                        continue
                    W = tc.tbn(space, a, b, c, d, (2 * s1, gg), (2 * s2, k), (2, kp))
                    Mn += th * tc.jvector_z(W, 1, kp)
        ok_all &= tc.compare(f"GC2e {name} (z)", Mref, Mn)
    print(f"   ({time.time() - t0:.0f} s)")

# ======================================================================== OZ2e
if "oz2e" in which:
    print("== OZ2e (Sec. 5.1)")
    t0 = time.time()
    # reference: O_z = -(1/2) sum_{i!=j} [r_i x (K_ij p_j)]_z;  int K_mn G_n = 2 V[G_m] - x_m V[div G] + V[x_m div G]
    F = np.stack([g.W * np.conj(O[a].v) * O[b].v for a in range(no) for b in range(no)])
    oz = np.zeros((no, no, no, no), dtype=complex)
    for c in range(no):
        for d in range(no):
            kmx = lof(c) + lof(d) + 2
            Df = sum(np.conj(O[c].g[..., i]) * O[d].g[..., i] for i in range(3)) + np.conj(O[c].v) * O[d].lap
            VD, _ = tc.potential(g, Df, kmx)
            KG = []
            for m in (0, 1):
                VG, _ = tc.potential(g, np.conj(O[c].v) * O[d].g[..., m], kmx)
                VXD, _ = tc.potential(g, g.X[..., m] * Df, kmx)
                KG.append(2 * VG - g.X[..., m] * VD + VXD)
            v = -1j * (g.X[..., 0] * KG[1] - g.X[..., 1] * KG[0])      # (r x (K p G))_z with p = -i grad
            oz[:, :, c, d] = -0.5 * (F * v[None]).sum((1, 2)).reshape(no, no)
    I = np.einsum("abcd,ij,kl->aickbjdl", oz, ID2, ID2)
    Mref = tc.assemble(space, orbs, I)
    Mn = np.zeros_like(Mref)
    for a, b, c, d in itertools.product(range(ns), repeat=4):
        la, lb, lc, ld = shells[a], shells[b], shells[c], shells[d]
        for k in range(0, max(la + lb, lc + ld) + 2):
            # term 1
            for g1 in range(max(k - 1, 0), k + 2):
                cg1 = sum(sixj2(2 * k, 2, 2 * g1, 2 * lb, 2 * la, 2 * lam) * C(la, k, lam) * C(lam, 1, lb)
                          for lam in range(abs(la - k), la + k + 1))
                cg1 *= (-1) ** (la + g1 + lb) * math.sqrt(2 * g1 + 1)
                if abs(cg1) < 1e-14:
                    continue
                for g2 in range(max(k - 1, 0), k + 2):
                    nj = ninej2(2 * k, 2 * k, 0, 2, 2, 2, 2 * g1, 2 * g2, 2)
                    if abs(nj) < 1e-14:
                        continue
                    j = ooj(k, g2, c, d)
                    if np.abs(j).max() < 1e-14:
                        continue
                    Ioz = tc.radint(r * RAD[a][0](r) * RAD[b][0](r) * tc.Yk_of(g, j, k) * r, g)
                    th = 2 * math.sqrt(6) * (-1) ** k * math.sqrt(2 * k + 1) * nj * cg1 * Ioz * neg(g1 - k)
                    if abs(th) > 1e-15:
                        Mn += th * tc.tbn(space, a, b, c, d, (0, g1), (0, g2), (0, 1))[(0, 0)]
            # term 2
            cc = C(lc, k, ld)
            if abs(cc) < 1e-14:
                continue
            q = ooq(c, d)
            for gg in range(max(k - 1, 0), k + 2):
                e = gce(k, 1, gg, a, b)
                if np.abs(e).max() < 1e-14:
                    continue
                Ioz = tc.radint(e * tc.Wk_of(g, q, k) * r ** 2, g)
                th = math.sqrt(2 / 3) * (-1) ** k / math.sqrt(2 * k + 1) * cc * Ioz
                if abs(th) > 1e-15:
                    Mn += th * tc.tbn(space, a, b, c, d, (0, gg), (0, k), (0, 1))[(0, 0)]
    ok_all &= tc.compare("OZ2e (z)", Mref, Mn)
    print(f"   ({time.time() - t0:.0f} s)")

# ======================================================================== m1s2
if "m1s2" in which:
    print("== M1 spin two-electron term (Sec. 5.3; K = 0 and K = 2)")
    t0 = time.time()
    # reference: dmu_z = -sum_{a!=b} sum_j s_{a,j} d_j d_z r_ab;
    # <ac|d_j d_z r12|bd> = -int d_j f (z V[rho] - V[z rho])   (electron 1 on a, b)
    dF = np.stack([np.stack([g.W * (np.conj(O[a].g[..., j]) * O[b].v + np.conj(O[a].v) * O[b].g[..., j])
                             for a in range(no) for b in range(no)]) for j in range(3)])
    M = np.zeros((3, no, no, no, no), dtype=complex)
    for c in range(no):
        for d in range(no):
            rho = np.conj(O[c].v) * O[d].v
            kmx = lof(c) + lof(d) + 1
            V0, _ = tc.potential(g, rho, kmx)
            Vz, _ = tc.potential(g, g.X[..., 2] * rho, kmx)
            dzU = g.X[..., 2] * V0 - Vz
            for j in range(3):
                M[j, :, :, c, d] = -(dF[j] * dzU[None]).sum((1, 2)).reshape(no, no)
    I = np.zeros((no, 2, no, 2, no, 2, no, 2), dtype=complex)
    for j in range(3):
        I += -np.einsum("abcd,ikjl->aickbjdl", M[j], np.einsum("ij,kl->ikjl", S[j], ID2))
    Mref = tc.assemble(space, orbs, I)

    def Zk_derivs(c, d, k):
        """Z^k(r) = int W_k(r, s) rho(s) s^2 ds and Z', Z'' (rho = R_c R_d), from the cumulative integrals."""
        Rc, dRc, _ = RAD[c]
        Rd, dRd, _ = RAD[d]
        rho = Rc(r) * Rd(r)
        drho = dRc(r) * Rd(r) + Rc(r) * dRd(r)
        from scipy.interpolate import CubicSpline

        def cum(p):
            y = np.concatenate([[0.0], rho * r ** p])
            return CubicSpline(g.r0, y).antiderivative()(g.r0)
        # terms c r^n X_p, X = inner (sign +1) or outer (sign -1)
        terms = [(1 / (2 * k + 3), -k - 1, +1, k + 4), (1 / (2 * k + 3), k + 2, -1, 1 - k),
                 (-1 / (2 * k - 1), -k + 1, +1, k + 2), (-1 / (2 * k - 1), k, -1, 3 - k)]
        Z = np.zeros_like(r)
        Z1 = np.zeros_like(r)
        Z2 = np.zeros_like(r)
        for cf, n, sgn_, p in terms:
            cu = cum(p)
            X = cu[1:] if sgn_ > 0 else (cu[-1] - cu)[1:]
            Z += cf * r ** n * X
            Z1 += cf * (n * r ** (n - 1) * X + sgn_ * r ** (n + p) * rho)
            Z2 += cf * (n * (n - 1) * r ** (n - 2) * X + n * sgn_ * r ** (n - 1 + p) * rho
                        + sgn_ * ((n + p) * r ** (n + p - 1) * rho + r ** (n + p) * drho))
        return Z, Z1, Z2

    def ddw(lam, k, F, F1, F2):
        """<lam||[grad x grad]^(2)||k> applied to F C^k: (-1)^(lam+k) sqrt5 sum_j {1 1 2; k lam j} <lam||grad||j><j||grad||k>."""
        w = np.zeros_like(r)
        for j in (k - 1, k + 1):
            if j < 0 or abs(lam - j) != 1:
                continue
            sj = sixj2(2, 2, 4, 2 * k, 2 * lam, 2 * j)
            if abs(sj) < 1e-14:
                continue
            if j == k + 1:
                cH, sH = math.sqrt(k + 1), -k
            else:
                cH, sH = -math.sqrt(k), k + 1
            H = cH * (F1 + sH * F / r)
            H1 = cH * (F2 + sH * F1 / r - sH * F / r ** 2)
            out = math.sqrt(j + 1) * (H1 - j * H / r) if lam == j + 1 else -math.sqrt(j) * (H1 + (j + 1) * H / r)
            w += sj * out
        return w * (-1) ** (lam + k) * math.sqrt(5)

    Mn = np.zeros_like(Mref)
    for a, b, c, d in itertools.product(range(ns), repeat=4):
        la, lb, lc, ld = shells[a], shells[b], shells[c], shells[d]
        Ra, Rb = RAD[a][0](r), RAD[b][0](r)
        for k in range(0, lc + ld + 1):
            cc = C(lc, k, ld)
            if abs(cc) < 1e-14:
                continue
            # K = 0
            ca = C(la, k, lb)
            if abs(ca) > 1e-14:
                Rk = tc.radint(Ra * Rb * tc.Yk_radial(g, RAD[c][0], RAD[d][0], k) * r, g)
                th = -(2 / 3) * (-1) ** k * ca * cc / math.sqrt(2 * k + 1) * Rk
                W = tc.tbn(space, a, b, c, d, (2, k), (0, k), (2, 0))
                Mn += th * W[(0, 0)]
            # K = 2
            Z, Z1, Z2 = Zk_derivs(c, d, k)
            for lam in (k - 2, k, k + 2):
                if lam < 0:
                    continue
                cl = C(la, lam, lb)
                if abs(cl) < 1e-14:
                    continue
                u = tc.radint(Ra * Rb * ddw(lam, k, Z, Z1, Z2) * r ** 2, g)
                th = (-1) ** k * cl * cc / (math.sqrt(3) * (2 * k + 1)) * u * neg((lam - k) // 2)
                if abs(th) < 1e-15:
                    continue
                W = tc.tbn(space, a, b, c, d, (2, lam), (0, k), (2, 2))
                Mn += th * tc.jvector_z(W, 1, 2)
    ok_all &= tc.compare("M1 spin two-electron, K = 0 + K = 2 (z)", Mref, Mn)
    print(f"   ({time.time() - t0:.0f} s)")

print("props2e:", "all passed" if ok_all else "FAILED")
