#!/bin/bash
# li2p_zoff_yan.sh -- S2-M7 (Z17), report only: the Zeeman coupling between
# Li-like 1s2 2p 2P1/2 and 2P3/2 (acatsp "zoff", ac_lsj_zeeman_reduced) term
# by term against Z.-C. Yan, Phys. Rev. A 66, 022502 (2002), Eq. (11), which
# gives <LSJ'M|H_Zeeman|LSJM> for J' != J through the same g'_L, g'_S, g_X
# (Eqs. (12)-(14)) as the diagonal g_J; they are built from his F_1 .. F_9
# (Table II). C's off-diagonal elements of each term (RMC, ORMC, GC1e, GC2e,
# OZ2e, rec2) are compared, at M = 1/2, as
#   Yan: F5 (RMC), F3 (ORMC), F6 + F8 (GC1e), F7 + F9 (GC2e), F2 + F4 (OZ2e),
#        F1 (two-body recoil), and the Lande part with g_L = 1, g_S = g_s.
# The wavefunctions are those of li2p_yan.sh with NOSO=1 (CI without
# spin-orbit, Yan's nonrelativistic setting); run it first for each Z.
# Also checked: the same formulas on C's own diagonal pieces reproduce C's
# off-diagonal elements (the angular part alone).
#   ZS    nuclear charges (default "3 6 10"); N  the n of the wavefunction (5)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
n=${N:-5}
for Z in ${ZS:-3 6 10}; do
    if [ "$Z" = 3 ]; then w=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2p}; else w=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2p_Z$Z}; fi
    [ -s "$w/L_$n.w" ] || { echo "li2p_zoff_yan: $w/L_$n.w missing (run li2p_yan.sh, Z=$Z NOSO=1)"; exit 1; }
    out=$w/zoff_${n}_noso.out
    [ -s "$out" ] || "$A" "$T" "$w/L_$n.w" "$w/L_$n.c" 2 alpha=137.035999084 gc2e oz2e rec2 noso zoff \
        > "$out"
    python3 - "$Z" "$out" <<'PY'
import re, sys
from fractions import Fraction as R
from math import factorial as fa, sqrt
Z, path = int(sys.argv[1]), sys.argv[2]
def f(x): return fa(int(x))
def tri(a, b, c):
    if c < abs(a - b) or c > a + b or (a + b + c) != int(a + b + c): return 0.0
    return f(a + b - c) * f(a - b + c) * f(-a + b + c) / f(a + b + c + 1)
def w3j(j1, j2, j3, m1, m2, m3):
    if m1 + m2 + m3 != 0 or tri(j1, j2, j3) == 0: return 0.0
    s = 0.0
    for t in range(0, 60):
        d = [t, j3 - j2 + t + m1, j3 - j1 + t - m2, j1 + j2 - j3 - t, j1 - t - m1, j2 - t + m2]
        if min(d) < 0: continue
        p = 1.0
        for x in d: p *= f(x)
        s += (-1) ** t / p
    return (-1) ** int(j1 - j2 - m3) * sqrt(tri(j1, j2, j3) * f(j1 + m1) * f(j1 - m1) * f(j2 + m2)
                                          * f(j2 - m2) * f(j3 + m3) * f(j3 - m3)) * s
def w6j(a, b, c, d, e, g):
    t = [tri(a, b, c), tri(a, e, g), tri(d, b, g), tri(d, e, c)]
    if 0 in t: return 0.0
    s = 0.0
    for k in range(0, 60):
        x = [k - a - b - c, k - a - e - g, k - d - b - g, k - d - e - c,
             a + b + d + e - k, a + c + d + g - k, b + c + e + g - k]
        if min(x) < 0: continue
        p = 1.0
        for y in x: p *= f(y)
        s += (-1) ** k * f(k + 1) / p
    return sqrt(t[0] * t[1] * t[2] * t[3]) * s
def w9j(a, b, c, d, e, ff, g, h, i):
    s, x = 0.0, R(0)
    while x <= 12:
        s += (-1) ** int(2 * x) * (2 * x + 1) * w6j(a, b, c, ff, i, x) * w6j(d, e, ff, b, x, h) * \
            w6j(g, h, i, x, a, d)
        x += R(1, 2)
    return s
def ph(x):
    assert x == int(x), x
    return -1.0 if int(x) % 2 else 1.0
L, S = 1, R(1, 2)
a2 = (1 / 137.035999084) ** 2
gs = 2.00231930436256
me_M = 5.48579909e-4 / (7.01600343 - 3 * 5.48579909e-4)
FT = {3: (-0.205493, -0.0742537, -1.01915618, -0.02184606, -0.5807004, 0.56460470,
          1.12871809, -0.3642931, -0.5288268),
      6: (-0.8697211, -0.9245113, -15.6968520, -0.1439617, -9.194131, 2.2179593,
          4.3137863, -1.4385481, -1.86082895),
      10: (-1.1207844, -2.1698596, -59.8555010, -0.3164352, -35.736365, 4.359530,
           8.360737013, -2.8048572, -3.53651421)}
F = dict(zip(("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"), FT[Z]))
sS, sL = sqrt(6 * (2 * S + 1)), sqrt(6 * (2 * L + 1))

def coefs(Jp, J, M):
    """Yan Eq. (11): <L S J' M|H|L S J M>/(mu_B H) = cL g'_L + cS g'_S + cX g_X"""
    pre = sqrt((2 * J + 1) * (2 * Jp + 1)) * w3j(Jp, 1, J, -M, 0, M) * sqrt(6)
    cL = pre * ph(1 - M + J + Jp + L + S) * w6j(L, Jp, S, J, L, 1)
    cS = pre * ph(1 - M + L + S) * w6j(J, Jp, 1, S, S, L)
    cX = pre * ph(1 - M + Jp) * w9j(L, L, 2, S, S, 1, Jp, J, 1)
    return cL, cS, cX

def parts(Jp, J, M, Fv):
    """Yan's contributions of each F to <J' M|H_i|J M>/(mu_B H), Eqs. (12)-(14)
    with (-1)^(2J'), grouped by C's term"""
    cL, cS, cX = coefs(Jp, J, M)
    s2 = ph(2 * Jp)
    return {"RMC": {"F5": cS * a2 / sL * s2 * Fv["F5"]},
            "ORMC": {"F3": cL * a2 / (2 * sS) * Fv["F3"]},
            "GC1e": {"F6": cS * a2 / sL * s2 * Z / 3 * Fv["F6"],
                     "F8": cX * sqrt(5) / 6 * a2 * -Z * Fv["F8"]},
            "GC2e": {"F7": cS * a2 / sL * s2 * -Fv["F7"] / 3,
                     "F9": cX * sqrt(5) / 6 * a2 * sqrt(6) / 2 * Fv["F9"]},
            "OZ2e": {"F2F4": cL * a2 / (2 * sS) * (-Fv["F2"] + Fv["F4"])},
            "rec": {"F1": cL * -me_M / sS * Fv["F1"]}}

def pieces(Jp, J, M, Fv):
    cL, cS, _ = coefs(Jp, J, M)
    out = {"Lande": cL * sqrt((2 * L + 1) * L * (L + 1) / 6)
           + cS * sqrt((2 * S + 1) * S * (S + 1) / 6) * gs * ph(2 * Jp)}
    for k, d in parts(Jp, J, M, Fv).items():
        out[k] = sum(d.values())
    return out

# C: the zoff row 1/2 #1 - 3/2 #1 and the diagonal Delta g of both levels
lv, zo = {}, None
for l in open(path):
    x = l.split()
    if len(x) == 12 and re.match(r"^\d/2$", x[0]) and x[1] == "1":
        lv[int(x[0][0])] = [float(v) for v in x[5:12]]
    if len(x) >= 13 and x[0] == "1/2" and x[1] == "1" and x[2] == "3/2" and x[3] == "1":
        zo = [float(v) for v in x[4:13]]
_, Sred, LgS, rmc, gc, ormc, gc2, oz, rec = zo
M = R(1, 2)
# <J_A M|O_z|J_B M> = (-1)^(J_A - M) (J_A 1 J_B; -M 0 M) <A||O||B>, A = 1/2, B = 3/2;
# Yan's <H_i>/(mu_B H) = <O_i>_C for H1..H4 (his g_J is negative) and
# -(m/M) <O_rec>_C for H5 (C's rec2 enters as -(m/M) rec2)
we = ph(R(1, 2) - M) * w3j(R(1, 2), 1, R(3, 2), -M, 0, M)
c = {"Lande": LgS * we, "RMC": rmc * we, "ORMC": ormc * we, "GC1e": gc * we, "GC2e": gc2 * we,
     "OZ2e": oz * we, "rec": -me_M * rec * we}
y = pieces(R(1, 2), R(3, 2), M, F)
sgn = 1.0 if c["Lande"] * y["Lande"] > 0 else -1.0   # the phase of C's eigenvectors
print(f"Z = {Z}: <2P1/2 M=1/2|H|2P3/2 M=1/2>/(mu_B H), C (wavefunction {path.split('/')[-1]}) against Yan's F_i")
for k in c:
    cv = sgn * c[k]
    print(f"   {k:6} C {cv:+.10e}   Yan {y[k]:+.10e}   C/Yan - 1 {cv / y[k] - 1:+.2e}")
# The angular part alone: scale each of Yan's F so that his diagonal pieces
# equal C's diagonal ones (2J = 1 and 3; GC1e and GC2e have two F each, the
# two J give two equations), then predict C's off-diagonal elements.
# C's diagonal <J J|O|J J> = J Delta g (H1..H4), -(m/M) J rec2 (H5).
cols = {"RMC": 1, "GC1e": 2, "ORMC": 3, "GC2e": 4, "OZ2e": 5, "rec": 6}
cd = {}
for tj in (1, 3):
    J = tj / 2
    v = lv[tj]
    cd[tj] = {k: (J * v[i] if k != "rec" else -me_M * J * v[i]) for k, i in cols.items()}
pd = {tj: parts(R(tj, 2), R(tj, 2), R(tj, 2), F) for tj in (1, 3)}
po = parts(R(1, 2), R(3, 2), M, F)
print("   angular part: C's off-diagonal from C's own diagonal (two J) through Yan's Eq. (11)")
for k in cols:
    names = list(pd[1][k])
    if len(names) == 1:
        nm = names[0]
        s = cd[1][k] / pd[1][k][nm]
        s3 = cd[3][k] / pd[3][k][nm]
        pred = s * po[k][nm]
        extra = f"   (scale from J=1/2 and 3/2 agree to {s3 / s - 1:+.1e})"
    else:
        n1, n2 = names
        a11, a12, a21, a22 = pd[1][k][n1], pd[1][k][n2], pd[3][k][n1], pd[3][k][n2]
        det = a11 * a22 - a12 * a21
        s1 = (cd[1][k] * a22 - a12 * cd[3][k]) / det
        s2 = (a11 * cd[3][k] - a21 * cd[1][k]) / det
        pred = s1 * po[k][n1] + s2 * po[k][n2]
        extra = ""
    cv = sgn * c[k]
    print(f"   {k:6} C {cv:+.10e}   from C's diagonal {pred:+.10e}   ratio - 1 {cv / pred - 1:+.1e}{extra}")
PY
done
