#!/bin/bash
# li2p_yan.sh -- S2-M7 (Z13, Z15), report only (ATSP needed): Li-like 1s2 2p
# 2P_J term by term against Z.-C. Yan, Phys. Rev. A 66, 022502 (2002)
# (Hylleraas, 3463 terms, infinite nuclear mass). Yan's <H_i> =
# <LSJJ|H_i|LSJJ>, in units of 1e-6 mu_B H (-J), are rebuilt from his reduced
# matrix elements F_1 .. F_9 (Table II, Z = 3 .. 12) by his Eqs. (11)-(14);
# for Z = 3 they reproduce his Table IV. C's Delta g (positive g) is minus it:
#   H1 kinetic (RMC + ORMC), H2 one-electron gauge (GC1e), H3 two-electron
#   gauge (GC2e), H4 orbit-orbit Zeeman (OZ2e), H5 = (m/M) sum_{j!=k} i r_j x grad_k
#   (two-body recoil; C: (m/M) rec2 with rec2 = <sum r_j x p_k>/M_J; m/M of 7Li
#   for every Z, on both sides).
# C's RMC, GC1e and GC2e carry no g_s factor (g_s = 2), as Yan's H1 .. H4.
# Each H_i is also split into Yan's F pieces by its J dependence: for 2P,
# g'_L-type pieces at J = 3/2 are 1/2 of those at J = 1/2, g'_S-type -1 times,
# g_X-type -1/10 times (H1 = F3 (ORMC) + F5 (RMC), H2 = F6 + F8, H3 = F7 + F9).
# Orbitals: ATSP MCHF, complete active spaces of three electrons from 1s2 2p,
# n = 3 .. NMAX (l <= d, f, g, g, ...), n >= 4 varying only the new layer.
# For Li up to n = 6 the H4 and H5 terms converge to Yan (-3.9%, -1.8%) and
# H1-H3 stay ~1-2% low (wavefunction convergence; see the
# technical note, Sec. 7.4).
#   Z        nuclear charge, 3 .. 12 (default 3)
#   NMAX     largest n (default 6; for Li n = 6 takes about an hour and 5.4 GB)
#   LI_WORK  work directory (default $HOME/atomci_work/li2p for Z = 3,
#            $HOME/atomci_work/li2p_Z<Z> otherwise)
#   REUSE=1  keep the work directory; skip the n whose wavefunction (L_n.w)
#            or g_J output (g_n.out, non-empty) is already there
#   NOSO=1   CI without spin-orbit (Yan's setting; see below)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
Z=${Z:-3}
# NOSO=1: no spin-orbit in the CI, so that both J share Yan's nonrelativistic
# wavefunction (outputs g_<n>_noso.out); otherwise SO mixes the CSFs per J
co="alpha=137.035999084 gc2e oz2e rec2"; sfx=""
if [ -n "${NOSO:-}" ]; then co="$co noso"; sfx=_noso; fi
if [ "$Z" = 3 ]; then w=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2p}; else w=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2p_Z$Z}; fi
[ -n "${REUSE:-}" ] || rm -rf "$w"
mkdir -p "$w" && cd "$w"
[ -s L_2.w ] || hf > hf.log 2>&1 <<S
Z$Z,2P,$Z.
  1s
2p(1)
all
y
y
n
n
S
[ -s L_2.w ] || cp wfn.out L_2.w
{ printf '\n\n  1s( 2)  2p( 1)\n 1S0 2P1 2P \n*\n'; } > L_2.c
[ -s g_2$sfx.out ] || "$A" "$T" L_2.w L_2.c 2 $co > g_2$sfx.out
tags="2"
for n in $(seq 3 ${NMAX:-6}); do
    tags="$tags $n"
    if [ -s L_$n.w ] && [ -s L_$n.c ]; then
        [ -s g_$n$sfx.out ] || "$A" "$T" L_$n.w L_$n.c 2 $co > g_$n$sfx.out
        continue
    fi
    case $n in 3) l=d;; 4) l=f;; *) l=g;; esac
    lsgen > lsgen_$n.log 2>&1 <<S



$n
$l


2
2
0
1
2P
3

S
    cp clist.out cfg.inp
    cp L_$((n - 1)).w wfn.inp
    if [ $n = 3 ]; then vary=all; elif [ $n = 4 ]; then vary=4s,4p,4d,4f
    else vary=${n}s,${n}p,${n}d,${n}f,${n}g; fi
    nonh > nonh_$n.log 2>&1
    mchf > mchf_$n.log 2>&1 <<S
Z$Z,$Z.
1
$vary
1s,2p
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
S
    cp wfn.out L_$n.w
    cp cfg.inp L_$n.c
    echo "n=$n: $(grep -c '(' L_$n.c) CSFs, $(grep ETOTAL mchf_$n.log | tail -1), NaN lines $(grep -c NaN mchf_$n.log || true)"
    "$A" "$T" L_$n.w L_$n.c 2 $co > g_$n$sfx.out
done
python3 - "$sfx" $Z $tags <<'PY'
import re, sys
from math import sqrt
sfx = sys.argv[1]
Z = int(sys.argv[2])
me_M = 5.48579909e-4 / (7.01600343 - 3 * 5.48579909e-4)  # electron / 7Li nucleus
a2 = (1 / 137.035999084) ** 2
# Yan, Table II: F1 .. F9 of 2 2P (atomic units)
FT = {3: (-0.205493, -0.0742537, -1.01915618, -0.02184606, -0.5807004, 0.56460470,
          1.12871809, -0.3642931, -0.5288268),
      4: (-0.52620, -0.322199, -4.0957202, -0.0605960, -2.34640, 1.127791,
          2.2299654, -0.7337286, -0.994984270),
      5: (-0.734673, -0.6176277, -9.0121014, -0.1017831, -5.228485, 1.6763909,
          3.2830886, -1.0898777, -1.4333686),
      6: (-0.8697211, -0.9245113, -15.6968520, -0.1439617, -9.194131, 2.2179593,
          4.3137863, -1.4385481, -1.86082895),
      7: (-0.9626148, -1.2348035, -24.1293635, -0.1866840, -14.231577, 2.7558525,
          5.3330681, -1.7830648, -2.2830654),
      8: (-1.0299923, -1.5462220, -34.3019763, -0.2297388, -20.335859, 3.2916097,
          6.34585706, -2.1250906, -2.7024255),
      9: (-1.080951, -1.8580015, -46.21126859, -0.2730117, -27.504558, 3.8260207,
          7.3546250, -2.4655165, -3.1200390),
      10: (-1.1207844, -2.1698596, -59.8555010, -0.3164352, -35.736365, 4.359530,
           8.360737013, -2.8048572, -3.53651421),
      11: (-1.1527495, -2.4816881, -75.2337023, -0.3599670, -45.03051, 4.8924063,
           9.3650058, -3.1434293, -3.9522069),
      12: (-1.1789552, -2.7934478, -92.3452895, -0.4035793, -55.386526, 5.4248213,
           10.3679443, -3.4814374, -4.3673390)}
F = dict(zip(("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"), FT[Z]))
# Yan Eqs. (12)-(14) for L = 1, S = 1/2 ((-1)^{2J'} = -1), and Eq. (11)'s
# diagonal coefficients of g'_L, g'_S, g_X: J = 1/2: 2/3, 1/3, -2/9;
# J = 3/2: 1, -1, 1/15. <H> in 1e-6 mu_B H (-J) is c g 1e6 / (-J).
sL, sS = sqrt(18), sqrt(12)
gl = {"F1": -me_M * F["F1"] / sS, "F2": -a2 * F["F2"] / (2 * sS),
      "F3": a2 * F["F3"] / (2 * sS), "F4": a2 * F["F4"] / (2 * sS)}
gs_ = {"F5": -a2 * F["F5"] / sL, "F6": -a2 * Z / 3 * F["F6"] / sL, "F7": a2 * F["F7"] / 3 / sL}
gx = {"F8": sqrt(5) / 6 * a2 * -Z * F["F8"], "F9": sqrt(5) / 6 * a2 * sqrt(6) / 2 * F["F9"]}
coef = {1: (2 / 3, 1 / 3, -2 / 9), 3: (1.0, -1.0, 1 / 15)}
def piece(tj, k):
    cl, cs, cx = coef[tj]
    for d, c in ((gl, cl), (gs_, cs), (gx, cx)):
        if k in d:
            return c * d[k] * 1e6 / (-tj / 2)
yan = {tj: (piece(tj, "F3") + piece(tj, "F5"), piece(tj, "F6") + piece(tj, "F8"),
            piece(tj, "F7") + piece(tj, "F9"), piece(tj, "F2") + piece(tj, "F4"), piece(tj, "F1"))
       for tj in (1, 3)}
names = ("H1 kin", "H2 GC1e", "H3 GC2e", "H4 OZ2e", "H5 rec")
print(f"Z = {Z}: Yan's <H_i> from his F_i (x 1e-6)")
for tj in (1, 3):
    print(f"   2P{tj}/2 " + "  ".join(f"{nm.split()[0]} {v:+.6f}" for nm, v in zip(names, yan[tj])))
for n in sys.argv[3:]:
    v = {}
    for l in open(f"g_{n}{sfx}.out"):
        x = l.split()
        if len(x) >= 12 and re.match(r"^\d/2$", x[0]) and x[1] == "1":
            rmc, gc, ormc, gc2, oz, rec = map(float, x[6:12])
            v[int(x[0][0])] = (float(x[2]), -rmc * 1e6, -ormc * 1e6, -gc * 1e6, -gc2 * 1e6,
                               -oz * 1e6, me_M * rec * 1e6)
    lab = "HF" if n == "2" else f"n={n}"
    print(f"{lab}   E(2P1/2) = {v[1][0]:.8f}")
    for tj in (1, 3):
        e, rmc, ormc, gc, gc2, oz, rec = v[tj]
        c = (rmc + ormc, gc, gc2, oz, rec)
        print(f"   2P{tj}/2 C/Yan - 1: " + "  ".join(f"{nm.split()[0]} {cv / yv - 1:+.2e}"
                                                  for nm, cv, yv in zip(names, c, yan[tj])))
    # F pieces from the two J values
    (_, r1, o1, g1, t1, _, _), (_, r3, o3, g3, t3, _, _) = v[1], v[3]
    x8 = (g1 + g3) / 0.9; x9 = (t1 + t3) / 0.9
    pc = {"F3 (ORMC)": (o1, o3 / 0.5), "F5 (RMC)": (r1, -r3),
          "F6 (GC1e scalar)": (g1 - x8, None), "F8 (GC1e tensor)": (x8, None),
          "F7 (GC2e scalar)": (t1 - x9, None), "F9 (GC2e tensor)": (x9, None)}
    print("   pieces (J = 1/2 scale) C/Yan - 1: " + "  ".join(
        f"{k.split()[0]} {a / piece(1, k.split()[0]) - 1:+.2e}" for k, (a, _) in pc.items()))
    print(f"   J ratio checks: ORMC {o3 / o1 / 0.5 - 1:+.1e}  RMC {-r3 / r1 - 1:+.1e}")
PY
