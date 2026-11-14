#!/bin/bash
# he23s_g.sh -- S2-M7 (Z18), report only (ATSP needed): g_J of He 1s2s 3S1
# (infinite nuclear mass) against J. M. Anthony and K. J. Sebastian,
# Phys. Rev. A 48, 3792 (1993), Eq. (36) (erratum PRA 51, 868 (1995) only
# removes the mass-polarization term), with Pekeris's E = -2.17522937824
# and <1/r12> = 0.2681978553:
#   g_J = g_e [1 + alpha^2 (E/3 - <1/r12>/6) - alpha^3 E/(4 pi)]
#       = g_e [1 - (40.99163 + 0.06727) 1e-6].
# For an S state only the spin terms act: C's Delta g_RMC + Delta g_GC (one-
# electron gauge) + Delta g_GC2e (two-electron gauge, spin-same- and
# spin-other-orbit); ORMC and OZ2e vanish (L = 0). C keeps g_s in RMC and
# in the (g_s - 1)/4, g_s/4 factors of GC2e, which carries the alpha^3
# term of Eq. (36). Orbitals: ATSP MCHF for 1s2s 3S, complete active
# spaces n = 2 .. NMAX (l <= g); CI without spin-orbit on the same CSFs.
#   NMAX (default 7); HE_WORK (default $HOME/atomci_work/he23s)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${HE_WORK:-${AC_WORK:-$HOME/atomci_work}/he23s}
mkdir -p "$w" && cd "$w"
for n in $(seq 2 ${NMAX:-7}); do
    case $n in 2) l=s;; 3) l=d;; 4) l=f;; *) l=g;; esac
    ref="2
1
1"
    [ $l = s ] || ref="$ref
0"
    lsgen > lsgen_$n.log 2>&1 <<EOF



$n
$l


$ref
3S
2

EOF
    cp clist.out cfg.inp
    cp clist.out he_$n.c
    rm -f wfn.inp
    [ -s he_$((n - 1)).w ] && cp he_$((n - 1)).w wfn.inp
    vary=all
    [ $n -ge 7 ] && vary=${n}s,${n}p,${n}d,${n}f,${n}g
    nonh > nonh_$n.log 2>&1
    mchf > mchf_$n.log 2>&1 <<EOF
He,2.
1
$vary
1s,2s
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
EOF
    cp wfn.out he_$n.w
    for p in 1 2; do
        "$A" "$T" he_$n.w he_$n.c 1 noso gc2e gc2epart=$p > g_${n}_$p.out
    done
    python3 - "$n" g_${n}_1.out g_${n}_2.out <<'PY'
import sys
from math import pi
n = sys.argv[1]
a = 1 / 137.035999084
ge = 2.00231930436256
E, r12 = -2.17522937824, 0.2681978553
def row(path):
    for l in open(path):
        x = l.split()
        if len(x) >= 10 and x[0] == "2/2" and x[1] == "1":
            return float(x[2]), [float(v) for v in x[5:10]]
e, (gj, rmc, gc, ormc, sso) = row(sys.argv[2])
_, (_, _, _, _, soo) = row(sys.argv[3])
# (a) alpha^2 at g_s = 2: Anthony's Eq. (36) with g_e -> 2
c2 = rmc + gc + sso + soo
t2 = 2 * a * a * (E / 3 - r12 / 6)
# (b) with the g_s factors of Pachucki's Eq. (32): RMC (1 + (g_s - 2)/6),
# GC1e and SSO (g_s - 1), SOO g_s/2; against g_e alpha^2 (...) - g_e alpha^3 E/(4 pi)
c3 = rmc * (1 + (ge - 2) / 6) + (gc + sso) * (ge - 1) + soo * ge / 2
t3 = ge * a * a * (E / 3 - r12 / 6) - ge * a ** 3 * E / (4 * pi)
print(f"   n={n}: E {e:.8f}  RMC {rmc:+.6e} GC1e {gc:+.6e} SSO {sso:+.6e} SOO {soo:+.6e} (ORMC {ormc:+.0e})")
print(f"      g_s = 2:        C {c2 / 2 * 1e6:+.6f}e-6 x 2   Eq. (36) alpha^2 {t2 / 2 * 1e6:+.6f}e-6 x 2"
      f"   C/A-S - 1 {c2 / t2 - 1:+.1e}")
print(f"      g_s factors:    C {c3 / ge * 1e6:+.6f}e-6 x g_e  Eq. (36) alpha^2 + alpha^3 {t3 / ge * 1e6:+.6f}e-6 x g_e"
      f"   C/A-S - 1 {c3 / t3 - 1:+.1e}")
PY
done
