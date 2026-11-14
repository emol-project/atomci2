#!/bin/bash
# he_m1.sh -- S2-M8 (P4), report only (ATSP needed): the relativistic M1
# transition He 1s2s 3S1 -> 1s2 1S0 from the spin-scalar part of the M1
# operator (acatsp "trans", line M1r: ac_lsj_m1_rel1e, whose orbital and
# tensor one-electron terms vanish here) against
# K. Pachucki, Phys. Rev. A 69, 052502 (2004), Eqs. (64)-(67):
#   <2 3S| k^2 (r1^2 - r2^2)/12 + (p1^2 - p2^2)/3 - (Z/r1 - Z/r2)/6 |1 1S>
#     = alpha^2 0.3189852,   A = 1.272426e-4 s^-1   (k = 0.7284949988).
# The nonrelativistic M1 (L + g_s S) vanishes between the two; the two-
# electron, orbital and tensor parts of the operator vanish too (singlet-
# triplet, S states). With the spin factor <3S1 0|s1z - s2z|1S0> = 1,
#   |<1S0||delta mu||3S1>| = sqrt(3) 2 alpha^2 X,   X -> 0.3189852.
# Both states are CI expansions (all two-electron CSFs of the term, nothing
# else) on one ATSP orbital set, MCHF-optimized for one of them:
#   ORB=3S  for 1s2s 3S (spectroscopic 1s, 2s), n = 2 .. NMAX
#   ORB=1S  for 1s2 1S (spectroscopic 1s), n = 2 .. NMAX
#   ORB=AV  for the average of the two (ATSP mchf, one block per term)
# l <= g. CI without spin-orbit (the operator acts on nonrelativistic
# wavefunctions). The k of the C line is that of the CI energies.
#   NMAX  (default 6); HE_WORK (default $HOME/atomci_work/hem1)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${HE_WORK:-${AC_WORK:-$HOME/atomci_work}/hem1}/${ORB:-3S}
mkdir -p "$w" && cd "$w"
orb=${ORB:-3S}
gen() { # n, l, term (1S: reference 1s2, 3S: 1s2s)
    if [ "$3" = 1S ]; then ref="1
2"; else ref="2
1
1"; [ "$2" = s ] || ref="$ref
0"; fi  # (2p asked once l >= p)
    lsgen > lsgen_$3_$1.log 2>&1 <<EOF



$1
$2


$ref
$3
2

EOF
}
for n in $(seq 2 ${NMAX:-6}); do
    case $n in 2) l=p;; 3) l=d;; 4) l=f;; *) l=g;; esac
    [ $n = 2 ] && l=s
    # orbitals
    rm -f wfn.inp
    [ -s he_$((n - 1)).w ] && cp he_$((n - 1)).w wfn.inp
    eig=1
    if [ $orb = AV ]; then
        # one block per term: the energy average of 1 1S and 2 3S
        gen $n $l 1S; cp clist.out S1.c
        gen $n $l 3S; cp clist.out S3.c
        { head -2 S1.c; sed -n '3,$p' S1.c | grep -v '^\*'; echo '*'
          head -2 S1.c; sed -n '3,$p' S3.c | grep -v '^\*'; echo '*'; } > cfg.inp
        eig="1
1"
    else
        gen $n $l $orb
        cp clist.out cfg.inp
    fi
    vary=all
    [ $n -ge 7 ] && vary=${n}s,${n}p,${n}d,${n}f,${n}g
    if [ $orb = 1S ]; then spec=1s; else spec=1s,2s; fi
    nonh > nonh_$n.log 2>&1
    mchf > mchf_$n.log 2>&1 <<EOF
He,2.
$eig
$vary
$spec
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
    # both terms on these orbitals: all two-electron CSFs of 1S and of 3S
    gen $n $l 1S; cp clist.out S1.c
    gen $n $l 3S; cp clist.out S3.c
    # (ORB=3S: the n l=n-1 orbital of the last layer never occurs in 3S, so
    # MCHF leaves it out; drop the 1S CSFs that need an orbital not in he_n.w)
    python3 - he_$n.w S1.c S3.c > both.c <<'PY'
import struct, sys
labs = set()
with open(sys.argv[1], "rb") as f:
    while (b := f.read(4)):
        r = f.read(struct.unpack("<i", b)[0]); f.read(4)
        labs.add(r[12:15].decode().strip())
out = []
for i, path in enumerate(sys.argv[2:]):
    lines = open(path).read().splitlines()
    if i == 0:
        out += lines[:2]
    body = [l for l in lines[2:] if not l.startswith("*")]
    for cfg, cpl in zip(body[0::2], body[1::2]):
        if all(t.split("(")[0].strip() in labs for t in cfg.split(")") if "(" in t):
            out += [cfg, cpl]
print("\n".join(out + ["*"]))
PY
    "$A" "$T" he_$n.w both.c 1 noso trans > m1_$n.out
    python3 - "$n" "$orb" m1_$n.out <<'PY'
import re, sys
from math import sqrt
n, orb, path = sys.argv[1:4]
alpha = 1 / 137.035999084
tau0 = 2.4188843265857e-17
e, dm, a = {}, None, None
for l in open(path):
    x = l.split()
    if len(x) >= 3 and re.match(r"^\d+/2$", x[0]) and x[1] == "1" and x[0] not in e:
        e[x[0]] = float(x[2])
    if len(x) >= 9 and x[4] == "M1r" and x[0] == "0/2" and x[2] == "2/2":
        dm, a = float(x[7]), float(x[8])
X = abs(dm) / (2 * sqrt(3) * alpha ** 2)
w = e["2/2"] - e["0/2"]
# A at Pachucki's k, from X: A = (4/3) alpha^9 k^3 X^2 / tau0 (k in hartree);
# X itself carries this k only through its k^2 r^2 term (C's k here)
k = 0.7284949988
print(f"   ORB={orb} n={n}: E(1 1S) {e['0/2']:.8f}  E(2 3S) {e['2/2']:.8f}  k {w:.8f}"
      f"  X {X:.7f} (0.3189852)  A {a:.6e} s^-1, at k = {k}: {4 / 3 * alpha ** 9 * k ** 3 * X ** 2 / tau0:.6e} (1.272426e-04)")
PY
done
