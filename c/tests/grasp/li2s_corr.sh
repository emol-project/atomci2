#!/bin/bash
# li2s_corr.sh -- S2-M7 (Z5b), report only (ATSP needed). Li-like 1s2 2s 2S:
# does correlation (core polarization) close the 7-10% gap between C's full BP
# and arXiv:2604.25515's P+N? Orbitals: HF, then MCHF on complete active spaces
# n = 3, 4, 5 (l <= d, f, g; up to triple excitations from 1s2 2s, the CAS of
# the ATSP manual's Li example), n >= 4 varying only the new layer.
# Delta g_int = g_J - g_D (hydrogenic Dirac), as in run_lilike.sh.
#   ZS        nuclear charges (default "4 10")
#   LI_WORK   work directory (default $HOME/atomci_work/li2s)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
for Z in ${ZS:-4 10}; do
    w=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2s}/Z$Z
    mkdir -p "$w" && cd "$w"
    rm -f wfn.inp
    hf > hf.log 2>&1 <<S
L$Z,2S,$Z.
  1s
2s(1)
all
y
y
n
n
S
    cp wfn.out L_hf.w
    echo "1s(2) 2s(1)" > hf.cfg
    "$A" "$T" L_hf.w hf.cfg 2 alpha=137.035999084 gc2e oz2e > g_hf.out
    for n in 3 4 5; do
        case $n in 3) l=d;; 4) l=f;; *) l=g;; esac
        lsgen > lsgen_$n.log 2>&1 <<S



$n
$l


2
2
1
0
2S
3

S
        cp clist.out cfg.inp
        rm -f wfn.inp
        if [ $n = 3 ]; then cp L_hf.w wfn.inp; vary=all
        elif [ $n = 4 ]; then cp L_3.w wfn.inp; vary=4s,4p,4d,4f
        else cp L_$((n - 1)).w wfn.inp; vary=${n}s,${n}p,${n}d,${n}f,${n}g; fi
        nonh > nonh_$n.log 2>&1
        mchf > mchf_$n.log 2>&1 <<S
L$Z,$Z.
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
S
        cp wfn.out L_$n.w
        cp cfg.inp L_$n.c
        echo "Z=$Z n=$n: $(grep -c '(' L_$n.c) CSFs, $(grep ETOTAL mchf_$n.log | tail -1), NaN lines $(grep -c NaN mchf_$n.log || true)"
        "$A" "$T" L_$n.w L_$n.c 2 alpha=137.035999084 gc2e oz2e > g_$n.out
    done
    python3 - "$Z" g_hf.out g_3.out g_4.out g_5.out <<'PY'
import math, re, sys
Z = int(sys.argv[1])
a = 1 / 137.035999084
k = 1; gam = math.sqrt(1 - (Z * a) ** 2)
E = 1 / math.sqrt(1 + (Z * a / (2 - k + gam)) ** 2)
gD = -1 * (2 * -1 * E - 1) / (2 * 0.5 * 1.5)
N, P = {4: (-0.00000249, 0.00008750), 6: (-0.00000572, 0.00013663),
        10: (-0.00001222, 0.00023491)}[Z]
for f, tag in zip(sys.argv[2:], ("HF", "n=3", "n=4", "n=5")):
    for l in open(f):
        x = l.split()
        if len(x) >= 10 and x[0] == "1/2" and x[1] == "1":
            v = [float(t) for t in x[6:]]
            terms = dict(zip(("RMC", "GC1e", "ORMC", "GC2e", "OZ2e"), v))
            dg = 2.0 + sum(v) - gD
            print(f"Z={Z:2} {tag:4} dg_int {dg:+.5e}  (P+N {P + N:+.5e}, P {P:+.5e})  "
                  + " ".join(f"{t} {u:+.4e}" for t, u in terms.items()))
            break
PY
done
