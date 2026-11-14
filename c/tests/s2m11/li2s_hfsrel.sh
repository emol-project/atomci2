#!/bin/bash
# li2s_hfsrel.sh -- S2-M11 C2, report only (ATSP needed): the regularized
# alpha^2 hfs correction A_AN/A^(4) of Li 1s2 2s 2S (acatsp hfsrel) against
# Puchalski and Pachucki (arXiv:1310.2161v2): 102.134/5.81063 = 17.577.
# Orbitals: ATSP MCHF, complete active spaces of three electrons from 1s2 2s,
# n = 3 .. NMAX (l <= d, f, g, ...), n >= 4 varying only the new layer; then
# lag= adds NS - 1 cusp-constrained Laguerre s functions (b = BS),
# orthogonalized to the MCHF s orbitals ("far" s); lagfit first replaces the
# MCHF s orbitals by their projections onto the Laguerre span (cusp-exact),
# so that the far s are its exact complement. NS below ~32 distorts 4s, 5s.
# CI space of hfsrel (technical note, Sec. 5.2): the n = 3
# orbitals inner, the n = 4 .. NMAX orbitals near, the added s far; up to two
# electrons outside the inner orbitals, at most one in the far s.
# The second order converges like 1/NS: the NS series is fitted in 1/NS.
#   NMAX     largest MCHF n (default 5)
#   NSLIST   the NS series (default "32 40 48 56 64")
#   BS       b of the added s (default 1.2)
#   LI_WORK  work directory (default $HOME/atomci_work/li2s_hfsrel)
#   REUSE=1  keep the work directory and the outputs already there
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf mchf nonh lsgen hfs || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
py=${PYTHON:-python3}
NMAX=${NMAX:-5}
BS=${BS:-1.2}
w=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2s_hfsrel}
[ -n "${REUSE:-}" ] || rm -rf "$w"
mkdir -p "$w" && cd "$w"
[ -s L_2.w ] || hf > hf.log 2>&1 <<S
Li,2S,3.
  1s
2s(1)
all
y
y
n
n
S
[ -s L_2.w ] || cp wfn.out L_2.w
for n in $(seq 3 $NMAX); do
    [ -s L_$n.w ] && [ -s L_$n.c ] && continue
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
    cp L_$((n - 1)).w wfn.inp
    if [ $n = 3 ]; then vary=all; elif [ $n = 4 ]; then vary=4s,4p,4d,4f
    else vary=${n}s,${n}p,${n}d,${n}f,${n}g; fi
    nonh > nonh_$n.log 2>&1
    mchf > mchf_$n.log 2>&1 <<S
Li,3.
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
    echo "n=$n: $(grep -c '(' L_$n.c) CSFs, $(grep ETOTAL mchf_$n.log | tail -1)"
done
# orbital classes of L_NMAX.w
inner="1s,2s,3s,2p,3p,3d"
near=""
for n in $(seq 4 $NMAX); do
    for l in s p d f g h; do
        lv=$(expr index spdfgh $l); lv=$((lv - 1))
        [ $lv -lt $n ] && [ $lv -le $((n <= 3 ? 2 : (n == 4 ? 3 : 4))) ] && near="$near${near:+,}$n$l"
    done
done
for ns in ${NSLIST:-32 40 48 56 64}; do
    out=hfsrel_$ns.out
    [ -n "${REUSE:-}" ] && [ -s $out ] && { grep '^hfsrel 2S' $out; continue; }
    nfar=$((ns - 1 - NMAX))
    far=$(seq $((NMAX + 1)) $((NMAX + nfar)) | sed 's/$/s/' | paste -sd,)
    $py "$repo/c/python/hfsrel_cfg.py" 3 0 "$inner" "$near" "$far" 2 1 > cfg_$ns.inp
    AC_TIME=1 "$A" "$T" L_$NMAX.w cfg_$ns.inp noso occslots hfsrel lag=0:$ns:$BS lagfit > $out 2> time_$ns.log
    grep '^hfsrel 2S' $out
done
$py - ${NSLIST:-32 40 48 56 64} <<'PY'
import re, sys
import numpy as np
ns = [int(x) for x in sys.argv[1:]]
val = []
for n in ns:
    t = open(f"hfsrel_{n}.out").read()
    val.append(float(re.findall(r"hfsrel 2S: K .*A_AN/A4 (\S+)", t)[0]))
print("   NS     A_AN/A4")
for n, v in zip(ns, val):
    print(f"   {n:3d}  {v:.6f}")
x = 1.0 / np.array(ns, float)
for deg in (1, 2):
    if len(ns) > deg + 1:
        c = np.polyfit(x, val, deg)
        print(f"   fit a + c/N{' + d/N^2' if deg == 2 else ''}: limit {c[-1]:.4f}")
print("   Puchalski-Pachucki: 17.577 (102.134/5.81063)")
PY
