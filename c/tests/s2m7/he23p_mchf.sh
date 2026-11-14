#!/bin/bash
# he23p_mchf.sh -- MCHF orbitals for He 1s2p 3P (ATSP), complete active spaces
# (single and double excitations from 1s2p) for n in $NS (default 3 4 5; l <= g,
# so n = 6, 7 add only s..g orbitals), each started from he_<n-1>.w when it
# exists. Writes he_<n>.w and he_<n>.c in the work directory.
#   ATSP_BIN  ATSP binaries (default $HOME/atsp_work/atsp/bin)
#   HE_WORK   work directory (default $HOME/atomci_work/he23p)
set -euo pipefail
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$(dirname "$0")/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh lsgen || exit 0
w=${HE_WORK:-${AC_WORK:-$HOME/atomci_work}/he23p}
mkdir -p "$w" && cd "$w"
for n in ${NS:-3 4 5}; do
    case $n in 3) l=d;; 4) l=f;; *) l=g;; esac
    rm -f wfn.inp
    [ -s he_$((n - 1)).w ] && cp he_$((n - 1)).w wfn.inp
    lsgen > lsgen_$n.log 2>&1 <<EOF



$n
$l


2
1
0
1
3P
2

EOF
    cp clist.out cfg.inp
    nonh > nonh_$n.log 2>&1
    # n >= 7: vary only the new layer (all at once diverges at n = 7)
    vary=all
    [ $n -ge 7 ] && vary=${n}s,${n}p,${n}d,${n}f,${n}g
    mchf > mchf_$n.log 2>&1 <<EOF
He,2.
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
EOF
    cp wfn.out he_$n.w
    cp cfg.inp he_$n.c
    cp 3Po.l he_$n.l 2>/dev/null || true
    echo "n=$n: $(grep -c '(' he_$n.c) CSFs, $(grep 'ETOTAL' mchf_$n.log | tail -1)"
done
