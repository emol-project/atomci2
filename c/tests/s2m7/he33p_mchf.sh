#!/bin/bash
# he33p_mchf.sh -- MCHF orbitals for He 1s3p 3P (the second 3P root), state-
# averaged with 2 3P, on the n complete active spaces of he23p_mchf.sh for n in
# $NS (default 4 5). n = 4 starts from hydrogenic estimates, n >= 5 from he3_<n-1>.w.
# Writes he3_<n>.w and he3_<n>.c in the work directory.
set -euo pipefail
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$(dirname "$0")/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" mchf nonh || exit 0
w=${HE_WORK:-${AC_WORK:-$HOME/atomci_work}/he23p}
cd "$w"
for n in ${NS:-4 5}; do
    cp he_$n.c cfg.inp
    rm -f wfn.inp
    [ $n -ge 5 ] && cp he3_$((n - 1)).w wfn.inp
    nonh > nonh3_$n.log 2>&1
    mchf > mchf3_$n.log 2>&1 <<EOF
He,2.
1,2
all
1s,2p,3p
y
n
y
n
.f.,1.e-12,1.e-10
n
500,0
y
EOF
    cp wfn.out he3_$n.w
    cp cfg.inp he3_$n.c
    echo "n=$n: $(grep 'ETOTAL' mchf3_$n.log | tail -1)"
done
