#!/bin/bash
# he33p_g.sh -- C's Delta g terms of the two lowest He 3P roots (2 3P, 3 3P) on the
# state-averaged MCHF orbitals of he33p_mchf.sh; full, then SSO only.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
cd "${HE_WORK:-${AC_WORK:-$HOME/atomci_work}/he23p}"
for n in ${NS:-4 5}; do
    echo "== n=$n"
    "$A" "$T" he3_$n.w he3_$n.c 2 noso gc2e oz2e | grep -E '^ *[24]/2'
    echo "-- SSO only"
    "$A" "$T" he3_$n.w he3_$n.c 2 noso gc2e gc2epart=1 | grep -E '^ *[24]/2'
done
