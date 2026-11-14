#!/bin/bash
# he23p_g.sh -- C's Delta g terms of He 2 3P_J on the MCHF orbitals of
# he23p_mchf.sh (run that first), for each n.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
w=${HE_WORK:-${AC_WORK:-$HOME/atomci_work}/he23p}
cd "$w"
for n in ${NS:-3 4 5}; do
    echo "== n=$n"
    "$A" "$T" he_$n.w he_$n.c 1 noso gc2e oz2e
    echo "-- SSO only"
    "$A" "$T" he_$n.w he_$n.c 1 noso gc2e gc2epart=1 | grep -E '^ *[0-9]' || true
done
