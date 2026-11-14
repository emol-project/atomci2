#!/bin/bash
# run_ss.sh -- V5 of H_SS: acfs "ss" (Coulomb + H_SO1 + H_SS tensor part) against the
# determinant CI of v5_so1.py --ss (int2e_ip1ip2) on the f sample (see run_v5.sh).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../lib/need.sh"
need_py numpy pyscf || exit 0
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
py=${PYTHON:-python3}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf '33*2 34*5\n' > "$tmp/f1"
printf '35*2 33*2 34*2\n' > "$tmp/c1"
printf '33*2 34*5\n33*2 34*4 21*1\n33*1 34*5 27*1\n33*2 34*4 20*1\n' > "$tmp/f4"
printf '35*2 33*2 34*2\n35*2 33*1 34*1 11*1 21*1\n35*2 33*1 34*1 10*1 20*1\n' > "$tmp/d4"
status=0
for c in ${CASES:-f1 c1 f4 d4}; do
    for opt in "--ss" "--ss --so2"; do
        echo "== $c $opt"
        "$py" "$here/v5_so1.py" "$repo/c/build/acfs" "$work/m0/build/CFPTBL" "$work/m5/f/prep" \
            "$tmp/$c" $opt | sed 's/^/   /' || status=1
    done
done
echo "V5 SS $([ $status = 0 ] && echo passed || echo FAILED)"
exit $status
