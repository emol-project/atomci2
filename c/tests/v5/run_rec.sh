#!/bin/bash
# run_rec.sh -- the two-body recoil term (acfs's rec2, ac_ci_rec2) against
# <O_z>/M, O_z = sum_{i != j} (r_i x p_j)_z, in a determinant CI on PySCF's
# integrals (v5_gc2e.py --rec). f1 is a single configuration (the closed
# formula's case), f4 adds 2s 2p^5 d and 2s^2 2p^4 d, s', p' admixtures, so
# off-diagonal two-body elements between configurations enter.
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
printf '33*2 34*5\n33*2 34*4 21*1\n33*1 34*5 27*1\n33*2 34*4 20*1\n' > "$tmp/f4"
status=0
for c in f1 f4; do
    echo "== $c"
    "$py" "$here/v5_gc2e.py" "$repo/c/build/acfs" "$work/m0/build/CFPTBL" "$work/m5/f/prep" \
        "$tmp/$c" --rec | grep -E "^   J=|Z3" | sed 's/^/   /' || status=1
done
exit $status
