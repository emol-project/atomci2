#!/bin/bash
# run_gc1.sh -- Z3 for the one-electron gauge correction: acfs's dg_GC against <O_z>/M in
# a determinant CI (v5_gc2e.py --gc1). f4 mixes 2s 2p^5 d into 2s^2 2p^5, so the rank-2
# part's s-d elements (l -> l +- 2) enter; f1 is a single configuration.
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
        "$tmp/$c" --gc1 | grep -E "^   J=|Z3" | sed 's/^/   /' || status=1
done
exit $status
