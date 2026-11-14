#!/bin/bash
# run_props.sh -- S2-M8 (P3): acfs "hfs" and "trans" (ac_lsj_hfs_reduced,
# ac_lsj_trans_reduced), with "zoff h6" and the isotope-shift parameters of
# "is" (S2-M10, ac_lsj_is_level), against a determinant CI on PySCF's integrals
# (v5_props.py), in the f sample of run_v5.sh ($AC_WORK/m5/f/prep):
#   f4  1s2 2s2 2p5 + 2s2 2p4 p' + 2s 2p5 d + 2s2 2p4 p''   (odd; 2s open
#       shells for the contact term, s-d and p-p' pairs)
#   d4  1s2 2s2 2p2 + 1s2 2s 2p s' p' + 1s2 2s 2p s'' p''    (four open shells)
#   t5  1s2 2s2 2p5 + 2s2 2p4 p' (odd) and 2s 2p6 + 2s2 2p4 s' + 2s 2p5 p'
#       + 2s2 2p4 d (even): E1 between the two parities
#   PYTHON  a Python with PySCF
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../lib/need.sh"
need_py numpy pyscf || exit 0
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
py=${PYTHON:-python3}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf '33*2 34*5\n33*2 34*4 21*1\n33*1 34*5 27*1\n33*2 34*4 20*1\n' > "$tmp/f4"
printf '35*2 33*2 34*2\n35*2 33*1 34*1 11*1 21*1\n35*2 33*1 34*1 10*1 20*1\n' > "$tmp/d4"
printf '33*2 34*5\n33*2 34*4 21*1\n33*1 34*6\n33*2 34*4 11*1\n33*1 34*5 21*1\n33*2 34*4 27*1\n' > "$tmp/t5"
status=0
for c in f4 d4 t5; do
    echo "== $c"
    "$py" "$here/v5_props.py" "$repo/c/build/acfs" "$work/m0/build/CFPTBL" "$work/m5/f/prep" \
        "$tmp/$c" | sed 's/^/   /' || status=1
done
exit $status
