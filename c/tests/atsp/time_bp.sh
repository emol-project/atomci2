#!/bin/bash
# time_bp.sh -- wall time of the Breit-Pauli LSJ build on ATSP's Si example
# (e1.bp-5.c, 1360 CSFs, MCHF orbitals e1.14_5_3P.w), H_SO1 + H_SO2 + H_SS + H_OO
# and the g factors with GC2e and OZ2e, atspcompat, for each OMP_NUM_THREADS in
# $THREADS (default "1 8"). Every run must print the same output as the first
# (the parallel sums are added in a fixed order).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../lib/need.sh"
need_dir ATSP_SRC "the ATSP source tree, with examples/" || exit 0
repo=$(cd "$here/../../.." && pwd)
src=$ATSP_SRC
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ref=
for t in ${THREADS:-1 8}; do
    s=$(date +%s.%N)
    OMP_NUM_THREADS=$t "$A" "$T" "$src/examples/e1/e1.14_5_3P.w" "$src/examples/files_c/e1.bp-5.c" \
        6 so2 ss oo gc2e oz2e atspcompat > "$tmp/out_$t"
    e=$(date +%s.%N)
    printf 'threads %2d: %.1f s\n' "$t" "$(echo "$e - $s" | bc)"
    if [ -z "$ref" ]; then
        ref=$t
    else
        python3 - "$tmp/out_$ref" "$tmp/out_$t" <<'PY'
import re, sys
def lev(p):
    return [float(m.group(1)) for m in re.finditer(r"^ *\d+/2 +\d+ +(-?\d+\.\d+) ", open(p).read(), re.M)]
a, b = lev(sys.argv[1]), lev(sys.argv[2])
assert len(a) == len(b) and a, (len(a), len(b))
print(f"    {len(a)} levels, max |dE| against the first run {max(abs(x - y) for x, y in zip(a, b)):.1e}")
PY
        cmp -s "$tmp/out_$ref" "$tmp/out_$t" && echo "    output identical to the first run (g factors included)" \
            || echo "    output differs from the first run"
    fi
done
