#!/bin/bash
# lagfock.sh -- acatsp lagfock: the orbitals added by lag= are diagonalized
# on a screened operator instead of the bare h. They span the same space, so
# a configuration list that holds every distribution of its electrons over
# the added orbitals gives the same levels and g factors.
# Ti on the orbitals of examples/ti, lag=0:5:2.0,1:3:2.0 (4 s and 3 p added):
# 3d2 with two electrons in 4s and the added orbitals in every way. The run
# with lagfock must give the levels of the run without (1e-9 hartree, 1e-8
# in g_J) on different orbitals (<h> of the first added s differs).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
A=$repo/c/build/acatsp
T=${FTN_BUILD:-${AC_WORK:-$HOME/atomci_work}/m0/build}/CFPTBL
[ -r "$T" ] || T=-
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
core="1s(2) 2s(2) 2p(6) 3s(2) 3p(6) 3d(2)"
s=(4s 5s 6s 7s 8s)
p=(4p 5p 6p)
{
    for ((i = 0; i < ${#s[@]}; i++)); do
        echo "$core ${s[i]}(2)"
        for ((j = i + 1; j < ${#s[@]}; j++)); do echo "$core ${s[i]}(1) ${s[j]}(1)"; done
    done
    for ((i = 0; i < ${#p[@]}; i++)); do
        echo "$core ${p[i]}(2)"
        for ((j = i + 1; j < ${#p[@]}; j++)); do echo "$core ${p[i]}(1) ${p[j]}(1)"; done
    done
} > "$w/cfg"
lag=lag=0:5:2.0,1:3:2.0
"$A" "$T" "$repo/c/examples/ti/Ti.w" "$w/cfg" 6 $lag > "$w/a.out"
"$A" "$T" "$repo/c/examples/ti/Ti.w" "$w/cfg" 6 $lag lagfock > "$w/b.out"
grep '^lagfock:' "$w/b.out"
awk '
function abs(x) { return x < 0 ? -x : x }
FNR == 1 { f++ }
/^orbital 5s / { h[f] = $5 }
/^ *[0-9]+\/2 +[0-9]+ +-[0-9]+\.[0-9]+ / {
    k = $1 " " $2
    if (f == 1) { e[k] = $3; t[k] = $4; g[k] = $6; n1++ }
    else {
        n2++
        if (!(k in e)) { bad = 1; next }
        d = abs($3 - e[k]); if (d > de) de = d
        d = abs($6 - g[k]); if (d > dg) dg = d
        if ($4 != t[k]) nt++
    }
}
END {
    ok = n1 > 0 && n1 == n2 && !bad && de < 1e-9 && dg < 1e-8 && nt == 0 && abs(h[1] - h[2]) > 1e-3
    printf "lagfock: %d levels, largest differences %.1e hartree, %.1e in g_J, %d term labels; <h> of 5s %s and %s: %s\n",
        n2, de, dg, nt, h[1], h[2], ok ? "passed" : "FAILED"
    exit !ok
}' "$w/a.out" "$w/b.out"
