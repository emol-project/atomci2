#!/bin/bash
# hfsrel_proto.sh -- S2-M11: acatsp hfsrel (ac_hfsrel) against an independent
# Python prototype (not distributed; the reference values below are its
# output) in the same orbitals and CI space, with the Breit product taken
# through the resolution of the identity as in C.
# 1. li532: the prototype's orbitals (cusp-constrained Laguerre, 4s 3p 2d,
#    b = 1.2, 2.5, 2.5) on ATSP's grid, every even-parity configuration of
#    three electrons (the complete CI); run with every orbital a slot (allslots) and
#    with the occupied orbitals only (the default). C interpolates the ATSP
#    table, the prototype uses its own grid: agreement ~1e-6.
# 2. lag16: the restricted space of the second order on the
#    bare-h shell functions of the basis (16, 6, 4), b = 1.2, 1.5, 2.0: the
#    .w holds the inner 1s 2s 3s 2p 3p 3d, lag= adds the other 12 s, 4 p, 3 d
#    analytically; up to two electrons outside the inner orbitals, at most
#    one in 6s .. 15s (the prototype prints 6 decimals).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
A=$repo/c/build/acatsp
T=${FTN_BUILD:-${AC_WORK:-$HOME/atomci_work}/m0/build}/CFPTBL
status=0
check() { # label, reference "name value ..." , acatsp arguments...
    local lab=$1 ref=$2
    shift 2
    local out
    out=$("$A" "$T" "$@" | grep '^hfsrel 2S: [EK]')
    echo "$out"
    echo "$out" | awk -v lab="$lab" -v ref="$ref" '
    BEGIN { n = split(ref, r, " "); for (i = 1; i < n; i += 2) want[r[i]] = r[i + 1]; tol = 1e-5; bad = 0; got = 0 }
    {
        for (i = 3; i < NF; i++)
            if ($i in want) {
                d = $(i + 1) - want[$i]; if (d < 0) d = -d
                got++
                if (d > tol) { printf "  %s: %s, prototype %s (diff %.2e)\n", $i, $(i + 1), want[$i], d; bad = 1 }
                if (d > worst) worst = d
            }
    }
    END {
        if (got != n / 2) { print "hfsrel_proto " lab ": expected " n / 2 " values, found " got; exit 1 }
        printf "hfsrel_proto (%s): largest difference from the prototype %.1e (tolerance %g): %s\n", lab, worst, tol, bad ? "FAILED" : "passed"
        exit bad
    }' || status=1
}
ref1="E -6.7518348720 A^(4) 10.87538700 K -3.3192004781 sep 7.9844841090 N2e -0.0159766729
Breit -0.5853221251 prod -16.4619570169 anom 9.0 s2 0.9034333652 A_AN/A4 -2.4945388188"
check "li532, all slots" "$ref1" "$here/li532.w" "$here/li532.cfg" noso hfsrel allslots
check "li532, occupied slots" "$ref1" "$here/li532.w" "$here/li532.cfg" noso hfsrel
ref2="E -7.4689426064 A^(4) 5.8597013 K -3.363644 sep 7.283574 N2e -0.025691
Breit -0.721204 prod -13.972100 anom 9.0 s2 14.913171 A_AN/A4 13.114106"
check "lag16, restricted space" "$ref2" "$here/lag16.w" "$here/lag16.cfg" noso hfsrel occslots \
    lag=0:16:1.2,1:6:1.5,2:4:2.0
exit $status
