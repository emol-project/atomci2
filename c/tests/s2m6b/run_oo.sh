#!/bin/bash
# run_oo.sh -- S2-M6b: the orbit-orbit operator (ac_ci_oo) against exact
# values for Gaussian orbitals (gauss_w.py), through acatsp.
#
#   1s 2p 3P: the closed form of gauss_w.py (transverse projection in
#     momentum space), three exponent pairs; passes at 1e-6 relative
#   1s 2s 3S: exactly 0 (the s-s transition current is longitudinal); the
#     current and divergence parts cancel; passes below 1e-12 hartree
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
A=$repo/c/build/acatsp
T=$work/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
status=0
lowest() { "$A" "$T" G.w "$1" 1 alpha=137.036 noso "${@:2}" | awk '/^ ?[0-9]+\/2 +1 / { print $3; exit }'; }
for ab in "0.6 0.6" "2.0 0.5" "0.4 1.5"; do
    set -- $ab
    ex=$(python3 "$here/gauss_w.py" $1 $2 10 G.w | awk '/1s2p/ { print $NF }')
    printf '%64s\n\n  1s( 1)  2p( 1)\n 2S1 2P1 3P\n*\n' "" > sp.c
    printf '%64s\n\n  1s( 1)  2s( 1)\n 2S1 2S1 3S\n*\n' "" > ss.c
    esp=$(python3 -c "print($(lowest sp.c oo) - ($(lowest sp.c)))")
    ess=$(python3 -c "print($(lowest ss.c oo) - ($(lowest ss.c)))")
    python3 - "$1" "$2" "$ex" "$esp" "$ess" <<'PY' || status=1
import sys
a, b, ex, esp, ess = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:])
rel = abs(esp - ex) / abs(ex)
ok1, ok2 = rel < 1e-6, abs(ess) < 1e-12
print(f"   {'ok  ' if ok1 else 'FAIL'} a={a} b={b}: 1s2p 3P  exact {ex:+.12e}  C {esp:+.12e}  rel {rel:.1e}")
print(f"   {'ok  ' if ok2 else 'FAIL'} a={a} b={b}: 1s2s 3S  C {ess:+.1e} (exact 0)")
raise SystemExit(0 if ok1 and ok2 else 1)
PY
done
echo "S2-M6b orbit-orbit $([ $status = 0 ] && echo passed || echo FAILED)"
exit $status
