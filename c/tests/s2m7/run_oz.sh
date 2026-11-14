#!/bin/bash
# run_oz.sh -- S2-M7 (Z2): the orbit-orbit Zeeman term of g (ac_ci_oz2e)
# against python/ozref.py, which evaluates the operator as it stands for
# single Gaussians in momentum space (analytic Fourier transforms, the
# transverse kernel 8 pi/q^2 (1 - q^ q^), no multipoles).
#
# Orbitals from tests/s2m6b/gauss_w.py (1s, 2p); acatsp on the same .w.
#   1s 2p 3P: Delta g(J = 1, 2) = <O_z>(M_L = 1) / 2 (pure LS, g_L = 1/2)
#   1s 2p 1P: Delta g(J = 1)    = <O_z>(M_L = 1)
#   2p2 3P:   0 (one p radial function)
# Passes at 1e-6 of the larger of |1P| and |3P/2| per exponent pair (the
# Gaussians on ATSP's grid give ~2e-7; the 3P value is a direct-exchange
# difference and can be 20 times smaller than either).
#   PYTHON  a Python with numpy (make passes PYREF_PYTHON)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
py=${PYTHON:-python3}
A=$repo/c/build/acatsp
T=$work/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
status=0
printf '%64s\n\n  1s( 1)  2p( 1)\n 2S1 2P1 3P\n*\n' "" > t.c
printf '%64s\n\n  1s( 1)  2p( 1)\n 2S1 2P1 1P\n*\n' "" > s.c
printf '%64s\n\n  2p( 2)\n 3P2\n*\n' "" > pp.c
dg() { "$A" "$T" G.w "$1" 4 alpha=137.036 noso oz2e | awk -v j="$2" '$1 == j && $2 == 1 { print $NF }'; }
for ab in "0.6 0.6" "2.0 0.5" "0.4 1.5"; do
    set -- $ab
    "$py" "$repo/c/python/ozref.py" $1 $2 > ref.txt
    python3 "$repo/c/tests/s2m6b/gauss_w.py" $1 $2 10 G.w > /dev/null
    rt=$(awk '/1s2p 3P/ { print $6 }' ref.txt)
    rs=$(awk '/1s2p 1P/ { print $6 }' ref.txt)
    "$py" - "$1" "$2" "$rt" "$rs" "$(dg t.c 2/2)" "$(dg t.c 4/2)" "$(dg s.c 2/2)" "$(dg pp.c 2/2)" <<'PY' || status=1
import sys
a, b = sys.argv[1:3]
rt, rs, t1, t2, s1, pp = map(float, sys.argv[3:])
# relative to the size of the case: the 3P value is a difference of direct and
# exchange terms of ~|1P| and can be much smaller
scale = max(abs(rt / 2), abs(rs))
rel = lambda x, y: abs(x - y) / scale
e = [rel(t1, rt / 2), rel(t2, rt / 2), rel(s1, rs)]
ok = max(e) < 1e-6 and abs(pp) < 1e-15
print(f"   {'ok  ' if ok else 'FAIL'} a={a} b={b}: 3P J=1,2 {t1:+.10e} {t2:+.10e} (ref {rt / 2:+.10e});"
      f" 1P {s1:+.10e} (ref {rs:+.10e}); max rel {max(e):.1e}; 2p2 3P {pp:+.1e}")
raise SystemExit(0 if ok else 1)
PY
done
echo "S2-M7 orbit-orbit Zeeman $([ $status = 0 ] && echo passed || echo FAILED)"
exit $status
