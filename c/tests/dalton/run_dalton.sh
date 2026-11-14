#!/bin/bash
# run_dalton.sh -- S2-M7 (Z7), report only: g shifts of S-state atoms (Li 2S, N 4S, Na 2S)
# from Dalton's ROHF g tensor (RMC, GC1, GC2) against C on ATSP HF orbitals.
# Dalton's GC2 default (AVEDSO, DALTON/rsp/rspg.F) contracts the exact two-electron
# integrals with the inactive and active densities, which is exact for high-spin ROHF,
# unlike ORCA's effective-charge DSO (Z4). Dalton uses an uncontracted aug-cc-pCVQZ
# (uncontract.py). Dalton prints these terms in whole ppm only; rspg_digits.patch adds
# one line "@GDIGITS RMC GC1 GC2" with full digits (print only).
#   DALTON_BUILD  Dalton build directory (default $HOME/dalton_build; build_dalton.sh,
#                 with rspg_digits.patch applied to the source)
#   DALTON_SRC    Dalton source tree, for basis/aug-cc-pCVQZ (needed on the first run)
#   ATSP_BIN      ATSP binaries (default $HOME/atsp_work/atsp/bin)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
D=${DALTON_BUILD:-$HOME/dalton_build}
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_exe "$D/dalton" "Dalton: set DALTON_BUILD to its build directory" || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
W=${AC_WORK:-$HOME/atomci_work}/dalton
mkdir -p "$W/basis" && cd "$W"
if [ ! -s basis/unc-aug-cc-pCVQZ ]; then
    need_dir DALTON_SRC "the Dalton source tree, for basis/aug-cc-pCVQZ" || exit 0
    python3 -I "$here/uncontract.py" \
    "$DALTON_SRC/basis/aug-cc-pCVQZ" basis/unc-aug-cc-pCVQZ 3 7 11 > /dev/null
fi
run() { # name Z doubly singly term closed open cfg 2J
    local n=$1
    cat > $n.dal <<EOF
**DALTON INPUT
.RUN RESPONSE
**WAVE FUNCTIONS
.HF
*SCF INPUT
.DOUBLY OCCUPIED
 $3
.SINGLY OCCUPIED
 $4
**RESPONSE
*ESR
.G-TENSOR
**END OF DALTON INPUT
EOF
    cat > $n.mol <<EOF
BASIS
unc-aug-cc-pCVQZ
$n atom
uncontracted aug-cc-pCVQZ
Atomtypes=1 Generators=3 X Y Z
Charge=$2.0 Atoms=1
$n 0.0 0.0 0.0
EOF
    grep -q "@GDIGITS" ${n}_$n.out 2>/dev/null || \
        "$D/dalton" -noarch -mb 4000 -b "$W/basis" -o ${n}_$n.out $n $n > /dev/null 2>&1 || true
    mkdir -p hf_$n && cd hf_$n
    [ -s $n.w ] || { hf > hf.out 2>&1 <<S
$n,$5,$2.
$6
$7
all
y
y
n
n
S
    cp wfn.out $n.w; }
    echo "$8" > cfg
    local c=$($A $T $n.w cfg 2 alpha=137.035999084 gc2e gc2epart=1 | awk -v j="$9/2" '$1 == j && $2 == 1 { print $7, $8, $10; exit }')
    local c2=$($A $T $n.w cfg 2 alpha=137.035999084 gc2e | awk -v j="$9/2" '$1 == j && $2 == 1 { print $10; exit }')
    cd ..
    local d=$(awk '/@GDIGITS/ { print $5, $6, $7 }' ${n}_$n.out)
    local e=$(awk '/Final HF energy/ { print $NF; exit }' ${n}_$n.out)
    python3 - "$n" "$c $c2" "$d" "$e" <<'PY'
import sys
n, c, d, e = sys.argv[1:]
if not d:
    print(f"{n}: Dalton failed"); sys.exit()
ge2 = 2.00231930436256 / 2
rmc, gc, sso, gc2e = (ge2 * float(v) for v in c.split())
dr, d1, d2 = map(float, d.split())
print(f"{n:2} (E_HF Dalton {e})   C on ATSP HF x g_e/2  /  Dalton ROHF  (ratio)")
for k, a, b in (("RMC", rmc, dr), ("GC1", gc, d1), ("GC2", gc2e, d2), ("total", rmc + gc + gc2e, dr + d1 + d2)):
    print(f"   {k:5} {a:+.6e} / {b:+.6e}  ({a / b:.5f})")
print(f"   C's GC2 = SSO {sso:+.4e} + SOO {gc2e - sso:+.4e}")
PY
}
run Li 3 "1 0 0 0 0 0 0 0" "1 0 0 0 0 0 0 0" 2S "  1s" "2s(1)" "1s(2) 2s(1)" 1
run N 7 "2 0 0 0 0 0 0 0" "0 1 1 0 1 0 0 0" 4S "  1s  2s" "2p(3)" "1s(2) 2s(2) 2p(3)" 3
run Na 11 "2 1 1 0 1 0 0 0" "1 0 0 0 0 0 0 0" 2S "  1s  2s  2p" "3s(1)" "1s(2) 2s(2) 2p(6) 3s(1)" 1
