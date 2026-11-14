#!/bin/bash
# run_epr.sh -- S2-M7 (Z4), report only: ORCA's g-shifts of S-state atoms (CASSCF over the
# open shell, i.e. ROHF-like, and UHF with the RMC/DSO breakdown) against C on ATSP HF orbitals.
# ORCA's two-electron DSO uses an effective-charge (Zeff) approximation, not the exact
# operator; the comparison checks the one-electron terms and the g_e/2 convention.
#   ORCA_DIR  ORCA 6 directory (skipped when unset); serial runs
#   ATSP_BIN  ATSP binaries (default $HOME/atsp_work/atsp/bin)
O=${ORCA_DIR:-}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=$O:${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
export LD_LIBRARY_PATH=$O:${LD_LIBRARY_PATH:-}
. "$here/../lib/need.sh"
need_dir ORCA_DIR "the ORCA 6 directory" || exit 0
need_exe "$ORCA_DIR/orca" "ORCA 6" || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; cd $W
run() { # name element mult nel norb term Z closed open cfg
  local n=$1
  cat > ${n}_cas.inp <<EOF
! aug-cc-pwCVQZ DecontractBas TightSCF NoFrozenCore NoTRAH
%casscf
  nel $4
  norb $5
  mult $3
  rel
    dosoc true
    gtensor true
  end
end
%eprnmr
  ori CenterOfElCharge
end
* xyz 0 $3
$2 0.0 0.0 0.0
*
EOF
  cat > ${n}_uhf.inp <<EOF
! UHF aug-cc-pwCVQZ DecontractBas TightSCF NoFrozenCore NoTRAH
%eprnmr
  gtensor true
  gtensor_1el2el true
  ori CenterOfElCharge
end
* xyz 0 $3
$2 0.0 0.0 0.0
*
EOF
  [ -s ${n}_cas.out ] && grep -q "TERMINATED NORMALLY" ${n}_cas.out || $O/orca ${n}_cas.inp > ${n}_cas.out 2>&1
  [ -s ${n}_uhf.out ] && grep -q "TERMINATED NORMALLY" ${n}_uhf.out || $O/orca ${n}_uhf.inp > ${n}_uhf.out 2>&1
  mkdir -p hf_$n && cd hf_$n
  hf > hf.out 2>&1 <<S
$n,$6,$7.
$8
$9
all
y
y
n
n
S
  cp wfn.out $n.w
  echo "${10}" > cfg
  local tj=$(( $3 - 1 ))
  local c=$($A $T $n.w cfg 2 alpha=137.035999084 gc2e | awk -v j="$tj/2" '$1 == j && $2 == 1 { print $7, $8, $9, $10 }')
  local c1=$($A $T $n.w cfg 2 alpha=137.035999084 gc2e gc2epart=1 | awk -v j="$tj/2" '$1 == j && $2 == 1 { print $10 }')
  cd ..
  local cas=$(awk '/ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN/ {f=1} f && /g-shifts:/ {getline; print $NF; exit}' ${n}_cas.out)
  local uhf=$(awk '/Breakdown of the contributions/ {f=1} f && /gRMC/ {r=$2} f && /gDSO\(1el\)/ {d1=$2} f && /gDSO\(2el\)/ {d2=$2; print r, d1, d2; exit}' ${n}_uhf.out)
  python3 - "$n" "$cas" "$c" "$c1" "$uhf" <<'PY'
import sys
n, cas, c, c1, uhf = sys.argv[1:]
ge2 = 2.00231930436256 / 2
rmc, gc, ormc, gc2e = map(float, c.split())
sso = float(c1)
r_u, d1_u, d2_u = map(float, uhf.split())
tot = ge2 * (rmc + gc + gc2e)
print(f"{n:3} C (ROHF-like, x g_e/2): RMC {ge2 * rmc:+.4e} GC1e {ge2 * gc:+.4e} GC2e {ge2 * gc2e:+.4e}"
      f" (SSO {ge2 * sso:+.4e}, SOO {ge2 * (gc2e - sso):+.4e})  total {tot:+.4e}")
print(f"    ORCA CASSCF g-shift (RMC + DSO with Zeff 2e approx.) {float(cas):+.4e}   ratio C/ORCA {tot / float(cas):.3f}")
print(f"    ORCA UHF: RMC {r_u:+.4e} DSO(1el) {d1_u:+.4e} DSO(2el, Zeff) {d2_u:+.4e} total {r_u + d1_u + d2_u:+.4e}")
PY
}
run Li Li 2 1 1 2S 3 "  1s" "2s(1)" "1s(2) 2s(1)"
run Na Na 2 1 1 2S 11 "  1s  2s  2p" "3s(1)" "1s(2) 2s(2) 2p(6) 3s(1)"
run N N 4 3 3 4S 7 "  1s  2s" "2p(3)" "1s(2) 2s(2) 2p(3)"
