#!/bin/bash
# run_v5.sh -- S2-M4, V5: acfs (Coulomb + H_SO1, LSJ) against a determinant
# CI on PySCF integrals in the same GTO basis and orbitals (v5_so1.py).
#
# Uses the f sample ($AC_WORK/m5/f/prep; run c/tests/m5/run_m5.sh first):
#   f1  1s2 2s2 2p5
#   f4  1s2 2s2 2p5 + 2s2 2p4 p' + 2s 2p5 d + 2s2 2p4 p''
#       (p', p'': two virtual p MOs; d: a d MO; 306 determinants)
#   d4  1s2 2s2 2p2 + 1s2 2s 2p s' p' + 1s2 2s 2p s'' p'' (303 determinants):
#       four open shells, and double excitations ab -> cd over four
#       distinct orbitals, which ATSP's bpci leaves out of H_SO2 and H_SS
#       (tests/atsp/run_atsp_bp.sh, Si)
# Each case runs with H_SO1 and with H_SO1 + H_SO2 (--so2: PySCF's
# int2e_p1vxp1 as in libhexpr2, equal to GAMESS HSO2). Passes when every
# acfs level, counted 2J+1 times, matches the determinant spectrum to 1e-8,
# python/gc2eref.py (S2-M7, Z1) checks the two-electron gauge correction of
# g: the operator with grad(1/r12) (PySCF int1e_grids_ip) against the form
# integrated by parts in electron 1 (int1e_grids), for 1s, 2s, 2p with p'
# and with d; passes at 1e-10 relative.
# v5_gc2e.py (S2-M7, Z3) compares acfs's Delta g of the two-electron gauge
# correction ("gc2e") with <O_z>/M of the gc2eref operator in a determinant
# CI (H = h + V_ee + H_SO1, the M = J state of each level): f1 with the
# spin-same-orbit and the spin-other-orbit parts separately, and
# 1s2 2s2 2p2 + 1s2 2s 2p s' p' with both; passes at 1e-10.
# f1 and d4 also run with --contact (S2-M6b): the two-electron Darwin and
# spin-spin contact terms in their spin-dependent form, -pi alpha^2 delta -
# (8 pi/3) alpha^2 s1.s2 delta, on PySCF grid integrals of delta(r12),
# against acfs "contact" (the spin-free pi alpha^2 delta added to R^k);
# and when the independent operator of python/so2ref.py (S2-M5 derivation)
# equals int2e_p1vxp1 element by element (1s, 2s, 2p, p', d).
#   PYTHON  a Python with PySCF (make passes PYREF_PYTHON)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../lib/need.sh"
need_py numpy scipy pyscf || exit 0
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
py=${PYTHON:-python3}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf '33*2 34*5\n' > "$tmp/f1"
printf '33*2 34*5\n33*2 34*4 21*1\n33*1 34*5 27*1\n33*2 34*4 20*1\n' > "$tmp/f4"
printf '35*2 33*2 34*2\n35*2 33*1 34*1 11*1 21*1\n35*2 33*1 34*1 10*1 20*1\n' > "$tmp/d4"
status=0
for c in f1 f4 d4; do
    for opt in "" --so2; do
        echo "== $c $opt"
        "$py" "$here/v5_so1.py" "$repo/c/build/acfs" "$work/m0/build/CFPTBL" "$work/m5/f/prep" \
            "$tmp/$c" $opt | sed 's/^/   /' || status=1
    done
done
for c in f1 d4; do
    echo "== $c --contact"
    "$py" "$here/v5_so1.py" "$repo/c/build/acfs" "$work/m0/build/CFPTBL" "$work/m5/f/prep" \
        "$tmp/$c" --contact | sed 's/^/   /' || status=1
done
echo "== S2-M7 Z1: the two-electron gauge correction (python/gc2eref.py), the operator"
echo "   as it stands against its integrated-by-parts form, on a PySCF grid"
for orbs in "0:13 0:12 1:11 1:10" "0:13 0:12 1:11 2:6"; do
    "$py" "$repo/c/python/gc2eref.py" "$work/m5/f/prep" $orbs | sed 's/^/   /' || status=1
done
echo "== S2-M7 Z3: Delta g of the two-electron gauge correction, acfs against a"
echo "   determinant CI with the operator of python/gc2eref.py"
printf '35*2 33*2 34*2\n35*2 33*1 34*1 11*1 21*1\n' > "$tmp/d2"
for cp in "f1 1" "f1 2" "d2 3"; do
    set -- $cp
    "$py" "$here/v5_gc2e.py" "$repo/c/build/acfs" "$work/m0/build/CFPTBL" "$work/m5/f/prep" \
        "$tmp/$1" $2 | grep -E "^   J=|Z3" | sed 's/^/   /' || status=1
done
echo "== the two-electron spin-orbit operator of python/so2ref.py against int2e_p1vxp1"
"$py" "$here/so2_check.py" "$work/m5/f/prep" 35 33 34 21 27 | sed 's/^/   /' || status=1
echo "== S2-M7 Z3: Delta g of the one-electron gauge correction, l -> l +- 2 included (run_gc1.sh)"
PYTHON="$py" bash "$here/run_gc1.sh" | sed 's/^/   /' || status=1
echo "== H_SS (tensor part) against the determinant CI with int2e_ip1ip2 (run_ss.sh)"
PYTHON="$py" bash "$here/run_ss.sh" | sed 's/^/   /' || status=1
echo "== the two-body recoil term of g (ac_ci_rec2) against the determinant CI (run_rec.sh)"
PYTHON="$py" bash "$here/run_rec.sh" | sed 's/^/   /' || status=1
echo "== S2-M8 P3: hyperfine and transition operators (acfs hfs, trans) against the determinant CI (run_props.sh)"
PYTHON="$py" bash "$here/run_props.sh" | sed 's/^/   /' || status=1
echo "V5 $([ $status = 0 ] && echo passed || echo FAILED)"
exit $status
