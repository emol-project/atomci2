#!/bin/bash
# run_v3.sh -- S2-M3: radial integrals by numerical quadrature (ac_radial).
#
# Uses the M5 preparations ($AC_WORK/m5/{boron,f}/prep: fort.18 basis,
# fort.21 MO coefficients, fort.16 NFTRI; run c/tests/m5/run_m5.sh first).
# boron has an STO basis, f a contracted GTO basis.
#
# acrad checks, and fails on: the MO overlap (numeric and analytic) against
# 1; h against the analytic basis-pair formula and the Fortran NFTRI; every
# R^k against NFTRI (1e-8, the Fortran's own accuracy); <r^-3> against the
# analytic formula; hydrogenic 2p <r^-3> = Z^3/24.
# Then 30 R^k with the largest |C - F| and 30 random ones are recomputed
# independently (STO: exact rational closed form; GTO: incomplete gamma
# plus adaptive quadrature) and must agree with C to 1e-11.
#
# PYTHON must have numpy and scipy (make passes PYREF_PYTHON).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
py=${PYTHON:-python3}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
status=0
for s in boron f; do
    p=$work/m5/$s/prep
    echo "== $s"
    "$repo/c/build/acrad" "$p/fort.18" "$p/fort.21" "$p/fort.16" 0.005 100 "$tmp/rk.txt" \
        | sed 's/^/   /' || status=1
    if [ $s = boron ]; then ref=rk_sto_exact.py; else ref=rk_gto_quad.py; fi
    "$py" "$here/$ref" "$p/fort.18" "$p/fort.21" "$tmp/rk.txt" 30 30 | sed 's/^/   /' || status=1
done
echo "V3 $([ $status = 0 ] && echo passed || echo FAILED)"
exit $status
