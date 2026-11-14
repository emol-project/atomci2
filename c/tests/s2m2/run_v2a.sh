#!/bin/bash
# run_v2a.sh -- S2-M2, check V2a: the stage-2 CI (closed shells explicit,
# general-form engine) on the Fortran's own CSFs and integrals.
#
# Uses the M5 preparations ($AC_WORK/m5/{boron,f}/prep: fort.15 tape and
# fort.16 NFTRI; run c/tests/m5/run_m5.sh first). acci builds H twice:
# with the engine, and from the Fortran energy expression on the tape (the
# SMAT path, with its analytic closed-shell terms) times the same integrals.
# Passes when the two H agree element by element to 1e-10 and their lowest
# eigenvalues to 1e-11.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
status=0
for s in boron f; do
    p=$work/m5/$s/prep
    out=$("$repo/c/build/acci" "$work/m0/build/CFPTBL" "$p/fort.15" "$p/fort.16" 1 cmp)
    echo "== $s"
    echo "$out" | sed 's/^/   /'
    verdict=$(echo "$out" | python3 -c '
import re, sys
t = sys.stdin.read()
dh = float(re.search(r"max \|H_engine - H_tape\| = (\S+)", t).group(1))
e = float(re.search(r"^E1 = (\S+)", t, re.M).group(1))
et = float(re.search(r"^E1\(tape\) = (\S+)", t, re.M).group(1))
print("ok" if dh < 1e-10 and abs(e - et) < 1e-11 else "FAIL", f"|E1 - E1(tape)| = {abs(e - et):.1e}")')
    echo "   $verdict"
    case $verdict in FAIL*) status=1 ;; esac
done
echo "V2a $([ $status = 0 ] && echo passed || echo FAILED)"
exit $status
