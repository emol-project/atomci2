#!/bin/bash
# Phase 18, 10 Nov B: plain-basis reference sub-spaces of fam_d5f2g2h1 for the opened-d test (exact H_eff of the stored
# fam H, phase17c_scan eval), Co3+ (cond) then Fe2+ (fend), one at a time (~9.3 GB):
#   X-16s,15p,11h       the famdc_nsph rule in the plain basis (5 % check against fam)
#   X-16s,15p,11h,14d   the same without the plain set's most compact virtual d (<r> 0.84 bohr), i.e. the d region the
#                       opened-d space famdc_nsph_dd7 covers
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; L=$O/phase18_dcref.log
rc_all=0
for ion in cond fend; do
  echo "$(date +%T) begin $ion" >> $L
  env C_N=6 A_RUNFMT="v15u_fam_d5f2g2h1_d6_{t}e_${ion}6" A_ION=$ion SCAN_DUMP=$O/heff_dump/q18_${ion}_fam_dcref.npz \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HEFF_SOLVER=iter \
    python "$HEFF/scripts/decks/peakmem.py" "q18 $ion fam dcref" python "$HEFF/scripts/heff/phase17c_scan.py" eval fam_d5f2g2h1 \
    $O/q18_${ion}_fam_dcref.json X-16s,15p,11h X-16s,15p,11h,14d >> $L 2>&1; rc=$?
  echo "$(date +%T) end $ion rc $rc" >> $L
  [ $rc -ne 0 ] && rc_all=1
done
exit $rc_all
