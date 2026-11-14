#!/bin/bash
# Phase 18 Step 2: exact H_eff blocks of the Co3+ correction-chain spaces already analysed by the queue (as
# phase17q11_dump.py for Fe2+): phase17.py analyze with HEFF_DUMP, tag q18_<ion>_<space> (the queue's 18_* results are
# untouched) -> results/phase18/heff_dump/q18_<ion>_<space>_<space>_iter.npz. Reads only results/<run>.npz.
# One space at a time under the queue memory reservation (NEED = the queue's analysis estimate + margin).
# usage: phase18_codump.sh ion:space:need_mb ...
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
O=results/phase18; D=$O/heff_dump; R=$O/queue_reserve_mb; S=$O/queue_running_mb; L=$O/phase18_codump.log
mkdir -p $D
trap 'rm -f $R' EXIT
rc_all=0
for x in "$@"; do
  IFS=: read ion sp need <<< "$x"
  echo $need > $R
  echo "$(date +%T) $ion $sp reserved $need MB, waiting" >> $L
  while :; do r=$(cut -d' ' -f1 $S); [ "$r" -le $((13800 - need)) ] && break; sleep 30; done
  echo "$(date +%T) begin $ion $sp" >> $L
  env ION=$ion HEFF_DUMP=$D HEFF_SOLVER=iter HEFF_SPARSE_ABOVE=500 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python "$HEFF/scripts/decks/peakmem.py" "ana q18_${ion}_${sp}" python "$HEFF/scripts/heff/phase17.py" analyze q18_${ion}_${sp} $sp \
    > $D/q18_${ion}_${sp}.log 2>&1; rc=$?
  f=$D/q18_${ion}_${sp}_${sp}_iter.npz
  [ $rc -eq 0 ] && [ -s $f ] && st=ok || { st=MISSING; rc_all=1; }
  echo "$(date +%T) end $ion $sp rc $rc dump $st $(grep PEAKMEM $D/q18_${ion}_${sp}.log | tail -1)" >> $L
  rm -f $R
done
exit $rc_all
