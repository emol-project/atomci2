#!/bin/bash
# Phase 17 rule-based Q1: Q1_rule and leave-one-out, balanced (p_j), five ion/space runs in sequence.
HEFF="$(cd "$(dirname "$0")/../.." && pwd)"
export HEFF_WORK="${HEFF_WORK:-$HEFF/work}"; mkdir -p "$HEFF_WORK"; cd "$HEFF_WORK"
python() { "${PYTHON:-python3}" "$@"; }
ulimit -v 14300000
O=results/phase17
SETS="R R-n3 R-pf R-1s R-1d R-1g R-2s R-2p R-2d R-2f R-2g R-2h"
run() { local tag=$1 space=$2; shift 2
  [ -f $O/crule_$tag.json ] && return
  env "$@" python "$HEFF/scripts/heff/phase17c_scan.py" eval $space $O/crule_$tag.json $SETS >> $O/run_crule.log.$tag 2>&1 || exit 1
  git add $O/crule_$tag.json $O/run_crule.log.$tag && git commit -q -m "Phase 17 rule-based Q1: $tag

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}
run crnd_s8 besth_sall ION=crnd
run mnnd_s8 besth_sall ION=mnnd
run vnd_s8 besth_sall ION=vnd
run crnd_fam fam_d5f2g2h1 ION=crnd
run fend_fam fam_d5f2g2h1 C_N=6 A_RUNFMT='v15u_fam_d5f2g2h1_d6_{t}e_fend6' A_ION=fend
# Q4-b: Step 8 + one i (crti13); i singles and doubles in the rule, plus their leave-one-outs
SETS="$SETS R-1i R-2i" run crti1_s8i besth_salli ION=crti1
echo all done
