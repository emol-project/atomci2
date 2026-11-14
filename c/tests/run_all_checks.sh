#!/bin/bash
# run_all_checks.sh -- run every make check target in turn and report each
# (a convenience: each target can also be run alone, see tests/README.md).
# A target whose external program is absent prints SKIPPED and is reported as
# skipped, not passed.
#   PYTHON  a Python with numpy, scipy and PySCF (default python3)
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-python3}
cd "$here"
make -s || exit 1
status=0
for t in ${TARGETS:-check check-l0-exact check-l3 check-l3-addr check-l4 check-rank check-cfpgen check-pyref \
         check-s2m1 check-s2m2 check-s2m3 check-s2m4 check-s2m6 check-s2m6b check-s2m7 check-s2m9 check-s2m11 check-atsp check-library check-v5}; do
    log=$(mktemp)
    start=$(date +%s)
    if make -s "$t" PYREF_PYTHON="$py" PYTHON="$py" > "$log" 2>&1; then
        if grep -q '^SKIPPED ' "$log"; then
            echo "$t: skipped ($(( $(date +%s) - start )) s)"
            grep '^SKIPPED ' "$log" | sed 's/^/    /'
        else
            echo "$t: passed ($(( $(date +%s) - start )) s)"
        fi
    else
        echo "$t: FAILED ($(( $(date +%s) - start )) s)"
        tail -15 "$log" | sed 's/^/    /'
        status=1
    fi
    rm -f "$log"
done
exit $status
