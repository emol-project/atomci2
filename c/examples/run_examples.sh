#!/bin/bash
# run_examples.sh -- run the examples of the C tools and compare with the
# shipped outputs.
#
# Each example directory holds its inputs, a file "commands" and the outputs
# *.out. A line of "commands" is
#     <output file>   <tool> <arguments>
# e.g. "so1.out  acatsp - O.w cfg 9", run as  acatsp - O.w cfg 9 > so1.out
# ("-" for CFPTBL: the CFP table is generated in C, no file is needed). The
# tools are those of c/build. The example is copied to a scratch directory
# and run there; if it has a script prep.sh, that runs first (in the scratch
# directory, e.g. a Fortran ATOMCI run that writes the files acfs reads).
#
#   ./run_examples.sh [name ...]       run and compare every number with the
#                                      shipped output (relative 1e-7,
#                                      absolute 1e-9; text exactly). The
#                                      magnitudes are compared: the sign of a
#                                      matrix element between two levels
#                                      follows the phases of their
#                                      eigenvectors, which LAPACK may choose
#                                      either way
#   REF=1 ./run_examples.sh [name ...] overwrite the shipped outputs
#
#   CBUILD        the C tools (default ../build)
#   ATOMCI_HOME   the installed Fortran ATOMCI, for prep.sh (make install)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
CBUILD=${CBUILD:-$(cd "$here/../build" && pwd)}
export CBUILD
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
names=${*:-$(cd "$here" && for d in */commands; do echo "${d%/commands}"; done)}
status=0
for name in $names; do
    d=$here/$name
    [ -f "$d/commands" ] || { echo "no example $name" >&2; exit 1; }
    w=$tmp/$name
    cp -r "$d" "$w"
    if [ -f "$w/prep.sh" ]; then
        if ! (cd "$w" && EXAMPLE_DIR=$d bash prep.sh > prep.log 2>&1); then
            echo "$name: prep.sh failed"; tail -15 "$w/prep.log" | sed 's/^/    /'
            status=1; continue
        fi
    fi
    while read -r out tool args; do
        case $out in ''|'#'*) continue;; esac
        [ -x "$CBUILD/$tool" ] || { echo "$name/$out: $CBUILD/$tool not found (make in c/)"; status=1; continue; }
        start=$(date +%s.%N)
        # shellcheck disable=SC2086
        (cd "$w" && "$CBUILD/$tool" $args) > "$tmp/$out" 2> "$tmp/$out.err" ||
            { echo "$name/$out: $tool failed"; sed 's/^/    /' "$tmp/$out.err"; status=1; continue; }
        t=$(echo "$(date +%s.%N) - $start" | bc)
        if [ -n "${REF:-}" ]; then
            cp "$tmp/$out" "$d/$out"
            printf '%s/%s: written (%.1f s)\n' "$name" "$out" "$t"
            continue
        fi
        # compare token by token: numbers within the tolerance, text exactly
        if awk -v f="$tmp/$out" '
            function isnum(x) { return x ~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/ }
            function abs(x) { return x < 0 ? -x : x }
            { ref[NR] = $0 }
            END {
                n = 0
                while ((getline line < f) > 0) {
                    n++
                    if (n > NR) { print "    extra line " n ": " line; bad = 1; continue }
                    na = split(ref[n], a); nb = split(line, b)
                    if (na != nb) { print "    line " n " differs:\n    < " ref[n] "\n    > " line; bad = 1; continue }
                    for (i = 1; i <= na; i++) {
                        if (a[i] == b[i]) continue
                        if (isnum(a[i]) && isnum(b[i])) {
                            d = abs(abs(a[i]) - abs(b[i])); s = abs(a[i]) > abs(b[i]) ? abs(a[i]) : abs(b[i])
                            if (d <= 1e-9 || d <= 1e-7 * s) continue
                        }
                        print "    line " n " field " i ": " a[i] " -> " b[i]; bad = 1
                    }
                }
                if (n < NR) { print "    missing lines after " n; bad = 1 }
                exit bad
            }' "$d/$out" > "$tmp/diff"; then
            printf '%s/%s: ok (%.1f s)\n' "$name" "$out" "$t"
        else
            printf '%s/%s: DIFFERS (%.1f s)\n' "$name" "$out" "$t"
            head -20 "$tmp/diff"
            status=1
        fi
    done < "$d/commands"
done
exit $status
