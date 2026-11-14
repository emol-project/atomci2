#!/bin/bash
# run_trace.sh -- M0: run the boron and f samples with the plain and the
# traced atomci from build_trace.sh, each in its own clean work directory
# (INSTALL.md "Verifying the build"), and print the energies.
#
#   AC_WORK   work root (default: $HOME/atomci_work); this uses $AC_WORK/m0
#
# Layout under $AC_WORK/run:
#   plain-boron/ trace-boron/ plain-f/ trace-f/
#   trace-*/trace/   the recorded oracle (see trace_wrap.c)
# All fort.* files are kept: they include the NFTE tapes needed for L4/L5.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}/m0
build=$work/build
run=$work/run

rm -rf "$run"
mkdir -p "$run"

for kind in plain trace; do
    if [ "$kind" = plain ]; then bin=$build/atomci.plain; else bin=$build/atomci; fi

    d=$run/$kind-boron
    mkdir -p "$d/trace"
    (
        cd "$d"
        ln -sf "$build/CFPTBL" ./CFPTBL
        AC_TRACE_DIR=$d/trace "$bin" < "$repo/samples/native/boron.stdin" \
            > boron.output 2> boron.err
    )

    d=$run/$kind-f
    mkdir -p "$d/trace"
    (
        cd "$d"
        ln -sf "$build/CFPTBL" ./CFPTBL
        cp "$repo/samples/native/f.bas" .
        "$build/atgen" < "$repo/samples/native/f.atg"
        AC_TRACE_DIR=$d/trace "$bin" < "$repo/samples/native/f.stdin" \
            > f.output 2> f.err
    )
done

# The outputs contain carriage-control bytes, so grep needs -a (text mode);
# "|| true" keeps a missing line from aborting the summary under set -e.
echo "== energies (golden: boron ET = -24.6461953967, f SD-CI = -99.641223)"
for d in "$run"/*-boron; do
    printf '%-14s ' "$(basename "$d")"
    { grep -a -h "ET *=" "$d/boron.output" || true; } | tail -1
done
# f.stdin has three passes; the golden value is the first pass's SD-CI.
for d in "$run"/*-f; do
    printf '%-14s ' "$(basename "$d")"
    { grep -a -h "SD-CI ENERGY  *=" "$d/f.output" || true; } | head -1
done

# The wrappers must not change the computation: outputs of the plain and the
# traced binary may differ only in date/time and CPU-time lines.
echo "== plain vs traced output (lines differing after dropping time stamps)"
for s in boron f; do
    n=$(diff <(grep -a -v -E 'Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|TIME|CPU|SEC' "$run/plain-$s/$s.output") \
             <(grep -a -v -E 'Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|TIME|CPU|SEC' "$run/trace-$s/$s.output") \
         | grep -c '^[<>]' || true)
    echo "$s: $n"
done

echo "== stderr (expected empty)"
for e in "$run"/*/*.err; do
    [ -s "$e" ] && { echo "non-empty: $e"; head -5 "$e"; }
done
true

echo "== trace sizes"
for d in "$run"/trace-*; do
    echo "$(basename "$d"):"
    wc -l "$d"/trace/*.txt | sed 's/^/  /'
done
