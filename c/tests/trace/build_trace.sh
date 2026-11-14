#!/bin/bash
# build_trace.sh -- M0: build ATOMCI in a scratch copy and relink a traced
# atomci with trace_wrap.c via GNU ld --wrap. The repository is not modified.
#
#   AC_WORK   work root (default: $HOME/atomci_work); this uses $AC_WORK/m0
#
# Produces, under $AC_WORK/build:
#   atomci.plain   the normal binary
#   atomci         the traced binary (records to $AC_TRACE_DIR at run time)
#   atgen, CFPTBL  as built by src/Makefile
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}/m0
build=$work/build

wrapped="xwig6j_ xwig9x_ xcfp_ shms_ smat_ isqt_"

rm -rf "$build"
mkdir -p "$build"
cp -r "$repo/src/." "$build/"
cd "$build"

echo "== make all"
make all > make.log 2>&1 || { tail -40 make.log; exit 1; }
cp atomci atomci.plain

echo "== check that every wrapped symbol is defined in some object"
# Read nm's output from a file: with pipefail, `nm | grep -q` fails whenever
# grep exits early and nm gets SIGPIPE.
nm --defined-only *.o > defined.nm 2>/dev/null
for s in $wrapped; do
    if ! grep -q " T $s\$" defined.nm; then
        echo "symbol $s not defined in any object" >&2
        exit 1
    fi
done

echo "== compile trace_wrap.c"
gcc -O2 -std=c11 -Wall -Wextra -pedantic -Werror \
    -c "$here/trace_wrap.c" -o trace_wrap.o

wrapflags=""
for s in $wrapped; do
    wrapflags="$wrapflags,--wrap=$s"
done
wrapflags="-Wl${wrapflags}"

echo "== relink atomci with $wrapflags"
rm -f atomci
make atomci lf7="gfortran $wrapflags $build/trace_wrap.o" > relink.log 2>&1 \
    || { tail -40 relink.log; exit 1; }

echo "== confirm the traced binary calls the wrappers"
nm atomci > atomci.nm
for s in $wrapped; do
    grep -q " T __wrap_$s\$" atomci.nm || { echo "missing __wrap_$s" >&2; exit 1; }
done
echo "built: $build/atomci (traced), $build/atomci.plain"
