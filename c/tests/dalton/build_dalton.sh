#!/bin/bash
# build_dalton.sh -- configure and build Dalton (source in $DALTON_SRC, required)
# into $DALTON_BUILD (default $HOME/dalton_build), serial, system BLAS/LAPACK, 8 jobs.
set -euo pipefail
src=${DALTON_SRC:?set DALTON_SRC to the Dalton source tree}
b=${DALTON_BUILD:-$HOME/dalton_build}
cd "$src"
[ -f "$b/CMakeCache.txt" ] || python3 ./setup --fc=gfortran --cc=gcc --cxx=g++ "$b"
cd "$b"
make -j8 2>&1 | tail -20
ls -la dalton dalton.x
