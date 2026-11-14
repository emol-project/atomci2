#!/bin/bash
# prep.sh -- the Fortran ATOMCI runs of samples/native (Sc 3d 4s2 2D, Ti 3d2 4s2
# 3F, Cu 3d10 4s 2S: MCSCF orbitals in cc-pVTZ, SD-CI), each in its own
# directory sc/, ti/, cu/ here; acfs reads their fort.15, fort.16, fort.18
# and fort.21. Needs ATOMCI_HOME (make install at the top of the
# repository). EXAMPLE_DIR is this directory when run_examples.sh runs the
# script in a copy.
set -euo pipefail
: "${ATOMCI_HOME:?set ATOMCI_HOME to the installed ATOMCI (make install)}"
native=${EXAMPLE_DIR:-$(pwd)}/../../../samples/native
NATIVE_WORK=$(pwd) bash "$native/run_native.sh" sc ti cu
