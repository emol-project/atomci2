#!/bin/bash
# prep.sh -- the Fortran ATOMCI run whose files acfs reads: pass 1 of
# samples/native/f.stdin (F 1s2 2s2 2p5 2P: MCSCF orbitals in the GTO basis f.bas,
# the SD-CI tape and the integral transformation) with CI and RS switched
# off (&CNTL NRNFG=1,1,1,1,0,0,0,0). It leaves
#   fort.15  NFTE, the energy-expression tape (orbital list, addressing)
#   fort.16  NFTRI, the transformed integrals
#   fort.18  NFAII, the basis (its header)
#   fort.21  NFTRC, the MO coefficients
# Needs ATOMCI_HOME (make install at the top of the repository). EXAMPLE_DIR is
# this directory when run_examples.sh runs the script in a copy.
set -euo pipefail
: "${ATOMCI_HOME:?set ATOMCI_HOME to the installed ATOMCI (make install)}"
samples=${EXAMPLE_DIR:-$(pwd)}/../../../samples
[ -f "$samples/native/f.bas" ] || samples=${ATOMCI_SAMPLES:?set ATOMCI_SAMPLES to the samples directory}
ln -sf "$ATOMCI_HOME/CFPTBL" CFPTBL
cp "$samples/native/f.bas" .
"$ATOMCI_HOME/bin/atgen" < "$samples/native/f.atg" > /dev/null
"$ATOMCI_HOME/bin/atomci" < f.prep.stdin > f.prep.output
ls fort.15 fort.16 fort.18 fort.21 > /dev/null
