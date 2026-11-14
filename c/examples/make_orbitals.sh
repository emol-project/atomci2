#!/bin/bash
# make_orbitals.sh -- regenerate the ATSP orbital files (.w) of the examples.
# The .w files are shipped, so the examples run without ATSP; this script
# records how they were made. It needs ATSP's hf on PATH (or ATSP_BIN).
#
# Every orbital set is the Hartree-Fock solution of ATSP's hf for one LS
# term (C. Froese Fischer et al., Comput. Phys. Commun. 176, 559 (2007)).
# hf asks: atom, term, Z; the closed shells; the open shells; the orbitals
# to vary; then four yes/no questions (default electron parameters, default
# rate of convergence, additional parameters, a different configuration),
# answered y y n n.
#
#   ./make_orbitals.sh [name ...]    (default: all)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
command -v hf > /dev/null || { echo "ATSP hf not found (set ATSP_BIN)" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

hf_run() { # output .w, then the hf input on stdin
    local out=$1
    (cd "$tmp" && rm -f wfn.out && hf > hf.log 2>&1 && cp wfn.out "$out") ||
        { echo "hf failed for $out (log: $tmp/hf.log)" >&2; cat "$tmp/hf.log" >&2; exit 1; }
    echo "wrote $out"
}

for name in ${@:-o2p4 li he2p sc ti cu}; do
    case $name in
    o2p4) hf_run "$here/o2p4/O.w" <<'S'
O,3P,8.
  1s  2s
2p(4)
all
y
y
n
n
S
    ;;
    li) hf_run "$here/li/Li2s.w" <<'S'
Li,2S,3.
  1s
2s(1)
all
y
y
n
n
S
        hf_run "$here/li/Li2p.w" <<'S'
Li,2P,3.
  1s
2p(1)
all
y
y
n
n
S
    ;;
    he2p) hf_run "$here/he2p/He.w" <<'S'
He,3P,2.

1s(1)2p(1)
all
y
y
n
n
S
    ;;
    sc) hf_run "$here/sc/Sc.w" <<'S'
Sc,2D,21.
  1s  2s  2p  3s  3p
3d(1)4s(2)
all
y
y
n
n
S
    ;;
    ti) hf_run "$here/ti/Ti.w" <<'S'
Ti,3F,22.
  1s  2s  2p  3s  3p
3d(2)4s(2)
all
y
y
n
n
S
    ;;
    cu) hf_run "$here/cu/Cu.w" <<'S'
Cu,2S,29.
  1s  2s  2p  3s  3p
3d(10)4s(1)
all
y
y
n
n
S
    ;;
    *) echo "unknown example $name" >&2; exit 1;;
    esac
done
