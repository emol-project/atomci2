#!/bin/bash
# h6_hydro.sh -- S2-M7, report only (ATSP's hf needed): the rank-2 part of
# Pachucki's H6 (acatsp "h6", ac_lsj_g h6) in hydrogen-like 2p (H, He+),
# against the anomalous-moment part of the bound-electron g factor.
# With g_s = 2 (1 + kappa), the kappa (Z alpha)^2 terms of Pachucki's
# Hamiltonian (PRA 69, 052502 (2004), Eq. (32)) for one electron are
#   H2 (1 + 2 kappa)  -> 2 kappa dg_GC,
#   H6 scalar part    -> (kappa/3) dg_RMC,
#   H6 rank-2 part    -> kappa h6,
# and their sum is kappa (Z alpha)^2 2 b20^(H) / n^2 with Jentschura's
# (PRA 81, 012512 (2010), arXiv:1001.3913, Eq. (10); his kappa = alpha/(2 pi))
# b20^(H) = -1/2 (nP1/2) and 1/10 (nP3/2), from the Dirac equation with the
# Pauli term (Grotch and Kashuba 1973).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
for at in "H 1" "He 2"; do
    set -- $at
    hf > hf_$1.log 2>&1 <<S
$1,2P,$2.

2p(1)
all
y
y
n
n
S
    cp wfn.out $1.w
    printf '2p(1)\n' > $1.cfg
    "$A" "$T" $1.w $1.cfg 2 alpha=137.035999084 h6 > $1.out
done
python3 - <<'PY'
import re
alpha = 1 / 137.035999084
worst = 0.0
for el, Z in (("H", 1), ("He", 2)):
    for l in open(f"{el}.out"):
        x = l.split()
        if len(x) < 10 or not re.match(r"^\d+/2$", x[0]):
            continue
        tj = int(x[0][:-2])
        rmc, gc, h6 = float(x[6]), float(x[7]), float(x[-1])
        got = 2 * gc + rmc / 3 + h6
        want = (Z * alpha) ** 2 / 4 * (-1.0 if tj == 1 else 0.2)
        worst = max(worst, abs(got / want - 1))
        print(f"   {el} 2P{tj}/2: 2 GC {2 * gc:+.6e}  RMC/3 {rmc / 3:+.6e}  h6 {h6:+.6e}  sum {got:+.6e}"
              f"  (Z alpha)^2 2 b20/n^2 {want:+.6e}  ratio - 1 {got / want - 1:+.1e}")
print(f"h6_hydro: max |ratio - 1| = {worst:.1e} (C prints 7 digits of RMC and GC)")
PY
