#!/bin/bash
# run_gj.sh -- S2-M7 (Z5): the relativistic corrections to g_J against GRASP
# (Dirac-Coulomb: DHF with EOL over the levels, rci without Breit,
# rhfszeeman95's Dirac g_J). Report only (no pass/fail): the agreement and
# the open residual are discussed in the technical note, Sec. 7.3.
#
# For each system: GRASP's g_J minus the Lande value with g_s = 2, against C
# (acatsp on ATSP HF orbitals of the same configuration) with
#   1e = RMC + ORMC + GC1e   and   1e + GC2e (spin-same-orbit part only:
# the part that Dirac-Coulomb contains; SOO and OZ2e come from Breit).
#   GRASP_BIN, ATSP_BIN  binaries (defaults ~/grasp_work/grasp/bin,
#                        ~/atsp_work/atsp/bin)
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
export PATH=${GRASP_BIN:-$HOME/grasp_work/grasp/bin}:${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "GRASP: set GRASP_BIN to its bin directory" rcsfgenerate rmcdhf rci rhfszeeman95 || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf || exit 0
A=$repo/c/build/acatsp
T=$work/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# name Z mass term "grasp configuration" "grasp active set" "2J range" estimate
#   "atsp closed" "atsp open" "acatsp configuration"
run() {
    local n=$1 Z=$2 M=$3 term=$4 cfg=$5 act=$6 jr=$7 est=$8
    local W=$tmp/$n
    mkdir -p "$W" && cd "$W"
    rnucleus > nuc.out 2>&1 <<S
$Z
$(python3 -c "print(round(2*$Z+1))")
n
$M
1
1
1
S
    rcsfgenerate > csf.out 2>&1 <<S
*
0
$cfg

$act
$jr
0
n
S
    cp rcsf.out rcsf.inp
    rangular > ang.out 2>&1 <<'S'
y
S
    rwfnestimate > est.out 2>&1 <<S
y
$est
*
S
    # one J block: the level only; two blocks: a level each and standard weights
    local levels
    if [ "$jr" = "1,1" ]; then levels=$'1'; else levels=$'1\n1\n5'; fi
    rmcdhf > scf.out 2>&1 <<S
y
$levels
*
*
200
S
    if grep -qE "Convergence not obtained|Failure" scf.out; then
        echo "   $n: GRASP DHF did not converge"
        return
    fi
    rsave X > /dev/null 2>&1
    local rl=$'1'
    [ "$jr" = "1,1" ] || rl=$'1\n1'
    rci > rci.out 2>&1 <<S
y
X
n
n
n
n
n
n
$rl
S
    rhfszeeman95 > zee.out 2>&1 <<'S'
y
X
y
n
S
    hf > hf.out 2>&1 <<S
$n,$term,$Z.
$9
${10}
all
y
y
n
n
S
    cp wfn.out $n.w
    echo "${11}" > cfg
    "$A" "$T" $n.w cfg 2 alpha=137.035999084 gc2e gc2epart=1 > c.out
    python3 - "$n" "$term" <<'PY'
import re, sys
n, term = sys.argv[1:]
g = {}
for l in open("X.ch"):
    m = re.match(r"\s+1\s+(\d)/2\s+[-+]\s+\S+\s+\S+\s+(\S+)", l)
    if m:
        g[int(m.group(1))] = float(m.group(2).replace("D", "e"))
c = {}
for l in open("c.out"):
    f = l.split()
    if len(f) >= 10 and re.match(r"^\d/2$", f[0]) and f[1] == "1":
        c[int(f[0][0])] = [float(x) for x in f[6:10]]
lande = {"2S": {1: 2.0}, "2P": {1: 2 / 3, 3: 4 / 3}}[term]
for tj in sorted(g):
    dG = g[tj] - lande[tj]
    rmc, gc, ormc, gc2e = c[tj]
    e1 = rmc + gc + ormc
    print(f"   {n:5} {term}{tj}/2  GRASP {dG:+.4e}   C 1e {e1:+.4e} ({(e1 - dG) / dG:+6.1%})"
          f"   C 1e+GC2e(SSO) {e1 + gc2e:+.4e} ({(e1 + gc2e - dG) / dG:+6.1%})")
PY
}

echo "== S states"
run Li 3 6.941 2S "1s(2,i)2s(1,i)" "2s" 1,1 3 "  1s" "2s(1)" "1s(2) 2s(1)"
run Na 11 22.98977 2S "1s(2,i)2s(2,i)2p(6,i)3s(1,i)" "3s,2p" 1,1 2 "  1s  2s  2p" "3s(1)" "1s(2) 2s(2) 2p(6) 3s(1)"
echo "== one p electron over 1s2"
run Li2p 3 6.941 2P "1s(2,i)2p(1,i)" "2s,2p" 1,3 3 "  1s" "2p(1)" "1s(2) 2p(1)"
run F6p 9 18.998403 2P "1s(2,i)2p(1,i)" "2s,2p" 1,3 3 "  1s" "2p(1)" "1s(2) 2p(1)"
echo "== 2s2 2p and p5"
run B 5 10.811 2P "1s(2,i)2s(2,i)2p(1,i)" "2s,2p" 1,3 3 "  1s" "2s(2)2p(1)" "1s(2) 2s(2) 2p(1)"
run Ne5p 10 21 2P "1s(2,i)2s(2,i)2p(1,i)" "2s,2p" 1,3 3 "  1s" "2s(2)2p(1)" "1s(2) 2s(2) 2p(1)"
run F 9 18.998403 2P "1s(2,i)2s(2,i)2p(5,i)" "2s,2p" 1,3 3 "  1s  2s" "2p(5)" "1s(2) 2s(2) 2p(5)"
run Cl 17 35.453 2P "1s(2,i)2s(2,i)2p(6,i)3s(2,i)3p(5,i)" "3s,3p" 1,3 2 "  1s  2s  2p  3s" "3p(5)" "1s(2) 2s(2) 2p(6) 3s(2) 3p(5)"
exit 0
