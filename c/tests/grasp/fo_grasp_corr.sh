#!/bin/bash
# fo_grasp_corr.sh -- S2-M7 (Z12), report only (GRASP and ATSP needed): the
# residual C[1e + GC2e(SSO)] - GRASP (no-pair Dirac-Coulomb g_J) for the neutral
# atoms F 2p5 2P and O 2p4 3P, as blike_grasp_corr.sh does for B-like ions, in
# matched CSF spaces with 1s2 inactive:
#   1cfg  the ground configuration
#   n3    single and double excitations of 2s2 2p^q to n = 3 (l <= d)
#   n4    the same to n = 4 (l <= f)
# GRASP: MCDHF (EOL over the lowest level of each J), new layer varied, rci
# without Breit, rhfszeeman95. C: every LS term of the same configurations
# (acatsp configuration mode, H_SO1 + spin-same-orbit H_SO2) on the ATSP orbitals of f2p_gj.sh /
# o3p_gj.sh (HF and valence MCHF n = 3, 4; run those first with CORE unset).
# Delta g is taken from the Lande value with g_s = 2 on both sides; C's
# intermediate-coupling part uses the real g_s (an error of ~2e-8).
#   ATOMS  "F O" (default)
#   SKIP_GRASP=1  reuse the GRASP results of an earlier run
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${GRASP_BIN:-$HOME/grasp_work/grasp/bin}:${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "GRASP: set GRASP_BIN to its bin directory" rcsfgenerate rmcdhf rci rhfszeeman95 || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL

grasp_level() { # tag "reference" active excitations "orbitals to vary" Jrange
    local tag=$1 refs=$2 act=$3 ex=$4 vary=$5 jr=$6
    rcsfgenerate > csf_$tag.out 2>&1 <<S
*
0
$refs

$act
$jr
$ex
n
S
    cp rcsf.out rcsf.inp
    rangular > ang_$tag.out 2>&1 <<'S'
y
S
    if [ -s rwfn.out ]; then
        cp rwfn.out prev.w
        rwfnestimate > est_$tag.out 2>&1 <<'S'
y
1
prev.w
*
3
*
S
    else
        # first estimates: screened hydrogenic (3) for F; O's DHF fails for 2s
        # from those and converges from Thomas-Fermi (2)
        rwfnestimate > est_$tag.out 2>&1 <<S
y
$est
*
S
    fi
    rmcdhf > scf_$tag.out 2>&1 <<S
y
1
1
5
$vary
1s,2s,2p*
200
S
    grep -qE "Convergence not obtained|Failure" scf_$tag.out && echo "   GRASP $tag: MCDHF failed"
    rsave $tag > /dev/null 2>&1
    rci > rci_$tag.out 2>&1 <<S
y
$tag
n
n
n
n
n
n
1
1
S
    rhfszeeman95 > zee_$tag.out 2>&1 <<S
y
$tag
y
n
S
    echo "   GRASP $tag: $(grep -c '^ *[0-9]' rcsf.inp) lines in the CSF list"
}

for at in ${ATOMS:-F O}; do
    if [ $at = F ]; then Z=9; q=5; jr=1,3; est=3; src=${F_WORK:-${AC_WORK:-$HOME/atomci_work}/f2p}
    else Z=8; q=4; jr=2,4; est=2; src=${O_WORK:-${AC_WORK:-$HOME/atomci_work}/o3p}; fi
    w=${AC_WORK:-$HOME/atomci_work}/fo_grasp/$at
    if [ "${SKIP_GRASP:-0}" = 1 ]; then cd "$w"; echo "== $at (GRASP results reused)"; else
    rm -rf "$w" && mkdir -p "$w/grasp" && cd "$w/grasp"
    M=$((2 * Z + 1))
    rnucleus > nuc.out 2>&1 <<S
$Z
$M
n
$M
1
1
1
S
    echo "== $at"
    grasp_level 1cfg "1s(2,i)2s(2,i)2p($q,i)" "2s,2p" 0 "*" $jr
    grasp_level n3 "1s(2,i)2s(2,*)2p($q,*)" "3s,3p,3d" 2 "3s,3p*,3d*" $jr
    grasp_level n4 "1s(2,i)2s(2,*)2p($q,*)" "4s,4p,4d,4f" 2 "4s,4p*,4d*,4f*" $jr
    cd "$w"
    fi
    for t in 1cfg n3 n4; do
        python3 - $t $q > cfg_$t <<'PY'
import itertools, sys
tag, q = sys.argv[1], int(sys.argv[2])
orbs = [("2s", 0), ("2p", 1)]
if tag != "1cfg":
    orbs += [("3s", 0), ("3p", 1), ("3d", 2)]
if tag == "n4":
    orbs += [("4s", 0), ("4p", 1), ("4d", 2), ("4f", 3)]
ref = [2, q] + [0] * (len(orbs) - 2)
ne = 2 + q
for occ in itertools.product(*[range(0, min(4 * l + 2, ne) + 1) for _, l in orbs]):
    if sum(occ) != ne or sum(o * l for o, (_, l) in zip(occ, orbs)) % 2 != q % 2:
        continue
    if sum(max(r - o, 0) for r, o in zip(ref, occ)) > 2:
        continue
    print("1s(2) " + " ".join(f"{n}({o})" for (n, _), o in zip(orbs, occ) if o))
PY
    done
    # C's CI has H_SO1 + the spin-same-orbit part of H_SO2, the spin-orbit
    # content of Dirac-Coulomb (the term mixing of 3P2 with 1D2 depends on it)
    co="alpha=137.035999084 so2 so2part=1 gc2e gc2epart=1"
    "$A" "$T" $src/${at}_hf.w cfg_1cfg 2 $co > s_1cfg.out
    "$A" "$T" $src/${at}_3.w cfg_n3 2 $co > s_n3.out
    "$A" "$T" $src/${at}_4.w cfg_n4 2 $co > s_n4.out
    echo "   C: $(wc -l < cfg_n3) and $(wc -l < cfg_n4) configurations (n3, n4)"
    python3 - $at <<'PY'
import re, sys
at = sys.argv[1]
gs = 2.00231930436256
if at == "F":
    lev = {1: (4 / 3, -1 / 3), 3: (2 / 3, 1 / 3)}   # 2J: (c_L, c_S)
else:
    lev = {2: (0.5, 0.5), 4: (0.5, 0.5)}
def grasp(tag):
    g = {}
    for l in open(f"grasp/{tag}.ch"):
        # J is printed as 3/2 or, for integer J, as 2; the g_J line comes last
        m = re.match(r"\s+1\s+(\d+)(/2)?\s+[-+]\s+\S+\s+\S+\s+(\S+)", l)
        if m:
            tj = int(m.group(1)) * (1 if m.group(2) else 2)
            g[tj] = float(m.group(3).replace("D", "e"))
    return g
def c(tag):
    r = {}
    for l in open(f"s_{tag}.out"):
        x = l.split()
        if len(x) >= 10 and re.match(r"^\d+/2$", x[0]) and x[1] == "1":
            r[int(x[0].split("/")[0])] = [float(v) for v in x[5:10]]
    return r
for tj, (cl, cs) in lev.items():
    l2 = cl + 2 * cs
    for t in ("1cfg", "n3", "n4"):
        gg = grasp(t)[tj] - l2
        gj, rmc, gc, ormc, sso = c(t)[tj]
        cd = (gj - (cl + gs * cs)) + rmc + gc + ormc + sso
        print(f"{at} J={tj}/2 {t:4}  GRASP {gg * 1e6:+9.3f}e-6  C 1e+SSO {cd * 1e6:+9.3f}e-6"
              f"  residual {(cd - gg) * 1e6:+8.3f}e-6  /SSO {(cd - gg) / sso:+.3f}  (SSO {sso * 1e6:+.3f}e-6)")
PY
done
