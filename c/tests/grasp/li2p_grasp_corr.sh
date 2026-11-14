#!/bin/bash
# li2p_grasp_corr.sh -- S2-M7 (Z14), report only (GRASP and ATSP needed): the
# residual C[1e + GC2e(SSO)] - GRASP (no-pair Dirac-Coulomb g_J) for Li 1s2 2p
# 2P_J with correlation, as fo_grasp_corr.sh does for F and O. One valence
# electron has no valence correlation, so the 1s shell is active here: complete
# active spaces of three electrons from 1s2 2p,
#   1cfg  1s2 2p
#   n3    CAS over 1s, 2s, 2p, 3s, 3p, 3d
#   n4    CAS over the n = 4 set (l <= f)
# GRASP: MCDHF (EOL over the lowest level of each J), 2s and the new layer
# varied, rci without Breit, rhfszeeman95. C: every LS term of the same
# configurations (acatsp configuration mode, H_SO1 + spin-same-orbit H_SO2) on
# the ATSP orbitals of li2p_yan.sh (L_2.w = HF, L_3.w, L_4.w; run it first).
# Delta g is taken from the Lande value with g_s = 2 on both sides; C's
# intermediate-coupling part uses the real g_s (an error of ~2e-8).
#   LI_WORK       li2p_yan.sh's work directory (default $HOME/atomci_work/li2p)
#   SKIP_GRASP=1  reuse the GRASP results of an earlier run
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
export PATH=${GRASP_BIN:-$HOME/grasp_work/grasp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "GRASP: set GRASP_BIN to its bin directory" rcsfgenerate rmcdhf rci rhfszeeman95 || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
src=${LI_WORK:-${AC_WORK:-$HOME/atomci_work}/li2p}
for f in L_2.w L_3.w L_4.w; do
    [ -s "$src/$f" ] || { echo "li2p_grasp_corr: $src/$f missing (run li2p_yan.sh first)"; exit 1; }
done

grasp_level() { # tag "reference" active excitations "orbitals to vary" "spectroscopic"
    local tag=$1 refs=$2 act=$3 ex=$4 vary=$5 spec=$6
    rcsfgenerate > csf_$tag.out 2>&1 <<S
*
0
$refs

$act
1,3
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
        rwfnestimate > est_$tag.out 2>&1 <<'S'
y
3
*
S
    fi
    rmcdhf > scf_$tag.out 2>&1 <<S
y
1
1
5
$vary
$spec
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

w=${AC_WORK:-$HOME/atomci_work}/li2p_grasp
if [ "${SKIP_GRASP:-0}" = 1 ]; then cd "$w"; echo "== Li 2p (GRASP results reused)"; else
rm -rf "$w" && mkdir -p "$w/grasp" && cd "$w/grasp"
rnucleus > nuc.out 2>&1 <<'S'
3
7
n
7
1
1
1
S
echo "== Li 2p"
grasp_level 1cfg "1s(2,i)2p(1,i)" "2s,2p" 0 "*" "*"
grasp_level n3 "1s(2,*)2p(1,*)" "3s,3p,3d" 3 "2s,3s,3p*,3d*" "1s,2p*"
grasp_level n4 "1s(2,*)2p(1,*)" "4s,4p,4d,4f" 3 "4s,4p*,4d*,4f*" "1s,2p*"
cd "$w"
fi
for t in 1cfg n3 n4; do
    python3 - $t > cfg_$t <<'PY'
import itertools, sys
tag = sys.argv[1]
orbs = [("1s", 0), ("2p", 1)]
if tag != "1cfg":
    orbs = [("1s", 0), ("2s", 0), ("2p", 1), ("3s", 0), ("3p", 1), ("3d", 2)]
if tag == "n4":
    orbs += [("4s", 0), ("4p", 1), ("4d", 2), ("4f", 3)]
for occ in itertools.product(*[range(0, min(4 * l + 2, 3) + 1) for _, l in orbs]):
    if sum(occ) == 3 and sum(o * l for o, (_, l) in zip(occ, orbs)) % 2 == 1:
        print(" ".join(f"{n}({o})" for (n, _), o in zip(orbs, occ) if o))
PY
done
co="alpha=137.035999084 so2 so2part=1 gc2e gc2epart=1"
"$A" "$T" $src/L_2.w cfg_1cfg 2 $co > s_1cfg.out
"$A" "$T" $src/L_3.w cfg_n3 2 $co > s_n3.out
"$A" "$T" $src/L_4.w cfg_n4 2 $co > s_n4.out
echo "   C: $(wc -l < cfg_n3) and $(wc -l < cfg_n4) configurations (n3, n4)"
python3 - <<'PY'
import re
gs = 2.00231930436256
lev = {1: (4 / 3, -1 / 3), 3: (2 / 3, 1 / 3)}   # 2J: (c_L, c_S)
def grasp(tag):
    g = {}
    for l in open(f"grasp/{tag}.ch"):
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
        print(f"Li J={tj}/2 {t:4}  GRASP {gg * 1e6:+9.3f}e-6  C 1e+SSO {cd * 1e6:+9.3f}e-6"
              f"  residual {(cd - gg) * 1e6:+8.3f}e-6  /SSO {(cd - gg) / sso:+.3f}  (SSO {sso * 1e6:+.3f}e-6)")
PY
