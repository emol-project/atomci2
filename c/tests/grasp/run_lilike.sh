#!/bin/bash
# run_lilike.sh -- S2-M7 (Z5b), report only (GRASP and ATSP needed).
# Li-like 1s2 nl: Delta g_int = g_J - g_D (hydrogenic Dirac) from C full BP and GRASP (no-pair DC),
# against arXiv:2604.25515 Table 3 (P = positive-energy, N = negative-energy parts)
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export PATH=${GRASP_BIN:-$HOME/grasp_work/grasp/bin}:${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "GRASP: set GRASP_BIN to its bin directory" rcsfgenerate rmcdhf rci rhfszeeman95 || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf || exit 0
A=$repo/c/build/acatsp
T=${AC_WORK:-$HOME/atomci_work}/m0/build/CFPTBL
one() { # Z state(2s|2p)
  local Z=$1 st=$2 W=$tmp/li_$1_$2 M=$((2 * $1 + 1))
  rm -rf $W; mkdir -p $W; cd $W
  local cfg act jr term open
  if [ $st = 2s ]; then cfg="1s(2,i)2s(1,i)"; act="2s"; jr="1,1"; term=2S; open="2s(1)"
  else cfg="1s(2,i)2p(1,i)"; act="2s,2p"; jr="1,3"; term=2P; open="2p(1)"; fi
  rnucleus > nuc.out 2>&1 <<S
$Z
$M
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
  rwfnestimate > est.out 2>&1 <<'S'
y
3
*
S
  if [ $st = 2s ]; then lv=$'1'; rl=$'1'; else lv=$'1\n1\n5'; rl=$'1\n1'; fi
  rmcdhf > scf.out 2>&1 <<S
y
$lv
*
*
200
S
  grep -qE "Convergence not obtained|Failure" scf.out && echo "Z=$Z $st: DHF failed"
  rsave X > /dev/null 2>&1
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
L$Z,$term,$Z.
  1s
$open
all
y
y
n
n
S
  cp wfn.out L.w
  echo "1s(2) $open" > cfg
  $A $T L.w cfg 2 alpha=137.035999084 gc2e gc2epart=1 > sso.out
  $A $T L.w cfg 2 alpha=137.035999084 gc2e oz2e > all.out
  python3 - $Z $st <<'PY'
import math, re, sys
Z, st = int(sys.argv[1]), sys.argv[2]
a = 1 / 137.035999084
tab = {  # arXiv:2604.25515 Table 3: (N, P)
    ("2s", 1, 4): (-0.00000249, 0.00008750), ("2s", 1, 6): (-0.00000572, 0.00013663),
    ("2s", 1, 10): (-0.00001222, 0.00023491), ("2p", 1, 4): (-0.00001063, 0.00011185),
    ("2p", 1, 6): (-0.00003008, 0.00019304), ("2p", 1, 10): (-0.00007047, 0.00035622),
    ("2p", 3, 4): (0.00000114, 0.00008096), ("2p", 3, 6): (0.00000306, 0.00012982),
    ("2p", 3, 10): (0.00000660, 0.00022663)}
def gD(kappa, n=2):
    k = abs(kappa); j = k - 0.5
    gam = math.sqrt(k * k - (Z * a) ** 2)
    E = 1 / math.sqrt(1 + (Z * a / (n - k + gam)) ** 2)
    return kappa * (2 * kappa * E - 1) / (2 * j * (j + 1))
g = {}
for l in open("X.ch"):
    m = re.match(r"\s+1\s+(\d)/2\s+[-+]\s+\S+\s+\S+\s+(\S+)", l)
    if m:
        g[int(m.group(1))] = float(m.group(2).replace("D", "e"))
def rd(f):
    c = {}
    for l in open(f):
        x = l.split()
        if len(x) >= 10 and re.match(r"^\d/2$", x[0]) and x[1] == "1":
            c[int(x[0][0])] = [float(v) for v in x[6:]]
    return c
sso, full = rd("sso.out"), rd("all.out")
for tj in sorted(g):
    kappa = -1 if (st == "2s") else (1 if tj == 1 else -2)
    lande = 2.0 if st == "2s" else (2 / 3 if tj == 1 else 4 / 3)
    d = gD(kappa)
    rmc, gc, ormc, s = sso[tj][:4]
    cs = lande + rmc + gc + ormc + s - d          # C with GC2e(SSO): the Dirac-Coulomb content
    f = full[tj]
    cf = lande + f[0] + f[1] + f[2] + f[3] + (f[4] if len(f) > 4 else 0.0) - d  # + SOO + OZ2e
    gg = g[tj] - d
    N, P = tab.get((st, tj, Z), (float("nan"), float("nan")))
    print(f"Z={Z:2} {st}{tj}/2  dg_int: GRASP(noPair DC) {gg:+.4e}  C[1e+SSO] {cs:+.4e}  C[full BP] {cf:+.4e}"
          f"   paper P {P:+.4e}  P+N {P + N:+.4e}  N {N:+.3e}   C[1e+SSO]-GRASP {cs - gg:+.2e}")
PY
}
for Z in 4 6 10; do one $Z 2s; one $Z 2p; done
