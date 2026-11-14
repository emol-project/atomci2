#!/bin/bash
# run_atsp.sh -- stage 2 against ATSP (Froese Fischer et al., MCHF/BP) in
# the same numerical orbitals: ATSP's HF orbitals (.w) are read by
# ac_tabrad and used by acatsp.
#
#   F 1s2 2s2 2p5 2P (HF):
#     - radial integrals printed by ATSP hf ("Additional parameters"):
#       <r^k>, F^k, G^k, R^k (8 digits printed; |rel| < 2e-7);
#     - per orbital: I(nl) = <h> and Rel(nl) = mass velocity + Darwin
#       (6 decimals printed; |diff| < 2e-6);
#     - the single-CSF energy against E_HF (|diff| < 2e-7).
#   B 1s2 2s2 2p + 1s2 2p3, 2P, in the B HF orbitals: both roots of the
#     non-relativistic CI against ATSP bpci (IREL = 0; |diff| < 1e-7).
#
#   ATSP_BIN   ATSP binaries (default $HOME/atsp_work/atsp/bin)
#   AC_WORK    work root ($AC_WORK/m0/build/CFPTBL)
#
# bpci input notes: the first line must be "NAME,y,n" without blanks (it
# reads the characters right after the commas), and in relativistic mode
# every level includes ATSP's scalar relativistic shift.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf bpci || exit 0
acatsp=$repo/c/build/acatsp
cfp=$work/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"

# ---- F ---------------------------------------------------------------------
mkdir f && cd f
hf > hf.screen 2>&1 <<'STOP'
F,2P,9.
  1s  2s
2p(5)
all
y
y
y
1
  2p -3
1
  2p -1
1
  1s -1
1
  2s  1
1
  1s  2
2
F 0( 1s, 1s)
2
F 2( 2p, 2p)
2
G 1( 2s, 2p)
2
G 0( 1s, 2s)
2
R 1( 1s, 2p, 2s, 2p)
6
n
STOP
printf '1s(2) 2s(2) 2p(5)\n' > cfg
"$acatsp" "$cfp" wfn.out cfg 1 noso q=r,2p,-3 q=r,2p,-1 q=r,1s,-1 q=r,2s,1 q=r,1s,2 \
    q=F,0,1s,1s q=F,2,2p,2p q=G,1,2s,2p q=G,0,1s,2s q=R,1,1s,2p,2s,2p > acatsp.out
cd ..

# ---- B ---------------------------------------------------------------------
mkdir b && cd b
hf > hf.screen 2>&1 <<'STOP'
B,2P,5.
  1s
2s(2)2p(1)
all
y
y
n
n
STOP
cp wfn.out B.w
cat > B.c <<'EOF'

  1s
  2s( 2)  2p( 1)
 1S0 2P1 2P
  2p( 3)
 2P1
*
EOF
bpci > bp.screen 2>&1 <<'EOF'
B,n,n
n
n
1,2
0
y
y
y
y
EOF
printf '1s(2) 2s(2) 2p(1)\n1s(2) 2p(3)\n' > cfg
"$acatsp" "$cfp" B.w cfg 4 noso > acatsp.out
cd ..

python3 - <<'PY'
import re
bad = 0
def chk(name, a, c, tol, rel=False):
    global bad
    d = abs(a - c) / (abs(a) if rel else 1)
    ok = d < tol
    bad += not ok
    print(f"   {'ok  ' if ok else 'FAIL'} {name:28s} ATSP {a: .10e}  C {c: .10e}  {'rel' if rel else 'abs'} {d:.1e}")

f = open("f/hf.screen").read()
fl = open("f/hf.log").read()
fc = open("f/acatsp.out").read()
print("== F 2p5 2P, ATSP HF orbitals")
atsp = [float(x.replace("D", "e")) for x in
        re.findall(r"VALUE OF <.*?> =\s+(\S+) a\.u\.|INTEGRAL\s+\w\s*\d+\(.*?\) =\s+(\S+) a\.u\.", f)
        for x in x if x]
cq = re.findall(r"^query (\S+)\s+(\S+)", fc, re.M)
for (name, cv), av in zip(cq, atsp):
    chk(name, av, float(cv), 2e-7, rel=True)
if len(cq) != len(atsp):
    print("   FAIL query count", len(cq), len(atsp)); bad += 1
for m in re.finditer(r"^\s+(\d[spdf])\s+\S+\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s+\S+\s*$", fl, re.M):
    el, inl, rel = m.group(1), float(m.group(2)), float(m.group(3))
    o = re.search(rf"^orbital {el}\s+l=\d\s+<h> (\S+).*MV (\S+)\s+D (\S+)", fc, re.M)
    chk(f"I({el})", inl, float(o.group(1)), 2e-6)
    chk(f"Rel({el}) = MV + Darwin", rel, float(o.group(2)) + float(o.group(3)), 2e-6)
ehf = float(re.search(r"Non-Relativistic\s+(\S+)", f).group(1))
ec = float(re.search(r"^ ?1/2 +1 +(\S+)", fc, re.M).group(1))
chk("E(1 CSF) = E_HF", ehf, ec, 2e-7)

print("== B 2s2 2p + 2p3 2P, non-relativistic CI")
bl = open("b/B.l").read()
bc = open("b/acatsp.out").read()
ea = [float(x) for x in re.findall(r"^\s+\d\s+(-\d+\.\d+)\s+\S", bl, re.M)]
ecs = sorted(float(m) for m in re.findall(r"^ ?1/2 +\d +(\S+) 2P", bc, re.M))
for i, (a, c) in enumerate(zip(ea, ecs)):
    chk(f"2P root {i + 1}", a, c, 1e-7)
if len(ea) != 2 or len(ecs) != 2:
    print("   FAIL root count", len(ea), len(ecs)); bad += 1
print("ATSP comparison", "passed" if not bad else "FAILED")
raise SystemExit(1 if bad else 0)
PY
