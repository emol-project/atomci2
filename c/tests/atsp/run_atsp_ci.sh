#!/bin/bash
# run_atsp_ci.sh -- large CI against ATSP on the same orbitals and CSFs.
#
# Si-like 3s2 3p2 (ATSP's own example, examples/e1 and examples/files_c):
#   MCHF orbitals (.w) and CSF lists (.c, 89 to 916 CSFs) for 1D, 1S, 3P,
#   n = 4 and 5, Z = 14 and 15. acatsp reads the .c as it is (read_atsp_c)
#   and its lowest root is compared with the MCHF energy (.l).
# Ti 3d2 4s2 + 3d3 4s + 3d4 in Ti HF orbitals: acatsp writes every LS term as
#   an ATSP .c (write_atsp_c); bpci (non-relativistic) solves each, and all
#   roots are compared.
#
# The two programs integrate the core differently, so the energies differ by
# an offset (here 2e-7 to 5e-7 hartree) that is the same for every CSF space
# of an atom. A wrong angular coefficient would not be. Passes when, per
# atom, the spread of C - ATSP over all compared energies is below 2e-8 and
# |C - ATSP| < 1e-6.
#
#   ATSP_BIN  ATSP binaries (default $HOME/atsp_work/atsp/bin)
#   ATSP_SRC  ATSP source tree with examples/ (skipped when unset)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
export PATH=${ATSP_BIN:-$HOME/atsp_work/atsp/bin}:$PATH
. "$here/../lib/need.sh"
need_dir ATSP_SRC "the ATSP source tree, with examples/" || exit 0
need_cmd "ATSP: set ATSP_BIN to its bin directory" hf bpci || exit 0
src=$ATSP_SRC
acatsp=$repo/c/build/acatsp
cfp=$work/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"

# ---- Si ----------------------------------------------------------------------
for z in 14 15; do for n in 4 5; do for t in 1D 1S 3P; do
    e=$(awk 'NR > 3 && $1 == 1 { print $2; exit }' "$src/examples/e1/e1.${z}_${n}_${t}.l")
    "$acatsp" "$cfp" "$src/examples/e1/e1.${z}_${n}_${t}.w" "$src/examples/files_c/${t}e1.$n.c" \
        1 ls noso | awk -v e="$e" -v tag="Si Z=$z n=$n $t" '/^term/ { print tag, $3, e, $4 }'
done; done; done > si.txt

# ---- Ti ----------------------------------------------------------------------
mkdir ti && cd ti
hf > hf.screen 2>&1 <<'S'
Ti,3F,22.
  1s  2s  2p  3s  3p
3d(2)4s(2)
all
y
y
n
n
S
cp wfn.out Ti.w
core="1s(2) 2s(2) 2p(6) 3s(2) 3p(6)"
printf '%s 3d(2) 4s(2)\n%s 3d(3) 4s(1)\n%s 3d(4)\n' "$core" "$core" "$core" > cfg
"$acatsp" "$cfp" Ti.w cfg 400 ls noso writec=Ti > c.out
cd ..

python3 - <<'PY'
import re, subprocess, os
bad = 0
# Si: groups by Z
rows = [l.split() for l in open("si.txt")]
for z in ("14", "15"):
    off = [float(r[-1]) - float(r[-2]) for r in rows if r[1] == f"Z={z}"]
    for r in rows:
        if r[1] == f"Z={z}":
            print(f"   Si {' '.join(r[1:4])}: {r[4]:>4} CSFs  ATSP {float(r[5]):.9f}  C {float(r[6]):.9f}"
                  f"  C-ATSP {float(r[6]) - float(r[5]):+.2e}")
    ok = len(off) == 6 and max(off) - min(off) < 2e-8 and max(map(abs, off)) < 1e-6
    bad += not ok
    print(f"   {'ok  ' if ok else 'FAIL'} Si Z={z}: {len(off)} cases, spread of C-ATSP {max(off) - min(off):.1e}")
# Ti
os.chdir("ti")
c = open("c.out").read()
terms = {m.group(1): [float(x) for x in m.group(3).split()]
         for m in re.finditer(r"^term (\S+) (\d+) (.*)$", c, re.M)}
off, nroots = [], 0
for term, ec in sorted(terms.items()):
    name = f"Ti_{term}"
    os.system(f"cp Ti.w {name}.w; rm -f *.lst")
    inp = f"{name},n,n\nn\nn\n{','.join(str(i + 1) for i in range(len(ec)))}\n0\ny\ny\ny\ny\n"
    subprocess.run(["bpci"], input=inp, capture_output=True, text=True)
    ea = sorted(float(x) for x in
                re.findall(r"^\s+\d+\s+(-\d+\.\d+)\s+\S", open(f"{name}.l").read(), re.M))
    if len(ea) != len(ec):
        print(f"   FAIL Ti {term}: ATSP {len(ea)} roots, C {len(ec)}")
        bad += 1
        continue
    off += [b - a for a, b in zip(ea, ec)]
    nroots += len(ec)
ok = off and max(off) - min(off) < 2e-8 and max(map(abs, off)) < 1e-6
bad += not ok
print(f"   {'ok  ' if ok else 'FAIL'} Ti 3d2 4s2 + 3d3 4s + 3d4: {len(terms)} LS terms, {nroots} roots,"
      f" C-ATSP {min(off):.2e} .. {max(off):.2e} (spread {max(off) - min(off):.1e})")
print("ATSP CI comparison", "passed" if not bad else "FAILED")
raise SystemExit(1 if bad else 0)
PY
