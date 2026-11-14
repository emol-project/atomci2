#!/bin/bash
# run_real.sh -- S2-M4 on real atoms: acfs with the radial integrals of the
# Fortran's own basis and orbitals (ac_radial).
#
# f sample (GTO, $AC_WORK/m5/f/prep; run c/tests/m5/run_m5.sh first):
#   F 1s2 2s2 2p5 2P, one CSF. The level order must be inverted
#   (J = 3/2 lowest) and the splitting must equal (3/2) zeta(2p) exactly,
#   with zeta(2p) as acfs prints it. g_J must be the Lande values with g_s.
#   With mass velocity + Darwin in h ("mvd"), both levels must shift by
#   sum_i n_i (MV + D)_ii (one CSF, so the energy is linear in h).
# boron V2b system ($AC_WORK/s2m2; run c/tests/s2m2/run_v2b.sh first):
#   the 5-configuration space; the ground level must be 2P_1/2 and its g_J
#   the Lande value 2/3 - (g_s - 2)/3 to 1e-9 (spin-orbit mixes in other
#   J = 1/2 terms, 4P of 2p3 among them, which moves g_J by ~3e-11).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
cfp=$work/m0/build/CFPTBL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

p=$work/m5/f/prep
printf '33*2 34*5\n' > "$tmp/f_configs"
"$repo/c/build/acfs" "$cfp" "$p/fort.15" "$p/fort.16" "$p/fort.18" "$p/fort.21" \
    "$tmp/f_configs" 2 > "$tmp/f.out"
"$repo/c/build/acfs" "$cfp" "$p/fort.15" "$p/fort.16" "$p/fort.18" "$p/fort.21" \
    "$tmp/f_configs" 2 mvd > "$tmp/fm.out"
r=$work/s2m2
"$repo/c/build/acfs" "$cfp" "$r/ref/fort.15" "$r/ref/fort.16" "$r/ref/fort.18" \
    "$r/ref/fort.21" "$r/configs" 2 > "$tmp/b.out"
grep -v '^zeta' "$tmp/f.out" | sed 's/^/   F  /'
grep -v '^zeta' "$tmp/b.out" | sed -n '1,4p' | sed 's/^/   B  /'

python3 - "$tmp/f.out" "$tmp/b.out" "$tmp/fm.out" <<'PY'
import re, sys
GS = 2.00231930436256
f = open(sys.argv[1]).read()
b = open(sys.argv[2]).read()
fm = open(sys.argv[3]).read()
zeta = float(re.search(r"^zeta\(34,34\) = (\S+) hartree", f, re.M).group(1))
lev = {m.group(1): (float(m.group(2)), float(m.group(3)))
       for m in re.finditer(r"^ *(\d)/2 +1 +(\S+) 2P +\S+ +(\S+)", f, re.M)}
e12, g12 = lev["1"]
e32, g32 = lev["3"]
split = e12 - e32
# energies near -99 hartree carry ~1e-14 rounding; g_J is printed to 13 decimals
ok_f = split > 0 and abs(split - 1.5 * zeta) < 1e-12 and \
    abs(g12 - (2 / 3 - (GS - 2) / 3)) < 1e-12 and abs(g32 - (4 / 3 + (GS - 2) / 3)) < 1e-12
print(f"   F 2P: E(1/2) - E(3/2) = {split:.15e}, 1.5 zeta = {1.5 * zeta:.15e}, "
      f"g = {g12:.12f}, {g32:.12f}  {'ok' if ok_f else 'FAIL'}")
m = re.search(r"^levels relative.*\n +J = +(\d+)/2", b, re.M)
g0 = float(re.search(r"^ 1/2 +1 +\S+ 2P +\S+ +(\S+)", b, re.M).group(1))
ok_b = m.group(1) == "1" and abs(g0 - (2 / 3 - (GS - 2) / 3)) < 1e-9
print(f"   B: ground J = {m.group(1)}/2, g = {g0:.12f}  {'ok' if ok_b else 'FAIL'}")
# one CSF: the energy is linear in h, so adding mass velocity + Darwin to h
# shifts every level by sum_i n_i (MV + D)_ii (1s2 2s2 2p5 = orbitals 35, 33, 34)
d = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^mvd\((\d+),\1\) = (\S+)", fm, re.M)}
want = 2 * d[35] + 2 * d[33] + 5 * d[34]
lm = {m.group(1): float(m.group(2))
      for m in re.finditer(r"^ *(\d)/2 +1 +(\S+) 2P", fm, re.M)}
shift = [lm[j] - lev[j][0] for j in ("1", "3")]
ok_m = all(abs(s - want) < 1e-9 for s in shift)
print(f"   F mass velocity + Darwin: level shifts {shift[0]:.12f}, {shift[1]:.12f}, "
      f"sum n_i (MV + D)_ii {want:.12f}  {'ok' if ok_m else 'FAIL'}")
ok = ok_f and ok_b and ok_m
print("real atoms", "passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY
