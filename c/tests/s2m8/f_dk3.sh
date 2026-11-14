#!/bin/bash
# f_dk3.sh -- S2-M8 (P7), report only (Fortran ATOMCI needed): the scalar
# Douglas-Kroll-Hess one-electron H of the Fortran (&AI NTYPE=4, gDKsapporo;
# docs/INPUT_MANUAL.md) fed to acfs unchanged, on the f sample
# (F 1s2 2s2 2p5 2P, Hartree-Fock in the GTO basis samples/native/f.bas, one CSF).
# Variants: NR (NTYPE=2), DK3 with a finite (Gaussian) nucleus, DK3 and DK2
# with a point nucleus; and NR, DK3 again with the tight s (5 primitives)
# and p (2 primitives) contractions of f.bas opened ("u").
# Printed per variant: the Fortran HF energy, the acfs levels (one-electron
# spin-orbit; acfs reproduces the HF energy as the centroid), the splitting
# E(1/2) - E(3/2) = 1.5 zeta(2p) and the alpha^2 g_J columns.
# Reference: the DK3-SCF energy of F, -99.49658346 hartree (B-spline limit;
# Nakajima-Hirao -99.49604040), in src/gDKsapporo/DKTAB.pdf, Table I, and
# the nonrelativistic HF limit -99.4093491. "mvd" (Breit-Pauli mass
# velocity + Darwin added to h) is the first-order relativistic energy on
# the NR orbitals; it must not be added to a DK H (double counting).
#   AC_WORK (default $HOME/atomci_work; needs m0/build and m5/f.prep.stdin
#   from c/tests/m5/run_m5.sh)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
ftn=$work/m0/build
prep=$work/m5/f.prep.stdin
root=$work/dk3
[ -s "$prep" ] || { echo "f_dk3.sh: run c/tests/m5/run_m5.sh first ($prep)"; exit 1; }
mkdir -p "$root"
printf '33*2 34*5\n' > "$root/f_configs"
awk '
/^  S  5  1.0/ { getline e; getline c; n = split(e, x)
                 for (i = 1; i <= n; i++) printf "  S    1  1.0  %s  1.000000\n", x[i]; next }
/^  P  2  1.0/ { getline e; getline c; n = split(e, x)
                 for (i = 1; i <= n; i++) printf "  P  1  1.0     %s  1.000000\n", x[i]; next }
{ print }' "$repo/samples/native/f.bas" > "$root/f_unc.bas"
for v in NR DK3 DK3pt DK2pt NRu DK3u; do
    d=$root/$v
    rm -rf "$d"
    mkdir -p "$d"
    ln -sf "$ftn/CFPTBL" "$d/CFPTBL"
    case $v in
        NR | NRu) ai="NTYPE=2,";;
        DK3 | DK3u) ai="NTYPE=4, ifdk3=.TRUE., ifnucl=.TRUE.,";;
        DK3pt) ai="NTYPE=4, ifdk3=.TRUE., ifnucl=.FALSE.,";;
        DK2pt) ai="NTYPE=4, ifdk3=.FALSE., ifnucl=.FALSE.,";;
    esac
    if [ "${v%u}" != "$v" ]; then
        # 17 s and 12 p functions; the transformation keeps the 13 s and 11 p
        # MOs of the tape (NOB, MPOB unchanged)
        cp "$root/f_unc.bas" "$d/f.bas"
        sed "s/^ &AI IUNIT=3, Z=9.0D0, NTYPE=2,/ \&AI IUNIT=3, Z=9.0D0, $ai/;
             s/ &TR NSYM=4, NBF=13,11,6,5,/ \&TR NSYM=4, NBF=17,12,6,5,/" "$prep" > "$d/run.stdin"
        grep -q 'NBF=17,12,6,5' "$d/run.stdin"
    else
        cp "$repo/samples/native/f.bas" "$d/f.bas"
        sed "s/^ &AI IUNIT=3, Z=9.0D0, NTYPE=2,/ \&AI IUNIT=3, Z=9.0D0, $ai/" "$prep" > "$d/run.stdin"
    fi
    grep -q "$ai" "$d/run.stdin"
    (cd "$d" && "$ftn/atgen" < "$repo/samples/native/f.atg" > /dev/null &&
        "$ftn/atomci.plain" < run.stdin > run.output 2> run.err)
    "$repo/c/build/acfs" "$ftn/CFPTBL" "$d/fort.15" "$d/fort.16" "$d/fort.18" "$d/fort.21" \
        "$root/f_configs" 2 > "$d/acfs.out"
    case $v in NR | NRu)
        "$repo/c/build/acfs" "$ftn/CFPTBL" "$d/fort.15" "$d/fort.16" "$d/fort.18" "$d/fort.21" \
            "$root/f_configs" 2 mvd > "$d/acfs_mvd.out";;
    esac
done
python3 - "$root" <<'PY'
import re, sys
root = sys.argv[1]
dktab, hf_lim = -99.49658346, -99.4093491
def lev(path):
    t = open(path).read()
    return {m.group(1): (float(m.group(2)), [float(y) for y in m.group(3).split()])
            for m in re.finditer(r"^ *(\d)/2 +1 +(\S+) 2P +\S+ +\S+ +(.*)$", t, re.M)}
res = {}
for v in ("NR", "DK3", "DK3pt", "DK2pt", "NRu", "DK3u"):
    e = float(re.search(r"Total Energy = +(\S+)", open(f"{root}/{v}/run.output").read()).group(1))
    l = lev(f"{root}/{v}/acfs.out")
    res[v] = (e, l["1"][0] - l["3"][0], l)
    cen = (2 * l["1"][0] + 4 * l["3"][0]) / 6
    print(f"   {v:6s} E_HF {e:.8f}  acfs centroid - E_HF {cen - e:+.1e}  split {res[v][1] * 219474.6313632:.4f} cm-1"
          f"  alpha^2 g columns (3/2): {' '.join(f'{x:.6e}' for x in l['3'][1])}")
for nr, dk in (("NR", "DK3"), ("NRu", "DK3u")):
    m = lev(f"{root}/{nr}/acfs_mvd.out")
    mvd = m["3"][0] - res[nr][2]["3"][0]
    d = res[dk][0] - res[nr][0]
    print(f"   {dk:5s}: E_rel = E(DK3) - E(NR) {d:.6f}  (DKTAB - HF limit {dktab - hf_lim:.6f}: {d / (dktab - hf_lim) - 1:+.2e})"
          f"  first-order MV+D {mvd:.6f}  E(DK3) - DKTAB {(res[dk][0] - dktab) * 1e3:+.3f} mEh"
          f"  (NR basis error {(res[nr][0] - hf_lim) * 1e3:+.3f} mEh)  split DK3/NR - 1 {res[dk][1] / res[nr][1] - 1:+.3e}")
print(f"   finite nucleus (DK3 - DK3pt) {(res['DK3'][0] - res['DK3pt'][0]) * 1e3:+.4f} mEh,"
      f"  DK3pt - DK2pt {(res['DK3pt'][0] - res['DK2pt'][0]) * 1e3:+.4f} mEh")
PY
