#!/bin/bash
# run_v2b.sh -- S2-M2, check V2b: every LS term of a set of configurations
# from one C run (acterms) against the Fortran run separately per symmetry.
#
# System: the boron sample's basis and orbitals, 1s2 closed, odd-parity
# valence configurations 2s2 2p, 2p3, 2s 2p 3s, 2s 2p 3d, 2p2 3p, with no
# LS-range restrictions (a complete CSF space per symmetry).
#
# Per symmetry (2S+1, L) the Fortran runs AI, WF (RULE=0), TR, EX, CI with
# direct diagonalization (NSDIM=700) and RS. Compared:
#   * the number of CSFs (the Fortran tape's image vs the C block);
#   * the C block's lowest eigenvalue against acci on the Fortran run's own
#     CSFs and integrals (same space: expected to agree to ~1e-12);
#   * the same against the Fortran ET read in full from unit 37 (the
#     Fortran's own diagonalizer agrees with LAPACK only to ~1e-10).
#
#   AC_WORK  work root (default $HOME/atomci_work); needs the M0 build.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${AC_WORK:-$HOME/atomci_work}
ftn=$work/m0/build
cbin=$repo/c/build
root=$work/s2m2
rm -rf "$root"
mkdir -p "$root"

configs_atomci=(
    " 2S*2) 2P*1))"
    " 2P*3))"
    " 3S*1) 2S*1) 2P*1))"
    " 3D*1) 2S*1) 2P*1))"
    " 3P*1) 2P*2))"
)

make_stdin() { # 2S+1, 2L+1, output file
    local b=$repo/samples/native/boron.stdin
    local nab nci ntr
    nab=$(grep -n 'A=2, B=1' "$b" | head -1 | cut -d: -f1)
    ntr=$(grep -n '&TR' "$b" | head -1 | cut -d: -f1)
    {
        sed -n "1,${nab}p" "$b" | sed "s/ISPIN=2,IPAR=1,IORB=3,/ISPIN=$1,IPAR=1,IORB=$2,/"
        printf '%s\n' "${configs_atomci[@]}"
        echo " ))))"
        sed -n "${ntr},\$p" "$b" | sed 's/NSDIM=100/NSDIM=700/'
    } > "$3"
    grep -q "ISPIN=$1,IPAR=1,IORB=$2," "$3"
}

run_fortran() { # dir stdin
    mkdir -p "$1"
    ln -sf "$ftn/CFPTBL" "$1/CFPTBL"
    (cd "$1" && "$ftn/atomci.plain" < "$2" > out.output 2> out.err)
}

# 1. a reference run (2P) for the orbital list, tape header and integrals
make_stdin 2 3 "$root/ref.stdin"
run_fortran "$root/ref" "$root/ref.stdin"

# sequence numbers of the orbitals by n and l (NWTN = 256*n + 2l+1)
seqof() { # n, 2l+1
    "$cbin/nfte_dump" "$root/ref/fort.15" | awk -v want="$1:$2" '
        /^NWTN/ { for (i = 2; i <= NF; i++) if ($i == want) { print i - 1; exit } }'
}
s2s=$(seqof 2 1); s2p=$(seqof 2 3); s3s=$(seqof 3 1); s3p=$(seqof 3 3); s3d=$(seqof 3 5)
cat > "$root/configs" <<EOF
$s2s*2 $s2p*1
$s2p*3
$s3s*1 $s2s*1 $s2p*1
$s3d*1 $s2s*1 $s2p*1
$s3p*1 $s2p*2
EOF

# 2. all terms from one C run
"$cbin/acterms" "$ftn/CFPTBL" "$root/ref/fort.15" "$root/ref/fort.16" "$root/configs" 3 \
    > "$root/acterms.out"
cat "$root/acterms.out"

# 3. the Fortran per symmetry, and acci on each Fortran tape
echo
printf '%-6s %6s %6s  %-22s %-22s %-22s %9s %9s\n' term C_csf F_csf "E1 C (all terms)" \
    "E1 C on F's CSFs" "E1 Fortran (direct)" "|C-acci|" "|C-F|"
status=0
while read -r tag mult L n e1 rest; do
    [ "$tag" = term ] || continue
    d=$root/sym_${mult}_${L}
    make_stdin "$mult" $((2 * L + 1)) "$root/sym_${mult}_${L}.stdin"
    run_fortran "$d" "$root/sym_${mult}_${L}.stdin"
    fcsf=$("$cbin/nfte_dump" "$d/fort.15" | awk '/^CSF image/ { print $5 }')
    eacci=$("$cbin/acci" "$ftn/CFPTBL" "$d/fort.15" "$d/fort.16" 1 | awk '/^E1 =/ { print $3 }')
    ef=$(${PYTHON:-python3} "$repo/c/tests/m5/et_from_civ.py" "$d/fort.37")
    line=$(python3 - "$e1" "$eacci" "$ef" <<'PY'
import sys
c, a, f = map(float, sys.argv[1:])
print(f"{abs(c - a):9.1e} {abs(c - f):9.1e} {'ok' if abs(c - a) < 1e-11 and abs(c - f) < 1e-9 else 'FAIL'}")
PY
)
    printf '%-6s %6s %6s  %-22s %-22s %-22s %s\n' "$mult$L" "$n" "$fcsf" "$e1" "$eacci" "$ef" "$line"
    case $line in *FAIL*) status=1 ;; esac
    [ "$n" = "$fcsf" ] || status=1
done < "$root/acterms.out"
echo "V2b $([ $status = 0 ] && echo passed || echo FAILED)"
exit $status
