# Input manual for spin-dependent calculations (the C tools acatsp and acfs)

> Ready-to-run examples: [`c/examples/`](../c/examples/README.md);
> transition-metal examples for the Fortran ATOMCI: [`samples/native/`](../samples/native/README.md).
> Theory and validation: the [ATOMCI2 technical note](https://doi.org/10.5281/zenodo.24223209).

The C version of ATOMCI (`c/`) adds the Breit–Pauli operators to the CI of LS
terms, solves for the fine-structure levels in the LSJ basis, and prints
properties of each level (g_J, hyperfine structure, transitions, isotope
shift). There are two entry points. They differ only in where the orbitals
come from.

| Tool | Orbitals | Use |
|---|---|---|
| `acatsp` | ATSP's (MCHF) numerical orbitals, `.w` | Numerical HF or MCHF orbitals. CSFs come from a configuration list or an ATSP CSF list (`.c`). Only acatsp has `bio` (two states on separate orbitals) and `hfsrel` (the α² hfs correction) |
| `acfs` | Fortran ATOMCI orbitals (STF or GTF basis and MO coefficients) | Fine structure and properties on the orbitals of an ATOMCI SCF, MCSCF or CI run |

`make` in `c/` builds both into `c/build/`. Operators and output tables are
shared (`ac_lsj`).

## 1. A minimal run

```bash
cd c/examples/o2p4
../../build/acatsp - O.w cfg 9 > so1.out
```

- The first argument is the CFPTBL. With `-`, the CFP table is generated in
  C, and no Fortran CFPTBL is needed (a file may be given instead).
- `O.w` holds the orbitals of O 2p⁴ ³P written by ATSP's hf. `cfg` is the
  configuration list (here `1s(2) 2s(2) 2p(4)`).
- `9` is the number of levels printed per J (default 4).
- Operator and property options follow (§4). With none, the CI is the
  Coulomb H plus the one-electron spin-orbit H_SO1 in the LSJ basis.

## 2. acatsp input

```
acatsp CFPTBL atom.w configs [nlev] [options]
```

### 2.1 Orbitals `atom.w`

ATSP's `wfn.out` from `hf` or `mchf`, as written: P(r)/√r on the log grid
r = e^{−4+j/16}/Z, unformatted. The orbitals of the file become the shells
(slots) of every CSF, in file order. The `.w` files of the examples were
made by `c/examples/make_orbitals.sh`, which contains the input to ATSP's
hf.

- `orth` orthonormalizes the orbitals of each l on the grid first. ATSP's
  orbitals, once interpolated, are orthonormal only to about 10⁻⁹. Use it
  with `bio`.
- `lag=l:N:b[,...]` adds N analytic Laguerre-type orbitals per l (the large
  s sets of hfsrel).

### 2.2 Configurations `configs`

One configuration per line:

```
1s(2) 2s(2) 2p(6) 3s(2) 3p(6) 3d(2) 4s(2)
1s(2) 2s(2) 2p(6) 3s(2) 3p(6) 3d(3) 4s(1)
```

- Orbitals of the file that a line omits are empty. Closed shells are
  written out too.
- The CI includes every CSF of every (S, L) of every configuration, and
  builds the LSJ matrix per J. Configurations of both parities may be mixed
  (for E1 with `trans`).
- A file name ending in `.c` is read as an ATSP CSF list. The CI then uses
  exactly those CSFs, coupled as ATSP couples them.
- With `occslots`, a CSF's slots are only the occupied orbitals of its
  configuration. The CSFs are the same and the engine does less work.
- `writec=P` also writes the CSFs as ATSP `.c` files, one per (S, L):
  `P_<2S+1><L>.c`.

## 3. acfs input

```
acfs CFPTBL nfte nftri nfaii nftrc configs [nlev] [options]
```

acfs reads the files of a Fortran ATOMCI run:

| Argument | File | Content |
|---|---|---|
| nfte | `fort.15` | Energy-expression tape (orbital list, closed shells, integral addresses), written by the WF step |
| nftri | `fort.16` | MO integrals, written by the TR step |
| nfaii | `fort.18` | Basis (its header record) |
| nftrc | `fort.21` | MO coefficients |

acfs takes only the orbital list and the integral addressing from the tape,
not its CSFs. The CSFs come from `configs`. The CI and RS steps (items 7 and
8 of `&CNTL NRNFG`) may be switched off.

`configs` has one configuration per line, written as `seq*occ`. seq is the
1-based position in the tape's orbital list, counted along the `NWTN` line
of `nfte_dump fort.15`. Each `NWTN` entry is `n:2l+1`. Example
(`c/examples/tm_acfs/tape.out`, the Sc cc-pVTZ run):

```
NWTN 13:1 12:1 11:1 14:3 13:3 12:3 11:3 13:5 12:5 11:5 12:7 11:7 11:9 4:1 3:5 1:1 2:1 2:3 3:1 3:3
```

Position 14 is 4s (`4:1`), 15 is 3d (`3:5`), and 16 to 20 are the closed
1s 2s 2p 3s 3p. So Sc 3d 4s² is `14*2 15*1`, and Cu's 3d¹⁰4s and 3d⁹4s²
are two lines, `14*1 15*10` and `14*2 15*9`. The tape's default closed
shells are added when not listed. Positions 1 to 13 are the correlating
orbitals (named `11S` and so on).

acfs uses the tape's integrals (h, R^k). From the basis and MO coefficients
it also computes ζ, ⟨r^p⟩, mass velocity and the like by numerical
quadrature (`ac_radial`).

## 4. Options

★ marks acatsp only. `alpha=` is also acatsp only, though not marked; acfs
fixes 1/α at the CODATA 2018 value.

### 4.1 Hamiltonian (terms that enter the level energies)

| Option | Effect |
|---|---|
| (default) | Coulomb H plus the one-electron spin-orbit H_SO1 |
| `noso` | H_SO1 off: an LS CI, so g_J and the other properties are those of the LS terms |
| `so2` | Two-electron spin-orbit H_SO2 (spin-same-orbit + spin-other-orbit). ★ `so2part=1` gives spin-same-orbit only, `=2` spin-other-orbit only |
| `ss` | Spin-spin H_SS (tensor part) |
| `oo` | Orbit-orbit H_OO |
| `mvd` | One-electron mass velocity and Darwin, added to h (variationally) |
| `rkrel` ★ / `contact` | Two-electron Darwin and spin-spin contact terms, added to R^k. acatsp's `rkrel` is ATSP's relativistic R^k; with `mvd` it gives ATSP bpci's relativistic shift |
| `alpha=INV` ★ | 1/α (default 137.035999084, CODATA 2018; ATSP uses 137.036) |
| `atspcompat` ★ | Drops what ATSP's bpci drops: the two-electron H_SO2 and H_SS elements over four distinct orbitals. For side-by-side comparison with ATSP only |

The full Breit–Pauli set is `so2 ss oo mvd rkrel` (`so2 ss oo mvd contact`
in acfs).

### 4.2 Zeeman (g_J)

By default, each level gets g_J = g_L + g_s g_S plus the one-electron α²
corrections Δg_RMC, Δg_GC and Δg_ORMC. These options add columns:

| Option | Column | Effect |
|---|---|---|
| `gc2e` | dg_GC2e | Two-electron gauge correction (spin-same-orbit + spin-other-orbit). `gc2epart=1` or `=2` gives one part |
| `oz2e` | dg_OZ2e | Orbit-orbit Zeeman term |
| `rec2` | rec2 | ⟨O_z⟩/M_J of the two-body recoil operator Σ_{i≠j} r_i × p_j. The recoil correction is −(m/M)(g_L + rec2) |
| `h6` | dg_H6/kappa | Rank-2 part of Pachucki's H6 (PRA 69, 052502 (2004), Eq. (32)) per κ = (g_s − 2)/2, so Δg = κ·h6 |
| `zoff` | (separate table) | ⟨A‖μ‖B⟩ between printed levels with \|J_A − J_B\| ≤ 1: the elements that mix levels in an intermediate field |

The g_s-dependent factor of each Δg column (Pachucki's form) is given in
the technical note, Sec. 5.1. `c/tests/s2m7/o3p_gj.sh` applies them in its
comparison with experiment.

### 4.3 Hyperfine structure `hfs`

For every pair of printed levels, this prints the reduced matrix elements
⟨A‖T^(K)‖B⟩ (Edmonds, atomic units) of the electronic operators: orbital
(orb), spin-dipole (sd), contact (con), their sum T^(1), and the electric
quadrupole (quad, T^(2)). They are the Lindgren–Rosén operators, as in
ATSP's hfs. Diagonal rows also give A_J and B_J.

- `A_J(MHz)` is per μ_I/I = 1, with μ_I in nuclear magnetons. Multiply by
  μ_I/I.
- `B_J(MHz/b)` is per Q = 1 b. Multiply by Q in barns.
- The off-diagonal rows (J ≠ J′) feed second-order hfs and level mixing.

### 4.4 Transitions `trans`

For every pair of printed levels, this prints, for E1 (length and velocity
forms), M1, E2, M2, E3 and M3: the line strength S = |⟨A‖T‖B⟩|² in atomic
units, the E1 absorption oscillator strength f, and the rate A in s⁻¹. Only
the multipoles that parity allows appear: E1, M2 and E3 need opposite
parity, M1, E2 and M3 the same parity.

- The `M1r` line is M1 plus the one-electron relativistic correction
  (Pachucki 2004, Eqs. (59)–(62)). Its second column is ⟨‖δμ‖⟩ in μ_B.
- `m1r2e` adds the two-electron terms to M1r. `m1r2epart=1` gives the spin
  terms only, `=2` the orbital terms only.
- Both levels use one orbital set. For initial and final states on separate
  orbitals, use `bio` (§4.7).

### 4.5 Isotope shift `is`

For each printed level: K_NMS = ⟨Σ p_i²/2⟩, K_SMS = ⟨Σ_{i<j} p_i·p_j⟩,
their sum, and ρ(0) = ⟨Σ δ(r_i)⟩. The mass shift is (m/M)(K_NMS + K_SMS),
the field shift (2π/3) Z ρ(0) δ⟨r²⟩ (atomic units).

### 4.6 α² hfs correction `hfsrel` ★

For the lowest root of each (S, L = 0) block, this prints A_AN/A⁽⁴⁾ of
Puchalski and Pachucki (arXiv:1310.2161v2, Eq. (29)). It needs two things:

- The CSF list must be the complete CI of its orbitals, that is, its
  configurations are closed under single excitations.
- The orbitals must satisfy the nuclear cusp. ATSP's numerical orbitals do;
  Gaussians do not, which is why acfs has no hfsrel.

`delta12=X` also prints the value with ⟨Σδ(r_ab)⟩ = X in the
electron–electron term. See the technical note, Sec. 5.2.

### 4.7 Two states on separate orbitals `bio=W2,C2` ★

State 1 is the `atom.w` and `configs` arguments; state 2 is `W2` (a `.w`)
and `C2` (configurations or a `.c`). acatsp solves each CI on its own
orbitals and makes the two orbital sets biorthonormal (Malmqvist; Olsen et
al.). It then prints the `zoff`, `hfs` and `trans` tables between the levels
of state 1 (bra) and state 2 (ket).

- Order the shared orbitals the same way in both states. Per l, one state's
  orbitals must begin with the other's.
- The CSF lists must be closed under deexcitation. `biorthforce` goes on
  when they are not.
- `bio` cannot be combined with `ls`. Use `orth`.

### 4.8 Other options

| Option | Effect |
|---|---|
| `ls` ★ | No J: only the eigenvalues of each (S, L) block (`term 3F n E1 E2 ...` lines) |
| `dav=N` ★ | J blocks of dimension ≥ N are solved by Davidson (default 500; `dav=0` always uses dsyev) |
| `digits` ★ | `trans` line strengths with 17 digits |
| `q=...` ★ | Print radial integrals in ATSP's notation (`q=r,2p,-3`, `q=F,2,2p,2p`, `q=G,1,2s,2p`, `q=R,1,1s,2p,2s,2p`, ...) |

Environment variables:

- `OMP_NUM_THREADS`: threads.
- `AC_TIME`: wall time per phase, to stderr.
- `AC_TRANS_GS2`: g_s = 2 in M1, M2 and M3, for comparison with ATSP's
  trans.
- `AC_SEQ_CACHE_MB`, `AC_PROP_CACHE_MB`: cache sizes.

## 5. Reading the output

Take `c/examples/o2p4/g.out` as an example.

1. The first line summarizes the input: orbitals, configurations and CSFs,
   1/α, and which H terms are on.
2. Per-orbital lines: `orbital` in acatsp, `zeta` and `mvd` in acfs. They
   give ⟨h⟩, ⟨r⁻³⟩, ζ (hartree and cm⁻¹), mass velocity and Darwin.
3. The level table: 2J, the number within J, the energy (hartree), the
   leading LS term and its weight, g_J and the Δg columns (§4.2). J = 0
   levels have no g.
4. The `zoff`, `hfs`, `trans` and `is` tables, when requested. J_A and #A
   identify a level (2J and number).
5. `levels relative to the lowest (cm-1)`: every printed level, as energy
   above the lowest.

Units: energies in hartree; level spacings in cm⁻¹ (1 hartree =
219474.6313632 cm⁻¹); reduced matrix elements in atomic units (μ_B for the
magnetic ones); rates in s⁻¹. g_s = 2.00231930436256 and 1/α = 137.035999084
(CODATA 2018).

The sign of a matrix element between two levels (`zoff`, and the
off-diagonal `hfs` and `trans` rows) follows the phases of the two
eigenvectors, which can change with the LAPACK version. The physical
quantities (line strengths, diagonal A_J and B_J, g_J) do not depend on that
sign.

## 6. Examples

Each directory of `c/examples/` holds the inputs, a `commands` file (the
commands that were run), and the outputs (`*.out`).
`c/examples/run_examples.sh` reruns them all and compares the numbers with
the shipped outputs. It takes about 20 s in all, including the Fortran
preparation.

| Directory | System | What it shows |
|---|---|---|
| `o2p4` | O 2p⁴ ³P, ¹D, ¹S (HF) | `ls`; H_SO1 only; full Breit–Pauli; g_J with α² terms and `zoff`; hfs and isotope shift; M1 and E2 among ³P, ¹D and ¹S |
| `li` | Li 1s²2s ²S and 1s²2p ²P (each on its own HF) | hfs (contact; orbital and spin-dipole); ²P fine structure and g; M1r with two-electron terms; 2S–2P E1 through `bio` |
| `he2p` | He 1s2p ³P, ¹P (HF) | With H_SO1 only, J = 0 lies lowest; with H_SO2 and H_SS the order inverts and J = 0 lies highest. g with α² terms |
| `hfsrel` | Li 1s²2s ²S (n = 3 MCHF orbitals, complete CI) | The α² hfs correction A_AN/A⁽⁴⁾ |
| `sc` | Sc 3d4s² ²D (HF) | ²D fine structure; g_J; hfs (A_J, B_J); M1 and E2 between ²D₃/₂ and ²D₅/₂ |
| `ti` | Ti 3d²4s² ³F and others (HF); CI with 3d³4s and 3d⁴ added | LS terms (`ls`); ³F_{2,3,4} fine structure; g; hfs |
| `cu` | Cu 3d¹⁰4s ²S and 3d⁹4s² ²D (²S HF orbitals) | ²D fine structure; ²S–²D E2 and M1; hfs |
| `f2p_acfs` | F 2p⁵ ²P (Fortran ATOMCI MCSCF orbitals, GTO) | How to use acfs. `prep.sh` runs the Fortran on `samples/f.*` |
| `tm_acfs` | Sc, Ti, Cu (Fortran ATOMCI MCSCF orbitals in cc-pVTZ) | Transition metals with acfs. `prep.sh` runs the Fortran on the decks of `samples/native/` |

Caveats:

- The example orbitals are the HF of one term, extended a little at most.
  There is almost no correlation. The numbers show usage, not accuracy.
  Comparison with experiment needs MCHF or CI expansions; the report scripts
  in `c/tests/s2m7/` and `s2m8/` show how to build such series.
- `cu` computes ²D on the ²S orbitals, so ²D lies much higher above ²S
  than in reality.
- The same systems on ATSP's numerical orbitals (`sc`, `ti`, `cu`) and on
  cc-pVTZ MCSCF orbitals (`tm_acfs`) give fine-structure splittings within
  1% of each other. Sc ²D: 192.99 and 192.25 cm⁻¹. Ti ³F, J = 3 and 4:
  183.3 and 418.3 vs 182.8 and 417.0 cm⁻¹.
