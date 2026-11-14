# ATOMCI angular engine in C

The modules `ac_wigner`, `ac_ftnio`, `ac_cfp`, `ac_cfpgen`, `ac_csf`, `ac_csfgen`,
`ac_optree` and `ac_factor` now live in the library hexpr-atomci, with the new
`ac_terms` that gives the Hamiltonian expressions. This tree links it; `make`
builds it from `HEXPR_ATOMCI` (default `../hexpr-atomci`, the library's folder
in this repository), statically by default; `HEXPR_ATOMCI_LINK=shared` links
the shared library, with matrix elements about 35% slower.

This is the C rewrite of the angular part of ATOMCI, extended with the
Breit–Pauli terms and atomic properties. The formulation is in the ATOMCI2
technical note (Secs. 3 to 6).

Conventions follow libhexpr2: C11, `-Wall -Wextra -pedantic -Werror`, a
plain Makefile, and the prefix `ac_`. Angular momenta are passed as 2j
integers. ATOMCI's Fortran works with multiplicities (2j+1); the conversion
happens only at the boundary to it, which today means the tests.

## Modules

| Module | Contents | Fortran counterpart |
|---|---|---|
| `ac_wigner` | 6j (Racah sum), 9j (sum of 6j), the triangle factor, F(a,b,c), H^k | `xwig6j.f`, `delta.f`, `f3l.f`, FSUBK in `smat.f` |
| `ac_ftnio` | gfortran unformatted sequential records: read, write, cursor | — |
| `ac_cfp` | CFPTBL loading, shell-state codes 256·isq+lsub, the per-(l, N) state lists, XCFP | `setcfp.fpl`, `cfpset.fpl`, `isp.f`, `ior.f`, `xcfp.f` |
| `ac_csf` | One side of a CSF: shells, shell states, electron counts, intermediate couplings | the arrays SHMS receives |
| `ac_optree` | Operator trees of b and b~, recoupling rewrites (swap, rotations), the sequential form of METECC eq. (49), normal-ordering contraction | `SHMSRC`'s case table, `FACTOR` |
| `ac_factor` | ⟨F~ O G⟩ by METECC (47)-(51), single-shell elements by (48), and a SHMS-compatible entry point | `SHMS`, `SHMSRC`, `WT9X`, `LK_TABLE`, `CFACRC` |
| `ac_smat` | The Coulomb energy-expression terms of one CSF pair. A near line-for-line port: labels L<n> are the Fortran statement numbers, and SHMS is replaced by `ac_factor_shms` | `smat.f`, `convt.f` |
| `ac_addr` | Integral addresses on the tape (one-electron block, then R^k blocks), for both the CI layout and the MCSCF layout | `setiad.f` (SETIAD, SETKAD, INT1AD, INT2AD) |
| `ac_tape` | The energy-expression tape NFTE: reading its header, CSF image and data records; writing the data records. The type loop of EXTAPE, EXIJ (pair loop with coefficient reuse) and STORE/STOREH | `extape.fpl`, `exij.f`, `ntisqt.fpl` |

Stage 2 (spin and properties; technical note, Secs. 4 and 5):

| Module | Contents |
|---|---|
| `ac_csfgen` | All CSFs of a configuration, every (S, L). Empty and closed shells are explicit slots |
| `ac_ints` | The Fortran NFTRI integral file, laid out so that the position of a value is its `ac_addr` address |
| `ac_ci` | The non-relativistic CI: CSFs from a Fortran tape or from a configuration list; matrix elements as sums of METECC (44)/(45) operators through the engine, on the union of the two slot lists; LAPACK diagonalization |
| `ac_basis` | The Fortran's radial basis (STO or contracted GTO, from the header of NFAII, unit 18) and MO coefficients (unit 21); analytic basis-pair moments and h |
| `ac_radial` | MO radial functions on a grid uniform in ln r; h, ⟨r^p⟩, R^k (through Y^k potentials, 6-point cumulative rule), ζ, mass velocity and Darwin by numerical quadrature |
| `ac_lsj` | The CI in the LSJ basis: Coulomb H (per (S, L), from `ac_ci_hij`) plus the Breit-Pauli terms (H_SO1, H_SO2, H_SS by 6j to J; H_OO and the contact terms in H); LSJ reduction of any double-tensor rank (9j); reduced elements between levels: g_J with its α² terms (RMC, ORMC, GC1e, GC2e, OZ2e, H6, recoil), Zeeman couplings between levels (zoff), hyperfine A and B, transitions E1 (length, velocity), M1 (with the relativistic M1r), E2, M2, E3, M3, and the isotope-shift parameters. The two-electron property elements of CSF pairs are kept per run and reused across levels and J blocks |
| `ac_tabrad` | ATSP's numerical orbitals (`.w`, P/√r on r = e^{−4+j/16}/Z) as an `ac_radial`: r-polynomial near the origin, 8-point Lagrange in ln r outside |
| `ac_biorth` | Two states on separately optimized orbitals (Malmqvist; Olsen et al.): the LU-based biorthonormal orbitals, the CI counter-transformation, the check of closure under deexcitation (S2-M9) |
| `ac_hfsrel` | The regularized α² hfs correction of S states (Puchalski and Pachucki), with the mixed CI space and the resolution of the second-order term (S2-M11) |
| `ac_cfpgen` | The CFP table generated in C (M1b): a line-by-line port of cfpa and cfpb (SYMTLZ) whose result is byte-identical to the Fortran CFPTBL |

`ac_ci` also takes its integrals through a replaceable provider (`ac_ci_integrals`),
builds CSFs from a bare orbital list (`ac_ci_from_orbitals`), and gives reduced
elements of any one-electron double tensor (`ac_ci_one`), of the two-electron
spin-orbit operator (`ac_ci_so2`, spin-same-orbit + spin-other-orbit) and of the
spin-spin operator (`ac_ci_ss`), and matrix elements of the orbit-orbit
operator (`ac_ci_oo`). Their tensor forms are derived in the technical
note, Sec. 4.2 (spin-orbit and spin-spin by integration by parts and the
gradient formula; orbit-orbit through the transverse kernel; the
two-electron Darwin and spin-spin contact terms as `ac_radial_contact`);
`python/so2ref.py` is an independent
m-resolved implementation of the two-electron spin-orbit operator.

`ac_factor_op` caches the sequential form per operator tree, and
`ac_ninej` caches 9j values.

Tools (built by `make`):

| Tool | Purpose |
|---|---|
| `build/acgen CFPTBL in.nfte out.nfte` | Copies the header and CSF image of a Fortran tape byte for byte, then writes data records generated by the C engine |
| `build/nfte_dump [-v] tape` | Prints a tape: the header, the CSF types and a record summary (`-v`: every record) |
| `build/acci CFPTBL nfte nftri [nroots [cmp]]` | Stage-2 CI on a Fortran tape's CSFs. With `cmp`, also compares against the H of the tape's energy expression |
| `build/acterms CFPTBL nfte nftri configs [nroots]` | Every LS term of a configuration list in one run |
| `build/acrad nfaii nftrc nftri [h [rmax [dump]]]` | Radial integrals by quadrature, checked against NFTRI and analytic formulas (S2-M3) |
| `build/acfs CFPTBL nfte nftri nfaii nftrc configs [nlev] [options]` | Fine structure on the Fortran's GTO orbitals (Coulomb + H_SO1 in LSJ; `so2`, `ss`, `oo`, `contact`, `mvd`) and the same properties as acatsp (`gc2e`, `oz2e`, `rec2`, `h6`, `zoff`, `hfs`, `trans`, `m1r2e`, `is`); the list is at the top of `tools/acfs.c` |
| `build/acatsp CFPTBL atom.w configs|list.c [nlev] [options]` | The same LSJ CI on ATSP's numerical orbitals, on a configuration list or an ATSP CSF list. Hamiltonian: `noso`, `mvd`, `so2`, `ss`, `oo`, `rkrel` (contact terms = ATSP's relativistic R^k), `atspcompat` (what bpci omits), `alpha=INV`. Properties: `gc2e`, `oz2e`, `rec2`, `h6` (α² terms of g_J), `zoff`, `hfs`, `trans` (E1, M1, M1r, E2, M2, E3, M3; `m1r2e` the two-electron M1 terms), `is`, `hfsrel`. Two states: `bio=W2,C2` (with `biomode=`, `rot1=`, `rot2=`, `orth`). Others: `ls`, `writec=P`, `occslots`, `lag=`, `dav=N`, `digits`, `q=...`. The full list is at the top of `tools/acatsp.c` |
| `build/cfpgen OUT` / `build/cfpgen -c CFPTBL` | Writes the CFP table generated in C as CFPTBL, or compares it with a CFPTBL |

The input manual for acatsp and acfs (options, input files, how to read the
output) is `docs/C_SPIN_MANUAL.md`.
`examples/` holds inputs and outputs for every operator and property, on
light atoms and on Sc, Ti and Cu; `examples/run_examples.sh` reruns and
checks them.

Every tool that takes a CFPTBL argument accepts `-` instead of a path: the
table is then generated in C (`ac_cfp_generate`) and no Fortran file is needed.

Environment variables: `AC_TIME` (wall times of the phases to stderr),
`AC_SEQ_CACHE_MB` (the engine's sequential-form cache per thread, default
128), `AC_PROP_CACHE_MB` (the per-run tables of two-electron property
elements, default 2048), `AC_TRANS_GS2` (g_s = 2 in the M1 and M2 lines, for
comparisons with ATSP), `OMP_NUM_THREADS`.

The checks (pass or fail), the report-only comparisons with other programs
and the literature, what each needs and the environment variables are
listed in `tests/README.md`. `tests/run_all_checks.sh` runs every check in
turn (about 30 minutes, most of it `check-v5`). The results are summarized
in the technical note, Sec. 7.

`ac_factor` is the general-form engine (technical note, Sec. 3.5). It does
not port SHMSRC's case table. Instead it rewrites any operator tree
into sequential form with three rewrites. Operator ranks are kept general
throughout.

`ac_cfp` deliberately differs from the Fortran in two places (see
`ac_cfp.h`):

- **Shell row of `im`.** It is decoded as im/256. The Fortran uses im/255,
  which gives the same answer for every table cfpa produces.
- **Links to ungenerated levels.** A link value of -1 gives 0. The Fortran
  would index out of bounds there.

## Building and checking

Tested on Debian 13:

```bash
make            # build/libac.a
make check      # L0, L0-exact, L1, L2, L3, L3-addr, L4
```

The checks compare against the Fortran in the same process. They link
objects from a built `src/` tree and read its `CFPTBL`, plus the XCFP calls
recorded in M0 (`tests/trace/`). The default locations are the M0 scratch
build under `AC_WORK` (default `$HOME/atomci_work`):

| Variable | Default |
|---|---|
| `FTN_BUILD` | `$AC_WORK/m0/build` |
| `M0_RUN` | `$AC_WORK/m0/run` |

Run `tests/trace/build_trace.sh` and `tests/trace/run_trace.sh` first.

| Check | What it compares | Result (2026-10-26) |
|---|---|---|
| `check-l0` | 6j for every 2j ≤ 12 (4,826,809 cases, valid or not); 9j for every triangle-valid 2j ≤ 8 (1,079,573); DELTA; F(a,b,c); H^k for l ≤ 9, k ≤ 18, all against the Fortran | max \|C−F\| ≤ 3.6e-15 (H^k; ≤ 3.3e-16 for 6j/9j) |
| `check-l0-exact` | 6j for 2j ≤ 10 and 9j for 2j ≤ 6, against exact rational Racah sums at 50 digits (independent of the Fortran). Each error is measured against the sum of the absolute summands, because a small 9j built from larger 6j products carries ~1e-16 absolute rounding error | max \|C−exact\| = 4.3e-16, max ratio to the scale 2.9e-15 |
| `check-l1` | All five CFPTBL records; the 54 (l, N) state lists (839 entries) against /COMWFA/; ISP, IQS, ILFO, IOR and LSQN for all 368 codes and 64 blocks; XCFP for every connected pair, id = ±1 (110,776 calls); replay of the 191 XCFP calls recorded in M0 | all equal. XCFP agrees bit for bit (max difference 0) |
| `check-l2` | Every SHMS call recorded in M0 (32,166: boron and f), recomputed from its arguments, both as the plain product and normal-ordered; grouped by the order pattern of the operator shells | all match the normal-ordered value, max \|C−F\| = 1.7e-14, across 47 patterns. The two values differ only when all four operators sit on one shell, and there SHMS is normal-ordered |
| `check-l3` | Every SMAT call recorded in M0 (15,242), replayed through `ac_smat`: the seven outputs (NOE, N1OE, NHTR, NFSR, NHDTX, NWST, NVST) and every emitted term (number, KM, four slots, coefficient) | all equal. 20,151 terms; max \|WT_C − WT_F\| = 5.2e-14 |
| `check-l3-addr` | `ac_addr` against the Fortran in the same process, on the 3 orbital sequences of the recordings. Each sequence is set up 4 ways (SETIAD and SETKAD × automatic and reversed NWSQ). Compared: LMAX, ISSH, LISLU, IGCD/ISCD, and every INT1AD and every triangle-allowed INT2AD address | all equal: 3,620 INT1AD and 8,847,768 INT2AD addresses |
| `check-l4` | Each M0 tape (`fort.15`) is regenerated with `acgen` and compared by meaning (`tests/nfte_diff.c`). The header and CSF image must be byte-identical. The data records are reduced to a map (row, column, integral address) → sum of coefficients, so record boundaries and order do not matter | boron (EXTAPE path): 173,057 entries, max \|A−B\| = 2.3e-14. f (WFETAP path): 1,233,228 entries, max 5.1e-14. No key missing on either side. The boron tape also has the same size and the same records as the Fortran one; only last-bit rounding of coefficients differs |

### L5: the CI energy on a C tape

`tests/m5/run_m5.sh` runs the Fortran for each sample up to the tape and
the integral transformation. From that same preparation it restarts CI + RS
twice: once on the Fortran tape (the control) and once on the tape written
by `acgen`. The control must reproduce the golden energy; the C-tape run
must agree with it to 1e-9.

The printed f energy has only 8 decimals, so ET is read in full precision
from the CI vector file (unit 37, `tests/m5/et_from_civ.py`).

| Sample | Control ET | C-tape ET | \|diff\| |
|---|---|---|---|
| boron (EXTAPE path) | −24.646195396688494 | −24.646195396688565 | 7.1e-14 |
| f (WFETAP path) | −99.64122309909808 | −99.64122309909841 | 3.3e-13 |

The other differences between the two outputs come from rounding in the
matrix elements: Davidson residual norms differ near 1e-7, and the
natural-orbital coefficients in the 8th decimal.

The LSQN closed form assumes that every parity-allowed (Q, S) block of a
shell exists. L1 confirms this holds for all 64 blocks of the current
CFPTBL.
