# Tests of the C tools

Each test runs on its own. There are two kinds:

- **Checks** (`make check-*` in `c/`): pass or fail.
- **Comparisons** (`bash c/tests/<dir>/<script>.sh`): report-only scripts.
  They print C's values next to another program's or the literature's, and
  you read the agreement from the output.

`tests/run_all_checks.sh` runs every check in turn. It is only a convenience;
running the checks one by one is the normal way.

A test whose external program (ATSP, GRASP, Dalton, ORCA, PySCF) is not found
prints one line and ends with status 0, without running anything:

```
SKIPPED run_atsp_bp.sh: ATSP_SRC is not set (the ATSP source tree, with examples/)
```

`run_all_checks.sh` reports such a target as `skipped`, not `passed`. The
helper functions are in `lib/need.sh`.

## Names used in the tests

The C tools were written in stages, and the tests keep the stage names as
labels: in target names (`check-s2m7`), directory names (`s2m7/`) and the
comments of the scripts and sources.

| Label | What it covers |
|---|---|
| L0 to L5 | Layers of the comparison with the Fortran ATOMCI: L0 Wigner symbols, L1 the CFP table and XCFP, L2 single-shell matrix elements (SHMS), L3 the Coulomb energy-expression terms (SMAT) and integral addresses, L4 the energy-expression tape, L5 the CI energy on a tape written in C |
| M0 | The traced Fortran build and runs that L1 to L4 read (`trace/`) |
| M1 to M5 | The C modules that pass L1 to L5 (M5: `m5/run_m5.sh`) |
| M1b | The CFP table generated in C (`check-cfpgen`) |
| S2-M0 | Conventions and PythonRef (`c/python`) |
| S2-M1 | The general engine for operators of any rank; explicit closed shells |
| S2-M2 | All LS terms of a configuration list |
| S2-M3 | Radial integrals |
| S2-M4 | H_SO1, the LSJ CI and the first-order and one-electron α² terms of g_J |
| S2-M5 | H_SO2 and H_SS |
| S2-M6a, S2-M6b | Mass velocity and Darwin; the two-electron contact terms and orbit-orbit |
| S2-M7 | The two-electron α² terms of g_J, Zeeman couplings between levels, H6, recoil |
| S2-M8 | Hyperfine A and B; transitions E1 to M3; the relativistic M1 corrections |
| S2-M9 | The biorthonormal transformation; the Davidson solver |
| S2-M10 | Isotope-shift parameters |
| S2-M11 | The α² correction of the hyperfine constant of S states |
| V1 to V5 | Kinds of reference: V1 PythonRef, V2 the Fortran ATOMCI, V3 analytic radial integrals, V4 analytic limits (Landé, jj, hydrogen-like), V5 a determinant CI on PySCF integrals in the same basis and orbitals |

Labels such as Z6 or P5 in the headers of the comparison scripts number the
items of the development record; the header of each script says what it
compares and with which reference.

## First: the Fortran build that the tests read

Most tests read the CFP table `CFPTBL` of a built Fortran tree, and the
checks of stage 1 also read recordings of the Fortran's calls. Make them once:

```bash
bash c/tests/trace/build_trace.sh     # builds src/ under $AC_WORK/m0/build
bash c/tests/trace/run_trace.sh       # runs the two samples under $AC_WORK/m0/run
```

`check-s2m2`, `check-s2m3`, `check-s2m4` and `check-v5` need `bash c/tests/m5/run_m5.sh` first.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `AC_WORK` | Work root of every test; the Fortran build is `$AC_WORK/m0` | `$HOME/atomci_work` |
| `FTN_BUILD`, `M0_RUN` | The Fortran build and the M0 runs (make only) | `$AC_WORK/m0/build`, `$AC_WORK/m0/run` |
| `PYTHON`, `PYREF_PYTHON` | Python with numpy (the checks), plus scipy and PySCF (V5) | `python3` |
| `ATSP_BIN` | ATSP's `bin/` (hf, mchf, nonh, lsgen, bpci, hfs, trans, biotr) | `$HOME/atsp_work/atsp/bin` |
| `ATSP_SRC` | ATSP's source tree, for its `examples/` (atsp/) | none: skipped when unset |
| `GRASP_BIN` | GRASP's `bin/` (rcsfgenerate, rmcdhf, rci, rhfszeeman95, ...) | `$HOME/grasp_work/grasp/bin` |
| `DALTON_BUILD` | Dalton's build directory, built by `dalton/build_dalton.sh` with `dalton/rspg_digits.patch` | `$HOME/dalton_build` |
| `DALTON_SRC` | Dalton's source tree, for `basis/aug-cc-pCVQZ` (first run of `run_dalton.sh`) and for `build_dalton.sh` | none |
| `ORCA_DIR` | The ORCA 6 directory | none: skipped when unset |
| `OMP_NUM_THREADS` | Threads of the C tools | all cores |

Several comparisons also take the work directory of an earlier script (for
example `LI_WORK`, `O_WORK`, `HE_WORK`); the header of each script lists its
variables and which script must run first. Their defaults are under
`$AC_WORK`. A fresh `AC_WORK` therefore reruns everything from the beginning.

## Checks (pass or fail)

| Target | Compares | Needs |
|---|---|---|
| `check` | L0 to L4: Wigner symbols, CFP, SHMS, SMAT, addresses and tapes against the Fortran in the same process | the Fortran build and recordings |
| `check-l0-exact` | 6j and 9j against exact rational sums | Python |
| `check-rank` | non-scalar one-body total ranks | |
| `check-cfpgen` | the CFP table generated in C against `CFPTBL`, byte for byte | |
| `check-pyref`, `check-s2m1` | the general engine against PythonRef (`c/python`) | numpy |
| `check-s2m2` | all LS terms in one run against the Fortran | `m5/run_m5.sh` first |
| `check-s2m3` | radial integrals | numpy, scipy; `m5/run_m5.sh` first |
| `check-s2m4` | H_SO1 and the Zeeman factors against PythonRef, analytic limits, real atoms | numpy; `m5/run_m5.sh` first |
| `check-s2m6` | mass velocity and Darwin against the Dirac energy of hydrogen-like ions | |
| `check-s2m6b` | orbit-orbit against closed forms for Gaussian orbitals | |
| `check-s2m7` | the orbit-orbit Zeeman term against a momentum-space reference | numpy |
| `check-s2m9` | the biorthonormal transformation and the Davidson solver | |
| `check-s2m11` | the regularized α² hfs correction against the Python prototype | numpy |
| `check-v5` | a determinant CI on PySCF integrals in the same basis and orbitals (about 25 minutes) | PySCF; `m5/run_m5.sh` first |
| `check-atsp` | ATSP in its own HF orbitals | ATSP, `ATSP_SRC` |
| `check-library` | the library hexpr-atomci alone, and its PySCF example | PySCF for the example |

## Checks of the formulas

`theory/` checks the formulas of the technical note (Secs. 3 to 5) without
the C code. Each formula is compared with an independent reference: exact
symbols, PythonRef's determinant space, or coordinate-space operators on a
grid (`theory/theory_coord.py`).

```bash
PYTHON=... bash c/tests/theory/run_theory.sh             # all, about 20 minutes
PYTHON=... bash c/tests/theory/run_theory.sh bp2e multipole    # some
PYTHON=... CONTROLS=1 bash c/tests/theory/run_theory.sh  # also the controls, about 15 minutes more
```

The controls flip one relative sign per operator; each must fail. The script
header lists what each check covers. The checks need numpy and scipy; `rme`
also needs sympy and is skipped without it.

## Comparisons (report only)

Run them from anywhere, e.g. `bash c/tests/s2m7/run_he.sh`. Times are for 4
threads on the author's PC.

| Directory | Scripts | Compared with | Needs |
|---|---|---|---|
| `atsp/` | `time_bp.sh` | the wall time of the full Breit–Pauli build | `ATSP_SRC` |
| `s2m6b/` | `run_oo.sh` (pass or fail; also `make check-s2m6b`) | orbit-orbit: closed forms for Gaussian orbitals | |
| `s2m7/` | `run_he.sh`, `he23s_g.sh`, `he23p_mchf.sh`, `he33p_mchf.sh`, `he23p_g.sh`, `he33p_g.sh` | He g_J: Puchalski–Jentschura, Yan–Drake, Anthony–Sebastian | ATSP |
| `s2m7/` | `li2p_yan.sh`, `li2p_zoff_yan.sh` | Li-like 2P g_J and Zeeman couplings: Yan (2002) | ATSP |
| `s2m7/` | `f2p_gj.sh`, `o3p_gj.sh`, `o3p_ic_atsp.sh` | F and O g_J: experiment, Beltrán-López et al., ATSP (about 80 minutes with `IC4=1`) | ATSP |
| `s2m7/` | `h6_hydro.sh` | H6 rank 2: Jentschura (2010) | ATSP |
| `grasp/` | `run_gj.sh`, `run_lilike.sh`, `fo_grasp_corr.sh`, `li2p_grasp_corr.sh`, `blike_corr.sh`, `blike_grasp_corr.sh`, `li2s_corr.sh` | GRASP g_J (no-pair Dirac–Coulomb) | GRASP, ATSP |
| `orca/` | `run_epr.sh` | ORCA g shifts of S states | ORCA, ATSP |
| `dalton/` | `run_dalton.sh` | Dalton ROHF g shifts of S states | Dalton, ATSP |
| `s2m8/` | `li_hfs.sh`, `li_e1.sh`, `trans_check.sh`, `he_m1.sh` | hfs and transitions: hydrogen, ATSP, Hylleraas values, Pachucki | ATSP |
| `s2m8/` | `f_dk3.sh` | the Fortran's DK3 one-electron Hamiltonian read by acfs | |
| `s2m9/` | `bio_li.sh` (pass or fail), `bio_atsp.sh`, `li_phys.sh`, `bi_ogjmf.sh` | the biorthonormal transformation: ATSP biotr, Yan–Tambasco–Drake, Olsen et al. | ATSP |
| `is/` | `is_atsp.sh` | isotope-shift parameters: ATSP mchf | ATSP |
| `s2m11/` | `li2s_hfsrel.sh` | the α² hfs correction of Li 2S: Puchalski–Pachucki | ATSP, numpy |

Some scripts reuse the outputs they find in their work directory, for example
C's `ac_n.out`, so that a second run reads instead of computing. Delete them,
or use a fresh `AC_WORK`, to compute again.
