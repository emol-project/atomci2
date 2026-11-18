# ATOMCI2

**ATOMCI2** is a configuration interaction (CI) program for many-electron
atoms. It computes atomic electronic structure in the Russell–Saunders (LS)
coupling scheme, building the CI Hamiltonian matrix from the vacuum
expectation values of tensor operators and their recoupling transformation
coefficients. Its capabilities include full use of atomic symmetry, STF
(l ≤ 9) or GTF (l ≤ 7) basis sets, SCF and MCSCF orbital optimization,
general CI or SD-CI, perturbative configuration selection, and
natural-orbital / first-order density-matrix analysis.

ATOMCI2 is based on **ATOMCI**, the non-relativistic CI program developed at
Hokkaido University by **F. Sasaki, M. Sekiya, T. Noro, K. Ohtsuki, and
Y. Osanai**. Its theoretical basis is described in F. Sasaki,
*Int. J. Quantum Chem.* **8**, 605 (1974), and in Sasaki, Sekiya, Noro,
Ohtsuki & Osanai, *"Non-Relativistic Configuration Interaction Calculations
for Many-Electron Atoms: ATOMCI"*, pp. 163–219 in *Methods and Techniques in
Computational Chemistry (METECC-94)* (ed. E. Clementi, 1993).

The original ATOMCI project continues in ATOMCI2. The first release of
ATOMCI, [emol-project/atomci](https://github.com/emol-project/atomci)
(v1.0.0), is deprecated and no longer supported; maintenance and further
development of ATOMCI are done in this repository.

To ATOMCI, ATOMCI2 adds:

- **an intermediate-coupling CI with the Breit–Pauli terms**: mass velocity,
  Darwin, one- and two-electron spin-orbit, spin-spin, orbit-orbit, and
  contact terms;
- **atomic properties**: g_J with its α² terms, hyperfine constants A and B,
  transitions E1 to M3, isotope-shift parameters, the biorthonormal
  transformation of two states on separate orbitals, and the α² hfs
  correction of S states;
- **matrix elements as expressions** (angular coefficients times symbolic
  integrals);
- **effective Hamiltonians of 3dᴺ ions**, built from the CI matrices and
  projected onto the operators of the parametric theory.

ATOMCI2 has three parts and an application package:

- **the Fortran program ATOMCI**, the same as in emol-project/atomci;
- **the C tools `acatsp` and `acfs`**, which do the Breit–Pauli CI and the
  properties above. `acatsp` runs on ATSP's numerical orbitals, `acfs` on
  the orbitals of a Fortran SCF or MCSCF run over Slater- or Gaussian-type
  functions;
- **the library hexpr-atomci**, the angular engine of the C tools, which
  gives the matrix elements as expressions;
- **the effective-Hamiltonian package `heff/`** (Python), which runs the
  Fortran ATOMCI and reads its CI matrices.

The C code reimplements ATOMCI's angular method (the two papers above) and
was checked against the Fortran, other programs and the literature; the
formulation and the checks are in the [ATOMCI2 technical note](https://doi.org/10.5281/zenodo.24223209).

## Status

ATOMCI builds and runs with gfortran 9 or later and the system BLAS/LAPACK.
The C parts need a C11 compiler with OpenMP and LAPACK, and `heff/` needs
Python 3.9 or later with numpy, scipy and sympy
([`heff/requirements.txt`](heff/requirements.txt)). See
[`INSTALL.md`](INSTALL.md) for details.

## License / attribution

Copyright © Fukashi Sasaki, Masahiro Sekiya, Takeshi Noro, Kazumasa Ohtsuki,
You Osanai, Nobuaki Miura, and Hiroaki Honda, 1990–2003, 2026.

The C port and the spin-dependent extensions were implemented by H. Honda,
with review and supervision by N. Miura.

Permission is granted to use, copy, modify, and redistribute this program and
its documentation, in source or binary form, for **non-commercial research and
educational purposes.** It may not be sold, licensed for a fee, included in a
commercial product, or otherwise commercially exploited, in whole or in part.

The methods and algorithms are free to study and to implement independently:
an independent implementation may use any license, including a commercial one,
and may resemble ATOMCI in structure, as long as it does not copy, translate, or
adapt ATOMCI's code. See Section 3(g) of [`LEGENDS.txt`](LEGENDS.txt).

Any publication based upon results obtained from this material must include the
attribution:

> This work is based, in part, on results from the ATOMCI package.

and cite the two papers listed above. Results of the C tools should also
cite the ATOMCI2 technical note. See [`LEGENDS.txt`](LEGENDS.txt) for the
full terms, including the system BLAS/LAPACK note; please read it, and use ATOMCI only if you agree to
them.

## Quick start

```sh
make
make install prefix=$HOME/atomci-local   # default prefix: ./local
export ATOMCI_HOME=$HOME/atomci-local    # `make install` prints this line

samples/native/run_native.sh             # runs nine examples, prints OK or FAIL
samples/inp/run_inp.sh                   # the same nine from the .inp form
```

[`docs/TUTORIAL.md`](docs/TUTORIAL.md) goes through this step by step and
shows how to run a calculation by hand and check the answer.

The C parts:

```sh
make -C hexpr-atomci && make -C c
bash c/examples/run_examples.sh          # each example: ok or FAIL
```

The examples generate the CFP table in C and need no Fortran file, except
`f2p_acfs` and `tm_acfs`, which first run the Fortran ATOMCI and need
`ATOMCI_HOME` (`make install` above). The input of the C tools is in
[`docs/C_SPIN_MANUAL.md`](docs/C_SPIN_MANUAL.md).

## Effective Hamiltonians of 3dᴺ ions (heff/)

[`heff/`](heff/README.md) holds Python code that builds the spin-free
effective Hamiltonian H_eff of d³ and d⁶ ions from ATOMCI CI matrices,
projects it onto the operators of the parametric theory, and compares the
results with fits of the NIST levels. It runs the Fortran ATOMCI of this
repository and does not modify it. Three small examples run from scratch
with the repository alone:

```sh
make heff-test                           # its tests (~15 s, no external data)
make heff-examples                       # three examples from scratch (after make install; ~8 min)
```

The H_eff data of the production spaces will be published separately;
with them, `make heff-tables` rebuilds the result tables without ATOMCI
(see [`heff/README.md`](heff/README.md)).

## Two input forms

ATOMCI accepts a calculation in either of two forms:

1. **NAMELIST deck** — the program's own input: an `OPEN` header, one
   NAMELIST group per step (`&CNTL`, `&AI`, `&MC`, `&WF`, `&TR`, `&PT`,
   `&EX`, `&CI`, `&RS`) and card data (basis, CSF list, orbitals). For an
   SD-CI the CSF list is written by `atgen` from a short `.atg` input.
   Every calculation ATOMCI can do can be written this way.
2. **`.inp` file** — a short, indented description (atom, basis file,
   core/active/correlating orbitals, references, steps to run) from which
   `atinpgen` writes the deck **and** its CSF list, checking the input
   first. It covers SCF/MCSCF and single- or multi-reference SD-CI, with
   or without perturbative selection, in Gaussian or Slater basis sets,
   over any number of passes (e.g. a second CI on natural orbitals).
   Anything else can still be written in an `.inp` as raw NAMELIST
   blocks.

Every example is shipped in both forms (`samples/native/` and `samples/inp/`,
each a closed set of inputs), with the same answers.

## Known boundaries

- **Four samples make up the regression suite** (`boron`, `boron_nsdim700`,
  `f`, `fpt`; see [INSTALL.md → "Verifying the build"](INSTALL.md#verifying-the-build)).
  `boron_nsdim700` has no historical reference: it exercises a code path
  (direct, non-Davidson diagonalization) the others do not reach, and its
  answer comes from this distribution's own run (gfortran 14).
- **52 Fortran `READ`/`WRITE` statements carry an `ERR=` handler with no
  matching `END=`**, across `src/*.f`. In Fortran, `ERR=` alone does not
  catch end-of-file, so any of these that reach end-of-file stop with a
  runtime error. One is known to do so (`RDCP`, reached via `&GET NIND=0`);
  the others have not been examined.
- **`&TR ITRNS≠2` is not usable.** It selects a one-electron transformation
  path (`TRNONA`/`TRNONE`) that reads a file nothing in the program writes;
  it stops with an end-of-file error in `TRMAIN`.

The C tools:

- **Shells with l > 3 hold at most two electrons.** The C tools use the
  same CFP table as the Fortran
  ([INPUT_MANUAL.md §18](docs/INPUT_MANUAL.md#18-limits-and-pitfalls)).
- **They do not optimize orbitals.** The orbitals are read from ATSP's
  MCHF (`acatsp`) or from a Fortran MCSCF (`acfs`).
- **Relativistic effects are those of the Breit–Pauli Hamiltonian**, to
  order α².

## Documentation

| Document | Contents |
|----------|----------|
| [Program description and input guide](https://doi.org/10.5281/zenodo.24045980) | **Guide of the Fortran ATOMCI** (PDF, on Zenodo): the structure of the program, its input and output, and the test calculations |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | **Tutorial**: clone, build, install, run the examples in both input forms, check the answers |
| [docs/INPUT_MANUAL.md](docs/INPUT_MANUAL.md) | **Input manual**: the deck, every NAMELIST group a calculation needs, basis and CSF-list formats, `atgen` and `atinpgen` input, output, restarts, limits, worked examples |
| [INSTALL.md](INSTALL.md) | Requirements, build, installation, verifying the build, platform notes |
| [samples/README.md](samples/README.md) | The examples |
| [docs/ATINPUTS_PY.md](docs/ATINPUTS_PY.md) | The `scripts/atinputs` generator (`atinpgen`) in full: `.inp` format, keys, checks, raw blocks and payloads, several passes |
| [docs/FPL_LANGUAGE_SPEC.md](docs/FPL_LANGUAGE_SPEC.md) | Quick reference for FPL, the in-tree Fortran preprocessor language |
| [ATOMCI2 technical note](https://doi.org/10.5281/zenodo.24223209) | **Technical note** (PDF, on Zenodo): the formulation of the C tools (angular algebra, Breit–Pauli terms, properties) and their validation, and the effective Hamiltonians of `heff/` |
| [docs/C_SPIN_MANUAL.md](docs/C_SPIN_MANUAL.md) | **Input manual of the C tools** `acatsp` and `acfs`: options, input files, reading the output, the examples |
| [docs/C_PERFORMANCE.md](docs/C_PERFORMANCE.md) | Long CSF lists with the C tools: times and memory measured, the settings that matter, what is not done yet |
| [CHANGELOG.md](CHANGELOG.md) | The changes of each release |
| [c/README.md](c/README.md) | The C modules and tools |
| [c/examples/README.md](c/examples/README.md) | The examples of the C tools |
| [c/tests/README.md](c/tests/README.md) | The checks and the comparisons with other programs and the literature |
| [hexpr-atomci/README.md](hexpr-atomci/README.md) | The library hexpr-atomci: build, API, example |
| [heff/README.md](heff/README.md) | The effective-Hamiltonian package: requirements, examples, scripts, tests |

The original 1994 program manual (installation, NAMELIST reference, file
formats) is kept under `docs/` as the `*.002` / `*.doc` / `atomci.tex` files.
