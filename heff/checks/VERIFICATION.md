# Verification of the packaged heff/

`heff/` was taken from the development version of the code, which produced the results of the production spaces.
The checks below compare the packaged code, run with the ATOMCI built from this repository, with the development
code and the ATOMCI binary used for the production spaces (sha256 6d6ed833…). They were run on Linux with gfortran
14.2.0 and Python 3.9.2 (numpy 2.0.2, scipy 1.13.1), and repeated on Debian 13 with Python 3.13 (numpy 2.5.4,
scipy 1.18.1, sympy 1.14).

## Fortran ATOMCI

The Fortran sources differ from those used for the production spaces only in comments, build files and two
corrections of ATOMCI2: blank title words (formerly NUL bytes in the output) and a stop on `&MC CSFS` shells out of
the coupling sequence (formerly a silent wrong MCSCF). Neither affects the Hamiltonian matrix of a correct input:
the two binaries give bit-identical H (below).

## Level 1: the result tables

With the H_eff data of the production spaces as `HEFF_DATA`, `scripts/tables/rebuild_level1.py --check` recomputes
every intermediate JSON from the H_eff dumps (none is copied) and writes six tables. All six are byte-identical with
the reference copies in [`reference/`](reference). The intermediate JSON agree with those of the development code to
3.1e-6 in their stored units, not bit for bit (BLAS reduction order; the operator matrices are regenerated and agree
to 1e-16), and the iteration counters of two least-squares fits differ while their parameters agree to 1e-6.

## Examples from scratch

The three examples of [`../examples/`](../examples) (orbitals, decks, ATOMCI, H, H_eff, projection) were run with
both codes and both binaries. For all three, the largest difference of an H_eff element (exact, Sylvester and
closure; 30, 30 and 104 elements) is zero, the exported H of every term block is bit-identical, and T1..T4, B, C, α,
β, and f1..f12 for d⁶, agree to all printed digits. `cr3_mrci` and `co3_mrci` have virtual s, p, d, f and g orbitals,
so they also exercise the integral and CSF paths that only virtual shells reach.

## Fresh clone

From a fresh clone, `make` and `make install`, then with no `HEFF_DATA`, `HEFF_WORK` or `ATOMCI_*` set:
`make heff-test` gives 44 passed and 5 skipped (4 tests that need local runs of the development code, and the
Level 1 regression without data) in about 15 s, and `make heff-examples` passes all three examples in 6–8 min.
`heff/data` is never created.

## Fixes made while packaging

The d⁶ example found four places on the Level 2 / 3 d⁶ path that built paths from `__file__` instead of
`HEFF_WORK` (`phase17.own_hf`, `phase17.OUT17`, the two `shadow21` modules) and one script that wrote next to
itself (`counts.py`). They were corrected before the checks above.
