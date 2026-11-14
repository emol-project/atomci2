# heff: effective Hamiltonians of 3d^N ions on top of ATOMCI

This directory holds a Python package and scripts that build the spin-free effective Hamiltonian H_eff of the 3d^N
configuration of V²⁺, Cr³⁺, Mn⁴⁺, Fe⁵⁺ (d³) and Fe²⁺, Co³⁺ (d⁶) from ATOMCI CI matrices (stop-after-H runs),
project H_eff onto the operator basis of the parametric theory (Racah B, C, α, β, the three-body T1..T4, the
four-body f1..f12), apply corrections computed in smaller spaces, and compare the results with fits of the NIST
levels and with parameter sets from the literature. The method and the checks are described in Sec. 10 of the
ATOMCI2 technical note.

ATOMCI is not modified. The code runs the ATOMCI binary built from this repository and reads the binary scratch
files ATOMCI writes (CI matrix units 38/39, the CSF list, the orbital file `be.NO`).

## Requirements

- Python 3.9 or later, with numpy, scipy and sympy, and pytest for the tests. The versions used are in
  [`requirements.txt`](requirements.txt).
- For the examples and Level 3: ATOMCI built and installed from this repository (`make && make install` at the
  top), and the `atinputs` package in `scripts/` of this repository.

## Levels of reproduction

| level | starts from | needs | time |
|---|---|---|---|
| examples | the inputs in [`examples/`](examples) (repository only) | Python, ATOMCI of this repository | a few minutes each |
| 1 | the H_eff dumps and per-space result JSON of the production spaces (not in the repository) | Python | ~7 min (the table scripts alone: seconds) |
| 2 | the stored CI matrices H (`results/<run>.npz`) | the matrices | not distributed (hundreds of GB) |
| 3 | nothing: orbitals → decks → ATOMCI → H → H_eff | ATOMCI | hours to days per space (largest spaces: days) |

**Examples** (repository only): three small spaces run from scratch, through the same functions as the
production runs: own HF orbitals → decks (the production configuration rule) → ATOMCI stop-after-H → H_eff →
projection. See [`examples/README.md`](examples/README.md).

```sh
make && make install                          # at the repository top: the ATOMCI the examples use
python3 scripts/decks/run_example.py cr3_mrci --check     # or: make heff-examples (all three)
```

**Level 1** reruns every assembly step that reads H_eff (the d³ and d⁶ correction chains, the projections, the NIST
fits, the four-body fits, the comparison with the literature), rebuilds all intermediate JSON from the dumps, writes
the tables, and with `--check` compares them byte for byte with the reference copies in
[`checks/reference/`](checks/reference):

```sh
export HEFF_DATA=/path/to/heff-data          # the H_eff data of the production spaces (read only)
python3 scripts/tables/rebuild_level1.py --check
```

The inputs are listed in [`checks/manifest_level1.txt`](checks/manifest_level1.txt) (134 files, 3.7 MB):
the H_eff dumps (`results/phase1{7,8}/heff_dump/*.npz`), the per-space results (`results/phase15/phase15[jk]_*.json`),
the second-order and Q1_rule results (`results/phase17/{a,c,crule,cscan}_*`, `results/phase18/q1diag_*`),
`results/phase19/phase19_pos.json` (from the stored H), and two orbital files (`runs/vq_orb_{crnd3,fe5nd3}/be.NO`).
Those JSON and the Q1 results come from Level 2 and are taken as given at Level 1.

**Level 2** (H_eff of a space from its stored H) needs the matrices, which are not distributed. Its scripts are here:
`scripts/heff/phase17.py analyze`, `phase17q11_dump.py`, `phase17a.py`, `phase17c_scan.py`, `phase18_q1diag.py`, and the
shell drivers `scripts/decks/phase18_*.sh`.

**Level 3** starts from scratch, as the examples do, for the production spaces. They run through the queue drivers
`scripts/decks/phase17_queue.py` and `phase18_queue.py`. These have a memory cap (`CAP_MB`, `MEMSTOP_MB`), per-job
timeouts and a disk check, and they delete large `fort.*` files after export (unless listed in
`results/phase15/run_files_keep.txt`).

## Result tables → commands

All tables are written under `$HEFF_WORK/results/` by `scripts/tables/rebuild_level1.py`. Each can also be rerun
alone once its inputs exist (`python3 <script>`):

| table file | contents | script | main inputs |
|---|---|---|---|
| `phase18/paper_tables.md` §1 | final T1, T2: projection and (1b) representation of the corrected H_eff, NIST (1b) fit, literature values | `tables/phase18_tables.py` | `fits/phase18_fe5.py final/refit`, `fits/phase18_co.py final/fefinal`, `fits/phase17q11.py chain` |
| same, §2 | second order and Q1_rule against exact: closure, S2-full, S2-diag, Q1_rule, + pf2 | same | `phase17a_tables.json`, `crule_*`, `q1diag_*`, `a_*` |
| same, §3 | four-body fits of Fe III and Co IV: λ fits, F tests | same | `fits/phase18_co.py refit2/ferefit2` |
| same, §4 | f1..f12: corrected values and literature values | same | `fits/phase18_co.py final`, `fits/phase17q20.py all` |
| same, §5 | Fe⁵⁺: second-order classes against the literature | same | `fits/phase18_fe5.py second/tables` |
| `phase19_tables.md` | level residuals, T1/T2-only test, configurations | `tables/phase19_report.py` | `fits/phase19.py nist/fits/single` |
| `phase17a_tables.md` | second-order variants | `tables/phase17a_tables.py` | `a_*.npz`, `phase15k_*` |
| `phase17c_rule_tables.md` | Q1_rule leave-one-out | `tables/phase17c_rule_tables.py` | `crule_*.json` |
| `phase17_final_tables.md` | final estimates and error budget | `tables/phase17_final.py` | `phase17q18.json`, `phase15[jk]_*` |
| `phase21_tables.md` | additional checks | `tables/phase21_report.py` | `results/phase21/*.json` (Level 2 / 3) |

## Layout

```
atomci_heff/          the package: ATOMCI file readers, GTO integrals, H_eff solvers, d^N operator algebra, config.py
scripts/orbitals/     own config-average HF orbitals (phase14a_orb, phase15f, phase15s_nd)
scripts/decks/        deck and CSF generation, the ATOMCI runner (run_atomci.py), queue drivers, run_example.py
scripts/heff/         spaces, H_eff (exact / Sylvester / closure / iterative), partial dressing, projection (phase9..phase21)
scripts/fits/         NIST (1b) fits, correction chains, four-body fits, literature comparison; trees/ (operator matrices)
scripts/tables/       table generators, init_work.py, rebuild_level1.py
tests/                pytest; tests/fixtures/cr3_fam (33 kB, the exact H_eff of Cr³⁺ fam)
examples/             three small runs from scratch with their inputs and expected values (cr3_n3h1, cr3_mrci, co3_mrci)
checks/               manifests, reference tables, and the verification of the packaging (VERIFICATION.md)
basis/                basis sets (ATOMCI native format; see basis/README.md)
refdata/              published reference data, not covered by LEGENDS.txt (see refdata/README.md):
  levels/             NIST ASD level lists (V III, Cr IV, Mn V, Fe III, Fe VI, Co IV)
  JL1986/             Judd & Leavitt (1986) Tables 7, 8 and 9, transcribed (+ check_jl.py)
```

The scripts keep their development names (`phaseNN*.py`) because they import each other by name, and the docstrings
still refer to the development phases. `scripts/heff/shadow21/` holds the two module versions that `phase21c.py`
imports instead of the current ones.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `ATOMCI_BIN` | ATOMCI binary | `<repo>/local/bin/atomci` (`make install`) |
| `ATOMCI_CFPTBL` | CFP table | `<repo>/local/CFPTBL` (`make install`) |
| `ATOMCI_SCRIPTS` | directory holding `atinputs/` | `<repo>/scripts` |
| `HEFF_DATA` | the H_eff data of the production spaces (Level 1), read only; not used by the examples | `<repo>/heff/data` |
| `HEFF_WORK` | work tree; every script reads and writes here | `<repo>/heff/work` |

`HEFF_WORK` has the layout `results/`, `runs/<run>/`, `inputs/<run>/`.
`scripts/tables/init_work.py level1` copies the inputs from `HEFF_DATA` into it. A missing input is an
error, never a silent fallback. `python3 -m atomci_heff.config` prints the values in use. Optional:
`HEFF_DUMP=<dir>` makes the H_eff analyses dump every H_eff block, and `HEFF_SOLVER`, `ION`, `CAP_MB`, `MAXJ`,
`JOB_H` are switches documented in the scripts.

## Data of the production spaces

The Level 1 inputs (the H_eff dumps, the per-space results and two orbital files; 134 files, 3.7 MB) are not in
the repository; they are distributed separately (`[to be added]`). Their layout is that of `HEFF_WORK`, and
[`checks/manifest_level1.txt`](checks/manifest_level1.txt) lists the files. The repository itself needs no external
data: the tests and the examples run with the repository alone.

## Tests

```sh
python3 -m pytest -q tests                    # ~15 s; no external data; the Level 1 regression is skipped without HEFF_DATA
python3 -m pytest -q -m slow tests/test_examples.py        # the three examples (needs `make install`; ~10 min in all)
HEFF_DATA=/path/to/heff-data python3 -m pytest -q tests   # + test_level1_tables (~7 min)
```

From the repository top: `make heff-test`, `make heff-examples` and `make heff-tables`.

## License and citation

This directory is part of ATOMCI2 and is distributed under the same terms ([`LEGENDS.txt`](../LEGENDS.txt):
non-commercial research and educational use). `atomci_heff/integrals.py` contains a Python port of the integral
addressing of ATOMCI (`src/setiad.f`: SETIAD, INT1AD, INT2AD). It is a derivative work of ATOMCI under LEGENDS.txt,
Section 3(b). The files in `refdata/` are published data of other authors and are not covered by LEGENDS.txt.

Any publication based on results obtained with this code must include the attribution

> This work is based, in part, on results from the ATOMCI package.

and cite

1. F. Sasaki, Int. J. Quantum Chem. **8**, 605 (1974);
2. F. Sasaki, M. Sekiya, T. Noro, K. Ohtsuki, and Y. Osanai, "Non-Relativistic Configuration Interaction
   Calculations for Many-Electron Atoms: ATOMCI", pp. 163–219 in *Methods and Techniques in Computational Chemistry:
   METECC-94*, edited by E. Clementi (ETEF, Cagliari, 1993).

Please also cite the ATOMCI2 technical note when you use the effective-Hamiltonian code.
