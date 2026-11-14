# Reference data

The files in this directory are published data of other authors. They are not part of ATOMCI and are not covered
by `LEGENDS.txt` (Section 5). Work that uses them should cite their sources.

| directory | contents | source |
|---|---|---|
| `JL1986/` | Tables 7 (d⁴), 8 (d⁵) and 9 (normalization factors) transcribed as CSV; `check_jl.py` checks the transcription | B. R. Judd and R. C. Leavitt, J. Phys. B **19**, 485 (1986) |
| `levels/` | energy levels of V III, Cr IV, Mn V, Fe III, Fe VI and Co IV, as exported from the database | A. Kramida, Yu. Ralchenko, J. Reader, and NIST ASD Team, NIST Atomic Spectra Database, https://physics.nist.gov/asd, doi:10.18434/T4W30F |

The code reads `JL1986/` to build the four-body operators f1..f12 (`atomci_heff/dshell.py`) and `levels/` for the
fits of the measured levels.
