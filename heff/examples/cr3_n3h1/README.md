# Cr³⁺ 3d³, the n = 3 complex with at most one 3s3p hole (`n3h1`)

## Space

Space `n3h1` of Phase 15u (step 2): every even-parity distribution of the 11 n = 3 electrons over
3s, 3p, 3d with at most one hole in 3s3p and no virtual orbitals: 3s² 3p⁶ 3d³ (P) and 3s 3p⁶ 3d⁴ (2 configurations).
All-electron decks (`allcore`), plain Sapporo-QZP-2012 (`basis/cr.without-diffuse.bas`), own config-average HF
orbitals of Cr³⁺ 3d³ (`crnd3`, made by the example), exact H_eff by des Cloizeaux (dense solver).

CSFs per term block (n3h1 has 2 configurations; 24 CSFs in all):

| 2P | 4P | 2D | 2F | 4F | 2G | 2H |
|---|---|---|---|---|---|---|
| 3 | 3 | 5 | 4 | 3 | 4 | 2 |

## Relation to the production spaces

One point of the 3s3p hole-count series (n3h0 ... n3h7) of Phase 15u, which fixed the pure n = 3 item of the d³
correction chain. It is not itself an entry of a result table.

## Run

```sh
make && make install                                   # at the repository top (the ATOMCI of this repository)
python3 heff/scripts/decks/run_example.py cr3_n3h1 --check   # or: make heff-examples (all three)
```

The work tree is `heff/work/examples/cr3_n3h1` (or `$HEFF_WORK`; it must be new). `--check` compares with
`expected.json` and `expected_heff.npz`. The example reads only `inputs/` and the code of `heff/`, never `HEFF_DATA`.

Runtime on the development machine (4 cores, Linux, gfortran 14.2, Python 3.9.2, numpy 2.0.2): ~100 s (orbitals 90 s; ATOMCI 1 s).

## Expected values

T1..T4 in mEh (Uylings O2', O2'', Tr O² = 1 in d³), B, C, α, β in cm⁻¹ (exact H_eff, complete projection):

| quantity | expected |
|---|---|
| T1 | 17.083001278 |
| T2 | 0.097603952 |
| T3 | -0.210350673 |
| T4 | 0.169697629 |
| B | 1125.753363 |
| C | 4119.525149 |
| alpha | 68.245228 |
| beta | 718.753369 |

Tolerances of `--check`: 1e-6 mEh (T), 1e-4 cm⁻¹ (B, C, α, β, f), 1e-9 Eh per H_eff element (`expected_heff.npz`: the
exact, Sylvester and closure H_eff of every term block), and the CSF counts exactly. They leave room for other
compilers and BLAS libraries; on the development machine the agreement is exact.

## Against the development code

The development version of this code with the ATOMCI binary used for the production spaces
(sha256 6d6ed833…), and the packaged code with the ATOMCI built from this repository (sha256 bce7a03d…), on the same
space: max |dH_eff| = 0 (all blocks, exact / Sylvester / closure), the exported H of every block bit-identical, and
every value above identical to all digits (difference 0).
