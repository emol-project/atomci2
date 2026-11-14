# Cr³⁺ 3d³, a small MRCI space with virtual s, p, d, f, g (`k1f1h1`)

## Space

The RAS rule of the production spaces (`phase15s.ras_cfgs_x`: excitation level <= 2 from the references
3s² 3p⁶ 3d³ and 3s 3p⁶ 3d⁴, at most 2 electrons in the virtuals) on the RAS k = 1 virtual set of `phase15t._virt(f=1)`,
the most compact virtual of each l: 16s 15p 14d 11f 11g, with at most one 3s3p hole (`maxh = 1`; the production
spaces use 2 or 3). 35 configurations. All-electron decks, plain Sapporo-QZP-2012, own orbitals `crnd3`, the
iterative H_eff solver of the production analyses (`HEFF_SOLVER=iter`, sparse above 500 CSFs).

Unlike n3h1, this space has virtual s, p, d, f and g orbitals, so the integral and CSF paths of virtual shells
are exercised.

CSFs per term block (35 configurations; 4573 CSFs in all):

| 2P | 4P | 2D | 2F | 4F | 2G | 2H |
|---|---|---|---|---|---|---|
| 509 | 399 | 746 | 829 | 629 | 800 | 661 |

## Relation to the production spaces

None directly. It is the RAS k = 1 space family of the d³ chain (`k1f`, `k1fh` of Phases 15t/17) reduced to one
virtual f and one 3s3p hole, so that it runs in about 2 minutes.

## Run

```sh
make && make install                                   # at the repository top (the ATOMCI of this repository)
python3 heff/scripts/decks/run_example.py cr3_mrci --check   # or: make heff-examples (all three)
```

The work tree is `heff/work/examples/cr3_mrci` (or `$HEFF_WORK`; it must be new). `--check` compares with
`expected.json` and `expected_heff.npz`. The example reads only `inputs/` and the code of `heff/`, never `HEFF_DATA`.

Runtime on the development machine (4 cores, Linux, gfortran 14.2, Python 3.9.2, numpy 2.0.2): ~2 min (orbitals 90 s; ATOMCI 11 s, 3 runs at a time).

## Expected values

T1..T4 in mEh (Uylings O2', O2'', Tr O² = 1 in d³), B, C, α, β in cm⁻¹ (exact H_eff, complete projection):

| quantity | expected |
|---|---|
| T1 | 14.787561969 |
| T2 | -0.185846286 |
| T3 | -0.291393515 |
| T4 | 0.009234205 |
| B | 1068.269888 |
| C | 4150.775078 |
| alpha | 2.986660 |
| beta | 1113.537111 |

Tolerances of `--check`: 1e-6 mEh (T), 1e-4 cm⁻¹ (B, C, α, β, f), 1e-9 Eh per H_eff element (`expected_heff.npz`: the
exact, Sylvester and closure H_eff of every term block), and the CSF counts exactly. They leave room for other
compilers and BLAS libraries; on the development machine the agreement is exact.

## Against the development code

The development version of this code with the ATOMCI binary used for the production spaces
(sha256 6d6ed833…), and the packaged code with the ATOMCI built from this repository (sha256 bce7a03d…), on the same
space: max |dH_eff| = 0 (all blocks, exact / Sylvester / closure), the exported H of every block bit-identical, and
every value above identical to all digits (difference 0).
