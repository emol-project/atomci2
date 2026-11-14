# Co³⁺ 3d⁶, a small MRCI space with virtual s, p, d, f, g, and the four-body operators (`k1f1h1`)

## Space

The same rule and virtual set as `cr3_mrci` for Co³⁺ 3d⁶ (references 3s² 3p⁶ 3d⁶ and 3s 3p⁶ 3d⁷;
16s 15p 14d 11f 11g; at most one 3s3p hole): 35 configurations, 11 term blocks. All-electron decks from the Z-generic
writer (`phase18_write_deck`), plain Sapporo-QZP-2012 (`basis/co.without-diffuse.bas`), own config-average HF orbitals
`cond6` (`phase18.py coorb`), the d⁶ HF P block from the space's own PHP (`phase17.own_hf`), iterative H_eff
(`phase15j.analyze2`), and the projection of the exact H_eff onto A, B, C, α, β, T1..T4 and the twelve pure four-body
operators f1..f12 (`phase18_co.three6`, as for the production Co³⁺ results).

f1..f12 are nonzero because the space has a virtual d (14d) and the 3s → 3d channel. Code paths of d⁶ not exercised
here: the correction-chain items (2p, 2s holes, i and k functions, DK3, the fam space with its diffuse
d), the dense d⁶ solver, and the NIST four-body refits (`phase18_co.py refit2`). Those run at Level 1 from the dumps.

CSFs per term block (35 configurations; 23934 CSFs in all):

| 1S | 3P | 1D | 3D | 5D | 1F | 3F | 1G | 3G | 3H | 1I |
|---|---|---|---|---|---|---|---|---|---|---|
| 441 | 1993 | 1735 | 2933 | 1522 | 1962 | 3411 | 2028 | 3411 | 3031 | 1467 |

## Relation to the production spaces

None directly. It is the Co³⁺ RAS k = 1 family (`cond:k1fh` of Phase 18) reduced to one virtual f and one
3s3p hole.

## Run

```sh
make && make install                                   # at the repository top (the ATOMCI of this repository)
python3 heff/scripts/decks/run_example.py co3_mrci --check   # or: make heff-examples (all three)
```

The work tree is `heff/work/examples/co3_mrci` (or `$HEFF_WORK`; it must be new). `--check` compares with
`expected.json` and `expected_heff.npz`. The example reads only `inputs/` and the code of `heff/`, never `HEFF_DATA`.

Runtime on the development machine (4 cores, Linux, gfortran 14.2, Python 3.9.2, numpy 2.0.2): ~4.5 min (orbitals 45 s; ATOMCI 165 s, 3 runs at a time; analysis and projection 50 s).

## Expected values

T1..T4 in mEh (Uylings O2', O2'', Tr O² = 1 in d³), B, C, α, β in cm⁻¹ (exact H_eff, complete projection):

| quantity | expected |
|---|---|
| T1 | 15.042499370 |
| T2 | -0.345598324 |
| T3 | -0.210810439 |
| T4 | 0.237679408 |
| B | 1270.213328 |
| C | 4798.713426 |
| alpha | 1.832613 |
| beta | 1160.603003 |

Four-body operators (cm⁻¹; as stored, and normalized as f·√A(f), Tr f² = 1 in d⁴):

| f | stored | normalized |
|---|---|---|
| f1 | +0.144962 | +15.579 |
| f2 | +0.112415 | +30.909 |
| f3 | +0.017289 | +17.029 |
| f4 | -0.013801 | -7.099 |
| f5 | -0.661099 | -99.561 |
| f6 | -0.048546 | -9.748 |
| f7 | +0.025802 | +51.549 |
| f8 | +0.239629 | +20.544 |
| f9 | +0.019292 | +13.314 |
| f10 | +0.030934 | +4.630 |
| f11 | +0.004198 | +38.434 |
| f12 | +0.011835 | +7.005 |

Tolerances of `--check`: 1e-6 mEh (T), 1e-4 cm⁻¹ (B, C, α, β, f), 1e-9 Eh per H_eff element (`expected_heff.npz`: the
exact, Sylvester and closure H_eff of every term block), and the CSF counts exactly. They leave room for other
compilers and BLAS libraries; on the development machine the agreement is exact.

## Against the development code

The development version of this code with the ATOMCI binary used for the production spaces
(sha256 6d6ed833…), and the packaged code with the ATOMCI built from this repository (sha256 bce7a03d…), on the same
space: max |dH_eff| = 0 (all blocks, exact / Sylvester / closure), the exported H of every block bit-identical, and
every value above identical to all digits (difference 0).
