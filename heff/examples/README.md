# Examples (repository only)

Three small calculations that run from scratch with this repository alone: own HF orbitals in Python → decks from
the production configuration rule → ATOMCI stop-after-H runs → H_eff (exact, Sylvester, closure) → projection onto
the operators of the parametric theory. They call the same functions as the production runs, through
`scripts/decks/run_example.py`. They read only their own `inputs/` and the ATOMCI of this repository
(`ATOMCI_BIN`, `ATOMCI_CFPTBL`; default `local/` after `make install`), never `HEFF_DATA`.

| example | ion | space | configurations / CSFs | virtuals | runtime | checks |
|---|---|---|---|---|---|---|
| [`cr3_n3h1`](cr3_n3h1) | Cr³⁺ 3d³ | n = 3 complex, ≤ 1 hole in 3s3p | 2 / 24 | none | ~100 s | T1..T4, B, C, α, β, H_eff |
| [`cr3_mrci`](cr3_mrci) | Cr³⁺ 3d³ | RAS k = 1 (16s 15p 14d 11f 11g), ≤ 1 hole in 3s3p | 35 / 4 573 | s p d f g | ~2 min | T1..T4, B, C, α, β, H_eff |
| [`co3_mrci`](co3_mrci) | Co³⁺ 3d⁶ | the same rule and virtuals | 35 / 23 934 | s p d f g | ~4.5 min | T1..T4, B, C, α, β, f1..f12, H_eff |

```sh
make && make install                     # at the repository top
make heff-examples                       # all three, compared with expected.json / expected_heff.npz (~8 min)
python3 heff/scripts/decks/run_example.py co3_mrci --check     # one of them
```

Each folder holds `example.json` (the space: ion, rule, solver), `inputs/` (deck templates, orbital-file headers
and small JSON in the layout of the work tree, copied into it), `expected.json`, `expected_heff.npz` (the H_eff
blocks, Eh) and a README with the space, the CSF counts, the expected values and the comparison with the development
code. The examples are smaller than any production space; their READMEs say which production space each one
reduces.
