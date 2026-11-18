# Long CSF lists with the C tools: where 1.1.0 stands

This note records what was measured with `acatsp` and `acfs` on long CSF
lists while 1.1.0 was developed, which settings matter, and what is not done.
The options and environment variables themselves are in the
[input manual](C_SPIN_MANUAL.md) (4.8, 4.9).

The numbers are single runs of development builds between 1.0.0 and 1.1.0,
not a benchmark: they show the order of the time and memory to expect.

## 1. Measured

Machine A: 4 cores (8 hardware threads), 32 GB of memory, a hard disk;
`OMP_NUM_THREADS=4`. Machine B: 2 cores (4 hardware threads);
`OMP_NUM_THREADS=4`.

### J levels (matrix of every J block, Davidson)

| Case | CSFs | Terms | Wall time | Peak memory |
|---|---|---|---|---|
| Ti, `acatsp`, `lagfock`, 9 levels per J | 388009 | Coulomb + H_SO1 | 1.6 h | 12.5 GB |
| Ti, `acatsp`, `AC_LSJ_DISK`, 9 levels per J | 171621 | + `so2 ss` | 1.4 h | 28 GB with the pages of the mapped file |
| Fe²⁺ 3d⁶, `acfs`, `noso`, 12 levels per J, g factors | 12436 | Coulomb | 11 s (machine B: 50 s) | |

In the 171621 CSF run the time divides as: the reduced matrices 76 %, the
Davidson iterations 12 %, filling the J blocks 6 %, g factors 6 %. A J block
takes 12 to 46 Davidson iterations for 9 levels, at 1652 to 149043 CSFs
alike. The reduced matrices are the cost, and in them H_SO2.

### LS levels and the terms between them (`acfs ... ls dump= proj`)

Fe²⁺ 3d⁶, 12 levels per (S, L) block, machine A.

| CSFs | Terms | Wall time | Peak memory |
|---|---|---|---|
| 7482 | H_SO1 | 4 s | 0.09 GB |
| 7482 | `so2 ss` | 73 s | 0.93 GB |
| 7482 | `so2 ss oo mvd contact` | 99 s | 0.96 GB |
| 12436 | `so2 ss` | 116 s | 1.06 GB |
| 12436 | `so2 ss oo mvd contact` | 154 s | 1.08 GB |
| 102001 | `so2 ss` | 1.2 h | 2.2 GB |

The list of 102001 CSFs (55 configurations: one and two electrons of 3s, 3p
and 3d moved into one s, p, d, f and g shell) was made for this
measurement. Its first pass (the LS levels) took 55 s and the second
(the terms between them) 4256 s, of which H_SO2 two thirds and H_SS one
third of the thread time.

This path stores no matrix of the terms, so its memory stays small; the time
grows with the number of CSF pairs that the terms connect (3.9 × 10⁹ of
5.2 × 10⁹ at 102001 CSFs).

## 2. Settings that matter

- **Threads.** `OMP_NUM_THREADS`. The reduced matrices and the second pass
  of `proj` are computed in parallel over pairs of configurations.
- **`nog`** when only the levels are wanted: on a long list the Zeeman
  factors can take most of the time.
- **`ls` + `proj`** instead of `dumpmat` on long lists: no matrix is stored.
- **`AC_LSJ_DISK=<directory>`** with `so2` or `ss` when the reduced matrices
  and the largest J block do not fit in memory together. The system then
  drops pages of the file instead of swapping. The J blocks themselves stay
  in memory.
- **`AC_FC_LIMIT_MB`** (default 128, per thread): the kept shell factors of
  the two-electron terms. They are dropped and built again whenever they
  pass the limit, which costs time on lists with an open d shell and many
  shell states. Fe²⁺, 7482 CSFs, `so2 ss ls proj`, 8 threads:

  | `AC_FC_LIMIT_MB` | 64 | 128 | 256 | 512 |
  |---|---|---|---|---|
  | Wall time | 118 s | 50 s | 46 s | 48 s |
  | Peak memory | 1.44 GB | 1.73 GB | 2.58 GB | 2.97 GB |

  A Ti list of 171621 CSFs (four valence electrons, many orbitals) did not
  change with the limit.
- **`AC_PROP_CACHE_MB`** (default 2048): the n × n tables of `zoff` and
  `trans`. The g factors no longer need them.
- **`lagfock`** (acatsp, with `lag=`): several times fewer Davidson steps
  when the virtual orbitals are the analytic ones of `lag=`.
- **`tol=`** saves iterations in trial runs. Keep the default (10⁻¹¹) for
  results: with a large tolerance a level can be missed between parts of a
  block that couple weakly (manual, 4.9).

## 3. Reproducibility

The levels, the eigenvectors and the printed output are the same from run to
run on the cases compared. The files of `proj`
differ in the last digits from run to run (10⁻¹⁷ to 10⁻¹⁵): the threads add
their contributions in the order in which they finish.

## 4. Not done

- **The J blocks are held in memory.** That sets the largest list: with
  `so2 ss` a list of 3 × 10⁵ CSFs is estimated at 26 to 28 GB (extrapolated
  from 171621 CSFs, not run). Keeping the J blocks in a file and reading it
  once per Davidson iteration would lift this; it is not written.
- **One machine.** The parallelism is OpenMP threads. Distributing the rows
  of the matrices over processes is not written.
- **`zoff`, `hfs`, `trans` on long lists.** They still use n × n tables or
  work out radial integrals on every call. The results are right; the time
  on lists above 10⁴ CSFs was not measured and is expected to be long.
- **H_SO2.** It takes most of the time of the reduced matrices. Evaluating
  its expressions together in the descent of the angular engine is estimated
  to save up to a fifth of that; it is not written.
