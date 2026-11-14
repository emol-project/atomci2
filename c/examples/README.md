# Examples of the C tools (spin-dependent calculations)

Inputs, the commands that were run, and their outputs, for every operator
and property of `acatsp` and `acfs`. The input manual is
[`docs/C_SPIN_MANUAL.md`](../../docs/C_SPIN_MANUAL.md). Its §6 says what
each example shows.

Each directory holds:

- the inputs: ATSP orbitals `*.w`, configuration lists `cfg*`, or the deck
  for acfs;
- `commands`, one line per run: `<output> <tool> <arguments>`. Run the
  arguments in that directory with stdout to the output, e.g.
  `../../build/acatsp - O.w cfg 9 > so1.out`;
- the outputs `*.out`;
- for the acfs examples, `prep.sh`: the Fortran ATOMCI run whose files acfs
  reads.

| Directory | System | Tool |
|---|---|---|
| `o2p4` | O 2p⁴ | acatsp |
| `li` | Li 1s²2s, 1s²2p | acatsp (with `bio`) |
| `he2p` | He 1s2p | acatsp |
| `hfsrel` | Li 1s²2s (complete CI) | acatsp `hfsrel` |
| `sc` | Sc 3d4s² | acatsp |
| `ti` | Ti 3d²4s² (+ 3d³4s, 3d⁴) | acatsp |
| `cu` | Cu 3d¹⁰4s, 3d⁹4s² | acatsp |
| `f2p_acfs` | F 2p⁵ (Fortran MCSCF orbitals, GTO) | acfs |
| `tm_acfs` | Sc, Ti, Cu (Fortran MCSCF orbitals, cc-pVTZ) | acfs |

## Running and checking

```bash
cd c && make                      # the tools in c/build
export ATOMCI_HOME=...            # only for f2p_acfs, tm_acfs (make install at the top)
examples/run_examples.sh          # all, or: examples/run_examples.sh o2p4 sc
```

`run_examples.sh` runs each example in a scratch copy and compares every
number with the shipped output (relative 1e-7, absolute 1e-9, magnitudes:
the sign of a matrix element between two levels follows eigenvector
phases). `REF=1` overwrites the shipped outputs instead. The whole set takes
about 20 seconds.

The acatsp examples need neither ATSP nor the Fortran ATOMCI: the CFP table
is generated in C (`-`) and the orbitals are shipped. `make_orbitals.sh`
records how the `.w` files were made (ATSP's `hf`; it reruns it when ATSP is
installed). `hfsrel` uses the files of `c/tests/s2m11`.

The shipped outputs were made on Debian 13 (gcc and gfortran 14.2.0, the
system LAPACK and BLAS).
