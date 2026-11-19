# Changes

## 1.1.0

The Fortran ATOMCI and `heff/` are as in 1.0.0. All changes are in the C
tools (`c/`) and their angular engine (`hexpr-atomci/`): long CSF lists, files
for other programs, and one correction.

### Corrected

- **Davidson could miss a level** in a J block that the Hamiltonian does not
  couple into one piece: a block without spin-dependent terms (`noso`), where
  the total (S, L) are separate, and a list with both parities. The iteration
  starts from as many vectors as levels are wanted and never leaves what the
  Hamiltonian couples to them. The case found: Fe²⁺ 3d⁶, 12436 CSFs, `noso`,
  12 levels per J; the twelfth level of J = 4 came out 6.6 × 10⁻³ hartree too
  high (a ³F level in place of the ¹G below it). 1.0.0 solves J blocks of
  dimension ≥ 500 by the same iteration, so its results for such blocks
  should be repeated. Blocks with a spin-orbit term and one parity are not
  affected by this cause.
  Now the parts that the Hamiltonian does not couple are found from the
  elements of the block, each is solved alone, and the lowest levels of all
  are returned. In addition every Davidson solution is checked against a
  small matrix that proves a miss if its n-th eigenvalue lies below the n-th
  level returned ([manual, 4.9](docs/C_SPIN_MANUAL.md)).

### Long CSF lists

- From 10000 CSFs the reduced matrices and the J blocks are kept sparse.
  Lists of 10⁵ CSFs run on a desktop computer; the times and the memory
  measured are in [docs/C_PERFORMANCE.md](docs/C_PERFORMANCE.md).
- The Coulomb Hamiltonian, H_SO2, H_SS, H_OO and the two-electron parts of
  g_J are computed for all the CSF pairs of two configurations together. The
  angular engine evaluates all the terms of an expression in one descent and
  keeps the shell factors.
- Davidson: a larger subspace with a thick restart, the mean diagonal taken
  out of H, the residual tolerance as an option (`tol=`).
- The g factors of all the J blocks are computed in one pass after the last
  block; the radial integrals of acfs are kept.
- `AC_LSJ_DISK=<directory>` keeps the reduced matrices of the spin-orbit and
  spin-spin terms in a file mapped into memory.
- `acatsp`: only the occupied orbitals are slots of a CSF (`allslots` gives
  the earlier behaviour; the results are the same to rounding). `lagfock`
  diagonalizes the orbitals of `lag=` on a screened operator, which shortens
  the Davidson iteration on long lists.
- `acfs` takes any number of configurations.

### Files for other programs (acfs)

- `info`, `dump=PREFIX`: the CSF list, the eigenvectors, energies and
  residuals of every block as NumPy files, and `meta.json` with the options,
  the conventions and the SHA-256 of the CSF list.
- `ls`: the LS levels of the Coulomb Hamiltonian, one block per (2S+1, L);
  `nlevSL=`, `nlevJ=`.
- `proj`: every other term (H_OO, mass velocity + Darwin, contact, H_SO1,
  H_SO2, H_SS) between the LS levels, each in files of its own, computed in
  a second pass that stores no matrix. `dumpmat`: every term as a sparse
  matrix (short lists).
- `nog`: the J levels without the Zeeman factors.

### Other

- `AC_TIME` reports the phases in more detail, with memory.
- New environment variables: `AC_LSJ_DISK`, `AC_FC_LIMIT_MB`, `AC_G2_PASS`,
  `AC_G_BY_LEVEL`, `AC_BP_BLOCK`, `AC_BP_PAIRWISE`, `AC_DAV_NOCHECK`
  ([manual, 4.8](docs/C_SPIN_MANUAL.md)). None changes a result.
- New checks: `c/tests/s2m4/dump.sh`, `c/tests/s2m4/terms.sh`,
  `c/tests/s2m11/lagfock.sh`.

### Documents

- The [technical note](https://doi.org/10.5281/zenodo.24223209) is revised
  with this release (version 1.1 of the note): the Davidson solver, the
  times of long lists, the options added here and the files for other
  programs. The formulation is unchanged. Version 1.0 of the note describes
  1.0.0.

### Compatibility

- The levels and properties of the examples are those of 1.0.0 to rounding.
- With `proj` or `dumpmat` the mass velocity + Darwin and the contact terms
  are computed as matrices of their own; the J levels then agree with a run
  without these options to 10⁻¹² hartree, not digit for digit.
- The g factors are printed after the last J block (with `nog` each block
  is printed when it is solved).

## 1.0.0

First release of ATOMCI2 (2026-11-15).
