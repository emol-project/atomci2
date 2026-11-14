# hexpr-atomci

A C library that gives the Hamiltonian matrix elements between atomic configuration state
functions (CSFs) as expressions: angular coefficients times symbolic integrals,

    <F|H|G> = sum coef h(a,b) + sum coef R^k(ab;cd)

It is the angular engine of ATOMCI2, a C rewrite of the angular part of the ATOMCI program,
taken out as a library. It uses the method of ATOMCI (F. Sasaki 1974 and the METECC-94
chapter; `LEGENDS.txt`, Section 4): LS-coupled CSFs in O(3) x SU(2) x SU(2), with fractional
parentage from a generated table. The formulation is in the ATOMCI2 technical note, Secs. 3
and 4.1.

The library is a self-contained folder of the ATOMCI2 repository and needs nothing outside it;
`make check-standalone` checks this on a copy elsewhere. To take it out, `make dist VERSION=x.y`
makes a tarball of this folder, and `git subtree split --prefix=hexpr-atomci` from the top of
the repository makes a branch with the history of this folder alone.

## Build

    make
    make check PYTHON=/path/to/python    # the Python needs numpy

This builds `build/libhexpr_atomci.a` and `build/libhexpr_atomci.so`. Use the static library
when speed matters: through the shared library, matrix elements took about 35% longer, because
the 6j, 9j and tree caches are thread-local.

## Example

```c
#include "hexpr_atomci.h"

ac_cfptab t;
char err[256];
ac_cfp_generate(&t, err, sizeof err);

/* orbitals 1s, 2s, 2p; the configurations 1s2 2s2 2p and 1s2 2s 2p2 */
int lorb[] = {0, 0, 1};
int occ[] = {2, 2, 1,
             2, 1, 2};
ac_orb_csf *csf;
int n;
ac_orb_csfs(&t, 3, lorb, 2, occ, &csf, &n, err, sizeof err);

ac_expr l;
ac_expr_init(&l);
ac_pair_expr(&t, &csf[0], &csf[1], 1e-13, &l);
for (int i = 0; i < l.len; i++) {
    /* k < 0: coef h(a,b); k >= 0: coef R^k(ab;cd) */
    ac_expr_term *x = &l.v[i];
    printf("%d %d %d %d %d %.15g\n", x->k, x->a, x->b, x->c, x->d, x->coef);
}
```

`ac_terms.h` describes this interface. For operators other than the Hamiltonian, `ac_optree.h`
and `ac_factor.h` give the operator trees and the general matrix-element engine.

## Using it with radial integrals from elsewhere

The library needs no integrals to give an expression, and any program that has the integrals
`h(a,b)` and `R^k(ab;cd)` of real radial orbitals can put them in.

- `examples/d2_terms.c` prints the terms of p2 and d2 as Slater integrals and checks them
  against the closed forms. No radial integrals are involved.
- `examples/pyscf_atom/` takes the integrals from PySCF in a Gaussian basis and compares every
  level with the FCI of PySCF (`make check-pyscf`). It is an example, not a PySCF plugin: the
  CI here is not an `fcisolver`, and it gives no density matrices for CASSCF.

## Checks

| Target | Compares |
|---|---|
| `check-example` | `tests/example.c`, through the public header, against coefficients known by elementary rules |
| `check-d2` | the terms of p2 and d2 as Slater integrals against their closed forms |
| `check-terms`, `check-terms-so` | expressions against reference values from ATOMCI2's `ac_ci_hij` (static and shared library) |
| `check-dlopen` | the shared library loaded at run time, in several threads |
| `check-cfp` | the generated CFP table against the SHA-256 of the Fortran ATOMCI's table (`tests/ref/CFPTBL.sha256`), byte for byte |
| `check-wigner` | 6j and 9j against exact rational arithmetic |
| `check-pyref` | the determinant reference `tests/python/pyref.py` against closed forms |
| `check-engine` | the engine against that determinant reference, which uses no CFPs and no code of this library |

## Origin

The source files come from the C tools of ATOMCI2 (`c/`). `ac_terms.c` is new and follows
`ac_ci_hij` of those tools. Parts of the CFP code are line-by-line ports of the Fortran
ATOMCI; `ac_cfpgen.c` also contains a least-squares routine that its comments trace to the IBM
Scientific Subroutine Package.

## License

Non-commercial research and educational use, under the terms of `LEGENDS.txt`. This library is a
derivative of ATOMCI, so those terms apply to all of it, and this is not an MIT-licensed library.
Publications must carry the attribution in Section 4 of `LEGENDS.txt`.
