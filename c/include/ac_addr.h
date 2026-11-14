/*
 * ac_addr.h -- integral addresses of the energy-expression tape: a port of
 * SETIAD/SETKAD/INT1AD/INT2AD (src/setiad.f).
 *
 * Orbitals are numbered by their position in the coupling sequence
 * (1..nwt). Each has an orbital l and an index within its l (NWSQ). The
 * one-electron integrals h(a,b) come first, triangular per l. They are
 * followed by the Slater integrals R^k(a,b;c,d), in blocks per k and per
 * (l1,l2),(l3,l4) pair class. The tape stores only these addresses, so the C
 * engine must produce them exactly as the Fortran does.
 */
#ifndef AC_ADDR_H
#define AC_ADDR_H

#include "ac_cfp.h"

#include <stdint.h>

#define AC_ADDR_MAXORB 150 /* Fortran KBASIS */
#define AC_ADDR_MAXL 20
#define AC_ADDR_LISL 2048  /* Fortran LISL (set in setcom.fpl) */

typedef struct {
    int nwt, lmax;               /* lmax = largest l+1                 */
    int issh[AC_ADDR_MAXL + 1];  /* [1..lmax] orbitals per l           */
    int nwlf[AC_ADDR_MAXORB + 1]; /* [1..nwt] l+1 of each orbital      */
    int nwsq[AC_ADDR_MAXORB + 1]; /* [1..nwt] index within its l       */
    int itad[AC_ADDR_MAXL + 1], ioead[AC_ADDR_MAXL + 1];
    int igcd[AC_ADDR_LISL + 1], iscd[AC_ADDR_LISL + 1];
    int lislu;  /* largest pair-class address used */
    int igtad;  /* one past the last address: the number of integrals */
} ac_addr;

/* SETIAD (kad = 0): l from the empty/closed codes nwth[nwt] via ILFO,
   two-electron addresses after the one-electron block.
   SETKAD (kad = 1): l+1 given directly in nwlf[nwt], two-electron
   addresses from 0. nwsq[nwt] is the index within l; if nwsq is NULL or
   nwsq[0] == 0 the orbitals are numbered within each l in sequence order.
   Returns 0, or -1 if the sizes exceed the Fortran limits. */
int ac_addr_setup(ac_addr *a, const ac_cfptab *t, int kad, int nwt,
                  const int32_t *nwth, const int32_t *nwlf,
                  const int32_t *nwsq);

/* INT1AD: address of h(m1, m2) (m1, m2 sequence numbers, same l). */
int ac_addr_one(const ac_addr *a, int m1, int m2);

/* INT2AD: address of R^k(m1 m2; m3 m4) with kxx = 2k+1. */
int ac_addr_two(const ac_addr *a, int kxx, int m1, int m2, int m3, int m4);

#endif
