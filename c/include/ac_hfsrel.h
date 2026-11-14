/*
 * ac_hfsrel.h -- the regularized alpha^2 correction to the hyperfine
 * contact interaction of an S state (S2-M11, technical note,
 * Sec. 5.2): Puchalski and Pachucki's A_AN (M. Puchalski and
 * K. Pachucki, "Ground state hyperfine splitting in 6,7Li atoms and the
 * nuclear structure", arXiv:1310.2161v2 (2013), Eq. (29)) as A_AN/A^(4),
 * regularized with their Eqs. (33), (34),
 *
 *   A_AN = 2 <X' G' H'^A> + T,
 *   T = sum_a <K_a> + sep + N2e - (4/3) sum_a <w_a V_a psi|H^A_2e psi>
 *       - 2 A4 <g> + 2 <F> <H^A> + Z^2 A4,
 *
 * in the CSF space of one (S, L = 0) block. The three-body Breit product and
 * the second-order term are taken through the resolution of the identity in
 * that space (exact for the three-body part when the CSF list is the
 * complete CI of its orbitals, and the same-electron part converges with
 * the orbital set). The orbitals must satisfy the nuclear
 * cusp (ATSP's numerical orbitals do; Gaussians do not).
 */
#ifndef AC_HFSREL_H
#define AC_HFSREL_H

#include "ac_ci.h"
#include "ac_radial.h"

#include <stdio.h>

/* the terms, in units of A4 (the contact term with the same spin factor) */
enum {
    AC_HR_K, AC_HR_SEP, AC_HR_N2E, AC_HR_BREIT, AC_HR_PROD, AC_HR_ANOM, AC_HR_S2, AC_HR_N
};

typedef struct {
    int ts, ncsf;
    double e;          /* CI energy                                         */
    double a4pp;       /* A^(4) in Puchalski-Pachucki's units (sigma_z at
                          M = S, times 2): (8 pi Z/3) <sum_a sigma_az delta(r_a)> */
    double ha;         /* <H^A>                                             */
    double t[AC_HR_N]; /* terms / A4                                        */
    double total;      /* A_AN / A4                                         */
    double delta;      /* <sum_{a<b} delta(r_ab)> of the CI                 */
    double ddelta;     /* d(A_AN/A4)/d<sum delta> through 2 <F> <H^A>       */
    double chk_x;      /* <X'>/A4 - 1 (0 for the exact eigenstate)          */
    double chk_h;      /* (<H'^A> - <H^A>)/A4                               */
} ac_hfsrel_res;

/* For every (S, L = 0) block of ci, its lowest root and A_AN/A^(4), printed
   to out and returned in res[0..*nres-1] (res may be NULL; at most maxres).
   ci->in.h and rk give the non-relativistic Hamiltonian, ci->in.roo the
   orbit-orbit radial integrals (required); mo[s-1] is the MO of slot s in
   rad. ci->in is restored on return. */
/* An external <sum_{a<b} delta(r_ab)> (NAN: none): the value of the CI lacks
   the electron-electron cusp and is ~10% high for Li; with a better one
   (Drachman's form, R. J. Drachman, J. Phys. B 14, 2733 (1981), Eq. (6);
   or a Hylleraas value such as 0.544325 for
   Li 2S, M. Puchalski and K. Pachucki, physics/0601213) the A_AN/A4 with it
   replaced in the product term 2 <F> <H^A> is also printed. */
extern double ac_hfsrel_delta;

int ac_hfsrel_print(FILE *out, ac_ci *ci, const ac_radial *rad, const int *mo,
                    ac_hfsrel_res *res, int maxres, int *nres, char *err, int errlen);

#endif
