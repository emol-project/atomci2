/*
 * ac_tabrad.h -- radial functions given on a table: the numerical
 * orbitals of ATSP (MCHF/HF, the <atom>.w files) as an ac_radial (stage 2,
 * comparison with ATSP).
 *
 * <atom>.w is gfortran unformatted, one record per orbital:
 *   ATOM*6, TERM*6, EL*3, M (int32), Z, E(nl), EK, AZ (real*8), F(1..M)
 * on the grid r_j = exp(-4 + (j-1)/16) / Z (hf.f, SUBROUTINE INIT). F is
 * ATSP's working variable P(r)/sqrt(r), with P(r) = r R(r) and
 * P ~ AZ r^(l+1) at the origin; ac_atspw_read stores P.
 *
 * On the ac_radial grid, Q = P / r^(l+1) is
 *   - for r >= r_1, the 8-point Lagrange interpolant in ln r;
 *   - for r < r_1, the degree-7 polynomial in r through (0, AZ) and
 *     r_1..r_7;
 *   - 0 beyond r_M.
 * R = Q r^l, R' = r^(l-1) (Q_x + l Q) and the radial Laplacian
 * R'' + 2R'/r - l(l+1)R/r^2 = r^(l-2) (Q_xx + (2l+1) Q_x), x = ln r, come from
 * the analytic derivatives of those interpolants.
 */
#ifndef AC_TABRAD_H
#define AC_TABRAD_H

#include "ac_radial.h"

#define AC_ATSP_MAXORB 128

typedef struct {
    int n;                       /* orbitals                      */
    char el[AC_ATSP_MAXORB][4];  /* labels, e.g. "2p"             */
    int l[AC_ATSP_MAXORB];
    int m[AC_ATSP_MAXORB];       /* points                        */
    double z;
    double e[AC_ATSP_MAXORB], az[AC_ATSP_MAXORB];
    double *p[AC_ATSP_MAXORB];   /* P(r_j), j = 0..m-1             */
} ac_atspw;

int ac_atspw_read(ac_atspw *w, const char *path, char *err, int errlen);
void ac_atspw_free(ac_atspw *w);

/* r_j of the ATSP grid (j from 0) */
double ac_atspw_r(const ac_atspw *w, int j);

/* An ac_radial on grid g with the orbitals of w as MOs, grouped by l
   (within an l in the order of w). mo[i] receives the MO of orbital i. */
int ac_radial_from_atsp(ac_radial *rad, const ac_atspw *w, const ac_rgrid *g, int *mo,
                        char *err, int errlen);

#endif
