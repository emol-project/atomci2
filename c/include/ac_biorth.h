/*
 * ac_biorth.h -- biorthonormal orbital sets, for matrix elements between
 * states whose orbitals were optimized separately (stage 2, S2-M9;
 * technical note, Sec. 5.4). The method of Olsen, Godefroid,
 * Joensson, Malmqvist and Froese Fischer, Phys. Rev. E 52, 4499 (1995):
 *
 *   1. for each l, the overlap S = <X|Y> of the two orbital sets is split
 *      S = L U (no pivoting), and A = X (L^-1)^T, B = Y U^-1 are
 *      biorthonormal, <A_i|B_j> = delta_ij (ac_biorth_lu);
 *   2. the CI vector of each state is rewritten in its new orbitals,
 *      X = A t with t upper triangular, by one orbital k at a time in
 *      increasing order:
 *        c <- exp(sum_(a<k) (t_ak / t_kk) E_ak) t_kk^(N_k) c
 *      (ac_biorth_transform), which is exact when the CSF space is closed
 *      under E_ak, a < k (ac_biorth_closure);
 *   3. matrix elements between the two states are then those of an
 *      orthonormal set, with the bra on A and the ket on B.
 *
 * Orbitals are the sequence orbitals 1..nwt of an ac_ci; "a < k" is the
 * order of their sequence numbers within one l.
 */
#ifndef AC_BIORTH_H
#define AC_BIORTH_H

#include "ac_ci.h"
#include "ac_radial.h"

/* how the pivots d_i of S = L U are split between L and U: L_ii = sqrt|d_i|
   (and U_ii = d_i / sqrt|d_i|), L_ii = 1, or U_ii = 1. No choice changes a
   matrix element between the two states (a check of the implementation). */
enum { AC_BIORTH_SYM = 0, AC_BIORTH_LUNIT = 1, AC_BIORTH_UUNIT = 2 };

/* One l: S[p*m + q] = <X_p|Y_q> for n orbitals X and m orbitals Y, each in
   its triangular order. Computes the upper-triangular cx[n*n] and cy[m*m]
   (row-major) of A = X cx, B = Y cy with <A_i|B_j> = delta_ij for i, j <
   min(n, m); when n != m the extra orbitals of the larger set (last in its
   order) are made orthogonal to the whole smaller set (Olsen et al.,
   appendix B). tx = cx^-1 and ty = cy^-1 (X = A tx, Y = B ty), also upper
   triangular. n or m may be 0. Fails when a pivot |d_i| < tol. */
int ac_biorth_lu(int n, int m, const double *S, int mode, double tol, double *cx, double *cy,
                 double *tx, double *ty, char *err, int errlen);

/* dst receives nmo MOs, MO j = sum_i coef[j*ntot + i] (MO i of the
   concatenation of src[0..nsrc-1], ntot MOs in all), with l[j] its l. The
   MOs must be grouped by l in increasing order. All sources must share one
   grid (dst takes the grid and Z of src[0]). R, R', the radial Laplacian
   and R(0) are linear in R, so every radial integral of ac_radial works on
   dst. */
int ac_radial_mix(ac_radial *dst, int nsrc, const ac_radial *const *src, int nmo, const int *l,
                  const double *coef, char *err, int errlen);

/* <i|E_ab|j>, E_ab = sum_(m_l, m_s) a+_(a m_l m_s) a_(b m_l m_s), for
   sequence orbitals a, b of one l: the coefficient of h(a, b) in
   ac_ci_hij (METECC (44); Olsen et al. (A4)). */
double ac_biorth_eij(ac_ci *ci, int i, int j, int a, int b, int *err);

/* The elements <i|E_ak|j>, a < k, of a CSF list, sorted by k. */
typedef struct {
    int ncsf, nwt;
    unsigned char *occ; /* [ncsf * (nwt+1)] occupation of each sequence orbital */
    int *start;         /* [nwt+2]: E_ak, a < k, are elements start[k]..start[k+1]-1 */
    int *a, *i, *j;
    double *v;
    int n;
} ac_biorth_ops;

int ac_biorth_ops_build(ac_biorth_ops *o, ac_ci *ci, char *err, int errlen);
void ac_biorth_ops_free(ac_biorth_ops *o);

/* c[v*ncsf + i] (nvec vectors) <- the same states with the orbitals X
   replaced by A, X = A t: t[a*(nwt+1) + k] = t_ak (sequence orbitals,
   1-based), used only for a <= k of equal l; t_kk must not be 0. */
int ac_biorth_transform(const ac_biorth_ops *o, const ac_ci *ci, const double *t, int nvec,
                        double *c, char *err, int errlen);

/* The closure of the CSF list under E_ak (a < k, equal l): for every CSF j
   and move k -> a, the squared norm of the part of E_ak|j> outside the span
   of the list (the CSFs of the target configuration are generated in full).
   *dev receives the largest; what (if not NULL) describes where. A target
   configuration that CFPTBL cannot couple counts as weight 1. */
int ac_biorth_closure(ac_ci *ci, double *dev, char *what, int whatlen, char *err, int errlen);

#endif
