/*
 * ac_factor.h -- matrix elements by the factorization of METECC eqs. (47)-(51),
 * with single-shell elements from eq. (48) (atomci2 technical note, Sec. 3.5).
 *
 * The invariant <F~ O G> couples the bra F, the operator O and the ket G to
 * the identity (METECC section 3.2); for a scalar O the matrix element
 * between components is <F~ O G> / sqrt[F].
 */
#ifndef AC_FACTOR_H
#define AC_FACTOR_H

#include "ac_cfp.h"
#include "ac_csf.h"
#include "ac_optree.h"

#include <stdint.h>

/* Error codes left in *err (0 = none). */
enum {
    AC_FACTOR_OK = 0,
    AC_FACTOR_ERR_TREE = 1,   /* a rewrite failed (misuse or memory)      */
    AC_FACTOR_ERR_STATES = 2, /* an intermediate shell-state list missing */
    AC_FACTOR_ERR_CSF = 3     /* a bad shell-state code                   */
};

/* <F~ O G> for the operator expression e (any tree shapes; e is consumed:
   it is rewritten to sequential form in place). */
double ac_factor_invariant(const ac_cfptab *t, const ac_csf *bra,
                           const ac_csf *ket, ac_oexpr *e, int *err);

/* <F~ O G> for the operator tree tr, normal-ordered when normal = 1 and tr
   is a two-body tree of ac_ot_two_body() whose b~_b and b_c share a shell.
   The sequential form of tr (and of its contraction) depends only on the
   tree, so it is computed once per distinct tree and cached (not thread
   safe). */
double ac_factor_op(const ac_cfptab *t, const ac_csf *bra, const ac_csf *ket,
                    const ac_otree *tr, int normal, int *err);

/* The quantity the Fortran SHMS returns, from SHMS's own arguments
   (multiplicities, Fortran numbering): the component matrix element of
     ((b_c1 b~_a1)^(lsi,lri) (b_c2 b~_a2)^(lsf,lrf))^(lst,lrt)
   or, when c2 > nht, of the one-body (b_c1 b~_a1)^(lsi,lri); bra = NHW*,
   ket = NHV*. normal_order = 1 subtracts the contraction of b~_a1 b_c2
   (only present when a1 == c2). Scalar totals only (lst = lrt = 1). */
double ac_factor_shms(const ac_cfptab *t, int nht, const int32_t *nhlf,
                      const int32_t *nhwn, const int32_t *nhwq,
                      const int32_t *nhwcsl, const int32_t *nhvn,
                      const int32_t *nhvq, const int32_t *nhvcsl, int c1,
                      int a1, int lsi, int lri, int c2, int a2, int lsf,
                      int lrf, int lst, int lrt, int normal_order, int *err);

#endif
