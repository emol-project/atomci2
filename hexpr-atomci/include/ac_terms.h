/*
 * ac_terms.h -- the expression of a Hamiltonian matrix element between CSFs.
 *
 * <F|H|G> = sum_terms coef x integral, with the integrals h(a,b) and
 * R^k(ab;cd) left symbolic. The operators are those of METECC eqs. (44), (45);
 * a and c are on the creation side, b and d on the annihilation side, and
 * (a, b) belongs to electron 1. Orbitals are the caller's sequence numbers.
 *
 * A CSF lists its occupied orbitals ("slots") in increasing sequence order;
 * they are coupled in that order (ac_csf.h). Closed shells are ordinary slots.
 */
#ifndef AC_TERMS_H
#define AC_TERMS_H

#include "ac_cfp.h"
#include "ac_csf.h"

typedef struct {
    int seq[AC_CSF_MAXSH + 1]; /* [1..c.nsh] sequence orbital of each slot */
    ac_csf c;
} ac_orb_csf;

/* All CSFs, of every (S, L), of a list of configurations. Orbital s = 1..nwt
   has momentum lorb[s-1]; configuration c gives the occupation
   occ[c*nwt + s-1] of every orbital, and the occupied ones become its slots.
   Allocates *out (free it with free). Returns 0, or -1 with a message. */
int ac_orb_csfs(const ac_cfptab *t, int nwt, const int *lorb, int ncfg, const int *occ,
                ac_orb_csf **out, int *n, char *err, int errlen);

/* The two CSFs on the union of their slots: an orbital only one of them
   occupies becomes an empty shell in the other, which couples trivially.
   seq[1..n] receives the orbitals. Returns n, or -1 above AC_CSF_MAXSH slots. */
int ac_pair_union(const ac_cfptab *t, const ac_orb_csf *F, const ac_orb_csf *G, ac_csf *bf,
                  ac_csf *bg, int *seq);

/* Nonzero when every slot other than a, b, c and d (0 for none) carries the
   same shell state on both sides. */
int ac_pair_spectators_match(const ac_csf *bf, const ac_csf *bg, int a, int b, int c, int d);

/* One term: coef h(a,b) when k < 0 (c = d = 0), coef R^k(ab;cd) when k >= 0. */
typedef void (*ac_term_emit)(void *ctx, int k, int a, int b, int c, int d, double coef);

/* Asked before the angular coefficient of a two-electron term is computed;
   nonzero leaves the term out. For callers whose integral is zero. */
typedef int (*ac_term_skip)(void *ctx, int k, int a, int b, int c, int d);

/* The terms of <F|H|G>: emit is called once per ordered operator with a nonzero
   angular coefficient, so a label may repeat. skip and nops may be NULL; *nops
   is increased by the number of operators evaluated. Different total (S, L)
   gives no terms. Returns 0; 1 above AC_CSF_MAXSH slots; or the engine's error
   code (ac_factor.h). */
int ac_pair_terms(const ac_cfptab *t, const ac_orb_csf *F, const ac_orb_csf *G, ac_term_skip skip,
                  ac_term_emit emit, void *ctx, long *nops);

typedef struct {
    int k, a, b, c, d;
    double coef;
} ac_expr_term;

typedef struct {
    ac_expr_term *v;
    int len, cap;
} ac_expr;

void ac_expr_init(ac_expr *l);
void ac_expr_free(ac_expr *l);

/* The label an integral equals for real orbitals: h(a,b) = h(b,a);
   R^k(ab;cd) = R^k(ba;cd) = R^k(ab;dc) = R^k(cd;ab). A one-electron label gets
   k = -1 and c = d = 0. */
void ac_term_canon(int *k, int *a, int *b, int *c, int *d);

/* <F|H|G> as one term per canonical label, sorted by (k, a, b, c, d), for real
   orbitals. Terms with |coef| <= tol after merging are dropped. l is replaced.
   Returns as ac_pair_terms, or -1 when out of memory. */
int ac_pair_expr(const ac_cfptab *t, const ac_orb_csf *F, const ac_orb_csf *G, double tol,
                 ac_expr *l);

#endif
