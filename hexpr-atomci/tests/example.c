/*
 * example.c -- uses only the public header. Prints the diagonal element of the
 * first CSF of 1s2 2s2 2p and checks three of its coefficients by elementary
 * rules: 2 h(1s,1s), 1 h(2p,2p) and 1 R^0(1s1s;1s1s).
 */
#include "hexpr_atomci.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    ac_cfptab t;
    char err[256];
    if (ac_cfp_generate(&t, err, sizeof err))
        return 2;

    /* orbitals 1s, 2s, 2p; the configurations 1s2 2s2 2p and 1s2 2s 2p2 */
    int lorb[] = {0, 0, 1};
    int occ[] = {2, 2, 1,
                 2, 1, 2};
    ac_orb_csf *csf;
    int n;
    if (ac_orb_csfs(&t, 3, lorb, 2, occ, &csf, &n, err, sizeof err))
        return 2;

    ac_expr l;
    ac_expr_init(&l);
    if (ac_pair_expr(&t, &csf[0], &csf[0], 1e-13, &l))
        return 2;
    double h1s = 0, h2p = 0, r0 = 0;
    printf("%d CSFs; <0|H|0>:\n", n);
    for (int i = 0; i < l.len; i++) {
        /* k < 0: coef h(a,b); k >= 0: coef R^k(ab;cd) */
        const ac_expr_term *x = &l.v[i];
        printf("%3d %d %d %d %d %.15g\n", x->k, x->a, x->b, x->c, x->d, x->coef);
        if (x->k < 0 && x->a == 1 && x->b == 1) h1s = x->coef;
        if (x->k < 0 && x->a == 3 && x->b == 3) h2p = x->coef;
        if (x->k == 0 && x->a == 1 && x->b == 1 && x->c == 1 && x->d == 1) r0 = x->coef;
    }
    ac_expr_free(&l);
    free(csf);
    ac_cfp_free(&t);
    if (fabs(h1s - 2) > 1e-12 || fabs(h2p - 1) > 1e-12 || fabs(r0 - 1) > 1e-12) {
        printf("FAILED: h(1s) %.15g h(2p) %.15g R0(1s) %.15g\n", h1s, h2p, r0);
        return 1;
    }
    printf("passed\n");
    return 0;
}
