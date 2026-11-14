/*
 * d2_terms.c -- the terms of p2 and d2 as Slater integrals, with no radial
 * integrals at all. Uses only the public header.
 *
 * E = 2 h + a0 F0 + a2 F2 + a4 F4, with F2 = F^2/25 for p and F2 = F^2/49,
 * F4 = F^4/441 for d. The closed forms below are the standard ones; for d2 the
 * Racah parameters follow from F0 = A + 7C/5, F2 = B + C/7, F4 = C/35.
 */
#include "hexpr_atomci.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int ts, tl;     /* 2S, 2L */
    double a0, a2, a4;
} term;

static const term p2[] = {
    {2, 2, 1, -5, 0},
    {0, 4, 1, 1, 0},
    {0, 0, 1, 10, 0},
};

static const term d2[] = {
    {2, 6, 1, -8, -9},
    {2, 2, 1, 7, -84},
    {0, 8, 1, 4, 1},
    {0, 4, 1, -3, 36},
    {0, 0, 1, 14, 126},
};

/* Checks one shell l^2; returns the number of failures. */
static int shell(const ac_cfptab *t, int l, const term *want, int nwant, double d2k, double d4k)
{
    char err[256];
    int lorb[] = {l};
    int occ[] = {2};
    ac_orb_csf *csf;
    int n, bad = 0;
    if (ac_orb_csfs(t, 1, lorb, 1, occ, &csf, &n, err, sizeof err)) {
        printf("FAILED: %s\n", err);
        return 1;
    }
    if (n != nwant) {
        printf("FAILED: %d CSFs of %c2, expected %d\n", n, "spd"[l], nwant);
        bad++;
    }
    ac_expr e;
    ac_expr_init(&e);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (ac_pair_expr(t, &csf[i], &csf[j], 1e-13, &e)) {
                printf("FAILED: engine error\n");
                bad++;
            } else if (i != j && e.len != 0) {
                printf("FAILED: terms between different (S, L)\n");
                bad++;
            }
            if (i != j)
                continue;
            const ac_lab *lab = &csf[i].c.cpl[csf[i].c.nsh];
            double h = 0, a[3] = {0, 0, 0};
            for (int x = 0; x < e.len; x++) {
                const ac_expr_term *y = &e.v[x];
                if (y->k < 0)
                    h = y->coef;
                else if (y->k == 0)
                    a[0] = y->coef;
                else if (y->k == 2)
                    a[1] = y->coef * d2k;
                else if (y->k == 4)
                    a[2] = y->coef * d4k;
            }
            printf("%c2 %d%c:  %g h  %+g F0  %+g F2", "spd"[l], lab->ts + 1, "SPDFG"[lab->tl / 2], h,
                   a[0], a[1]);
            if (l == 2) {
                /* a0 F0 + a2 F2 + a4 F4 = a0 A + a2 B + (7 a0/5 + a2/7 + a4/35) C */
                double c = (49 * a[0] + 5 * a[1] + a[2]) / 35;
                printf("  %+g F4   = %+g A %+g B %+g C", a[2], a[0], a[1], fabs(c) < 1e-11 ? 0.0 : c);
            }
            printf("\n");
            int found = 0;
            for (int w = 0; w < nwant; w++) {
                if (want[w].ts != lab->ts || want[w].tl != lab->tl)
                    continue;
                found = 1;
                if (fabs(h - 2) > 1e-11 || fabs(a[0] - want[w].a0) > 1e-11 ||
                    fabs(a[1] - want[w].a2) > 1e-11 || fabs(a[2] - want[w].a4) > 1e-11) {
                    printf("FAILED: expected %g F0 %+g F2 %+g F4\n", want[w].a0, want[w].a2,
                           want[w].a4);
                    bad++;
                }
            }
            if (!found) {
                printf("FAILED: unexpected term\n");
                bad++;
            }
        }
    }
    ac_expr_free(&e);
    free(csf);
    return bad;
}

int main(void)
{
    ac_cfptab t;
    char err[256];
    if (ac_cfp_generate(&t, err, sizeof err)) {
        printf("FAILED: %s\n", err);
        return 2;
    }
    int bad = shell(&t, 1, p2, 3, 25, 1) + shell(&t, 2, d2, 5, 49, 441);
    ac_cfp_free(&t);
    if (bad)
        return 1;
    printf("passed\n");
    return 0;
}
