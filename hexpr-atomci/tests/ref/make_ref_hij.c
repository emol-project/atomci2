/*
 * make_ref_hij.c -- writes ref_hij.txt: <i|H|j> from atomci2's ac_ci_hij, before
 * atomci2 was moved onto this library, with the integrals of terms_spaces.h.
 * Built against an atomci2 tree (see README.md here), not against this library.
 *
 * Output: "space NAME ncsf" lines, then "i j value" for each nonzero element.
 */
#include "ac_ci.h"
#include "../terms_spaces.h"

#include <stdio.h>

static double rand_h(void *ctx, int a, int b)
{
    (void)ctx;
    int k = -1, c = 0, d = 0;
    terms_canon(&k, &a, &b, &c, &d);
    return terms_hashval(k, a, b, c, d);
}

static double rand_rk(void *ctx, int k, int a, int b, int c, int d)
{
    (void)ctx;
    terms_canon(&k, &a, &b, &c, &d);
    return terms_hashval(k, a, b, c, d);
}

int main(void)
{
    ac_cfptab t;
    char err[256];
    if (ac_cfp_generate(&t, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 2;
    }
    for (int sp = 0; sp < NSPACES; sp++) {
        const space *s = &spaces[sp];
        int occ[64];
        for (int c = 0; c < s->ncfg; c++)
            for (int o = 0; o < s->nwt; o++)
                occ[c * s->nwt + o] = s->occ[c][o];
        ac_ci ci;
        if (ac_ci_from_orbitals_occ(&ci, &t, s->nwt, s->lorb, s->ncfg, occ, err, sizeof err)) {
            fprintf(stderr, "%s: %s\n", s->name, err);
            return 2;
        }
        ci.in.h = rand_h;
        ci.in.rk = rand_rk;
        printf("space %s %d\n", s->name, ci.ncsf);
        for (int i = 0; i < ci.ncsf; i++)
            for (int j = 0; j < ci.ncsf; j++) {
                int er = 0;
                double v = ac_ci_hij(&ci, i, j, &er);
                if (er) {
                    fprintf(stderr, "%s: error %d at %d %d\n", s->name, er, i, j);
                    return 2;
                }
                if (v != 0.0)
                    printf("%d %d %.17g\n", i, j, v);
            }
        ac_ci_free(&ci);
    }
    ac_cfp_free(&t);
    return 0;
}
