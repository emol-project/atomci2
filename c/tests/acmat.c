/*
 * acmat.c -- invariants <F~ O G> of a list of operators between all CSFs of
 * a set of configurations (stage 2, S2-M1; driven by c/python/test_s2m1.py).
 *
 * usage: acmat CFPTBL < spec
 *
 * spec (shell indices are 1-based slots of the shell list; ranks as 2j):
 *   shells l1 l2 ...
 *   config n1 n2 ...                         (one line per configuration)
 *   one a b ts tl                            (b_a b~_b)^(ts/2, tl/2)
 *   two a b c d tsp tlp tsq tlq tsT tlT nrm  ((b_a b~_b)^p (b_c b~_d)^q)^T,
 *                                            normal-ordered when nrm = 1
 *   end
 * output:
 *   csf i config tS tL          (i from 0; tS = 2S, tL = 2L of the total)
 *   v op i j value              (bra i, ket j; only |value| > 1e-14)
 * Pairs whose totals cannot be coupled by the operator's total rank are
 * skipped.
 */
#include "ac_cfp.h"
#include "ac_csfgen.h"
#include "ac_factor.h"
#include "ac_optree.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXSH 16
#define MAXOP 4096

typedef struct {
    int two, a, b, c, d, nrm;
    ac_lab p, q, tot;
} opspec;

static int tri(int a, int b, int c)
{
    return (a + b + c) % 2 == 0 && c <= a + b && c >= abs(a - b);
}

int main(int argc, char **argv)
{
    ac_cfptab t;
    char err[256];
    if (argc != 2 || ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "usage: %s CFPTBL < spec\n", argv[0]);
        return 2;
    }
    int nsh = 0, l[MAXSH];
    static opspec ops[MAXOP];
    int nop = 0;
    ac_csflist csfs;
    ac_csflist_init(&csfs);
    int *cfg_of = NULL, ncfg = 0;

    char word[32];
    while (scanf("%31s", word) == 1) {
        if (!strcmp(word, "shells")) {
            int x;
            while (nsh < MAXSH && scanf("%d", &x) == 1)
                l[nsh++] = x;
        } else if (!strcmp(word, "config")) {
            int occ[MAXSH];
            for (int k = 0; k < nsh; k++)
                if (scanf("%d", &occ[k]) != 1)
                    return 2;
            int before = csfs.len;
            if (ac_csfgen(&t, nsh, l, occ, &csfs) < 0) {
                fprintf(stderr, "acmat: cannot generate config %d\n", ncfg);
                return 1;
            }
            cfg_of = realloc(cfg_of, (size_t)csfs.len * sizeof *cfg_of);
            for (int i = before; i < csfs.len; i++)
                cfg_of[i] = ncfg;
            ncfg++;
        } else if (!strcmp(word, "one") && nop < MAXOP) {
            opspec *o = &ops[nop++];
            memset(o, 0, sizeof *o);
            if (scanf("%d %d %d %d", &o->a, &o->b, &o->p.ts, &o->p.tl) != 4)
                return 2;
            o->tot = o->p;
        } else if (!strcmp(word, "two") && nop < MAXOP) {
            opspec *o = &ops[nop++];
            o->two = 1;
            if (scanf("%d %d %d %d %d %d %d %d %d %d %d", &o->a, &o->b, &o->c, &o->d,
                      &o->p.ts, &o->p.tl, &o->q.ts, &o->q.tl, &o->tot.ts, &o->tot.tl,
                      &o->nrm) != 11)
                return 2;
        } else if (!strcmp(word, "end")) {
            break;
        } else {
            fprintf(stderr, "acmat: unknown keyword %s\n", word);
            return 2;
        }
    }
    if (nsh == 0) {
        fprintf(stderr, "acmat: no shells\n");
        return 2;
    }

    for (int i = 0; i < csfs.len; i++) {
        const ac_csf *c = &csfs.v[i];
        printf("csf %d %d %d %d\n", i, cfg_of[i], c->cpl[nsh].ts, c->cpl[nsh].tl);
    }

    for (int k = 0; k < nop; k++) {
        const opspec *o = &ops[k];
        for (int i = 0; i < csfs.len; i++)
            for (int j = 0; j < csfs.len; j++) {
                const ac_csf *F = &csfs.v[i], *G = &csfs.v[j];
                ac_lab FT = F->cpl[nsh], GT = G->cpl[nsh];
                if (!tri(o->tot.ts, GT.ts, FT.ts) || !tri(o->tot.tl, GT.tl, FT.tl))
                    continue;
                ac_otree tr;
                if (o->two)
                    ac_ot_two_body(&tr, o->a, l[o->a - 1], o->b, l[o->b - 1], o->p, o->c,
                                   l[o->c - 1], o->d, l[o->d - 1], o->q, o->tot);
                else
                    ac_ot_one_body(&tr, o->a, l[o->a - 1], o->b, l[o->b - 1], o->p);
                ac_oexpr e;
                ac_oexpr_init(&e);
                ac_oexpr_push(&e, 1.0, &tr);
                int er;
                double v = ac_factor_invariant(&t, F, G, &e, &er);
                ac_oexpr_free(&e);
                if (!er && o->two && o->nrm && o->b == o->c) {
                    ac_oexpr_init(&e);
                    ac_oexpr_push(&e, 1.0, &tr);
                    if (ac_oexpr_contraction(&e))
                        er = AC_FACTOR_ERR_TREE;
                    else
                        v -= ac_factor_invariant(&t, F, G, &e, &er);
                    ac_oexpr_free(&e);
                }
                if (er) {
                    fprintf(stderr, "acmat: error %d, op %d, pair %d %d\n", er, k, i, j);
                    return 1;
                }
                if (fabs(v) > 1e-14)
                    printf("v %d %d %d %.17g\n", k, i, j, v);
            }
    }
    ac_csflist_free(&csfs);
    free(cfg_of);
    ac_cfp_free(&t);
    return 0;
}
