/*
 * test_terms.c -- the expressions of ac_terms.h against reference values.
 *
 * usage: test_terms REF        (REF: tests/ref_hij.txt)
 *
 * For every CSF pair of every space, with the integrals of terms_spaces.h:
 *   terms:  the sum of coef x integral from ac_pair_terms equals the reference;
 *   merged: the same from ac_pair_expr, with labels canonical, sorted and unique;
 *   parts:  skipping all two-electron terms leaves the one-electron terms;
 *   real:   the merged expression of (i, j) equals that of (j, i), since H is
 *           real and symmetric for real orbitals.
 * Pairs that are not in REF must give 0.
 */
#include "ac_terms.h"
#include "terms_spaces.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-11
#define TOL_MERGE 1e-13

static double integral(int k, int a, int b, int c, int d)
{
    terms_canon(&k, &a, &b, &c, &d);
    return terms_hashval(k, a, b, c, d);
}

typedef struct {
    double sum;
    long n, n1;
    double sum1;
} sum_ctx;

static void add(void *p, int k, int a, int b, int c, int d, double coef)
{
    sum_ctx *s = p;
    double v = coef * integral(k, a, b, c, d);
    s->sum += v;
    s->n++;
    if (k < 0) {
        s->sum1 += v;
        s->n1++;
    }
}

static int skip_all(void *ctx, int k, int a, int b, int c, int d)
{
    (void)ctx, (void)k, (void)a, (void)b, (void)c, (void)d;
    return 1;
}

static int run_space(const ac_cfptab *t, const space *s, FILE *ref)
{
    char word[32], name[128], err[256];
    int ref_n = 0;
    if (fscanf(ref, "%31s %127s %d", word, name, &ref_n) != 3 || strcmp(word, "space") ||
        strcmp(name, s->name)) {
        printf("FAIL %s: reference out of step (read %s)\n", s->name, name);
        return 1;
    }
    int occ[64];
    for (int c = 0; c < s->ncfg; c++)
        for (int o = 0; o < s->nwt; o++)
            occ[c * s->nwt + o] = s->occ[c][o];
    ac_orb_csf *csf;
    int n;
    if (ac_orb_csfs(t, s->nwt, s->lorb, s->ncfg, occ, &csf, &n, err, sizeof err)) {
        printf("FAIL %s: %s\n", s->name, err);
        return 1;
    }
    if (n != ref_n) {
        printf("FAIL %s: %d CSFs, reference %d\n", s->name, n, ref_n);
        free(csf);
        return 1;
    }
    double *val = calloc((size_t)n * n, sizeof *val);
    int i, j;
    double v;
    long nref = 0;
    for (;;) {
        long pos = ftell(ref);
        if (fscanf(ref, "%d %d %lf", &i, &j, &v) != 3) {
            clearerr(ref);
            fseek(ref, pos, SEEK_SET);
            break;
        }
        val[(size_t)i * n + j] = v;
        nref++;
    }

    double dterm = 0, dmerged = 0, dparts = 0, dreal = 0;
    long raw = 0, merged = 0;
    int bad = 0;
    ac_expr l, lr;
    ac_expr_init(&l);
    ac_expr_init(&lr);
    for (i = 0; i < n && !bad; i++)
        for (j = 0; j < n && !bad; j++) {
            sum_ctx a = {0}, b = {0};
            long nops = 0;
            if (ac_pair_terms(t, &csf[i], &csf[j], NULL, add, &a, &nops) ||
                ac_pair_terms(t, &csf[i], &csf[j], skip_all, add, &b, NULL) ||
                ac_pair_expr(t, &csf[i], &csf[j], TOL_MERGE, &l) ||
                ac_pair_expr(t, &csf[j], &csf[i], TOL_MERGE, &lr)) {
                printf("FAIL %s: error at pair %d %d\n", s->name, i, j);
                bad = 1;
                break;
            }
            raw += a.n;
            merged += l.len;
            double r = val[(size_t)i * n + j], m = 0;
            for (int t2 = 0; t2 < l.len; t2++) {
                const ac_expr_term *x = &l.v[t2];
                m += x->coef * integral(x->k, x->a, x->b, x->c, x->d);
                if (t2 && (x[-1].k > x->k || (x[-1].k == x->k && x[-1].a > x->a)))
                    bad = 1;
                if (t2 && !memcmp(&x[-1], x, 5 * sizeof(int)))
                    bad = 1;
            }
            if (fabs(a.sum - r) > dterm) dterm = fabs(a.sum - r);
            if (fabs(m - r) > dmerged) dmerged = fabs(m - r);
            if (fabs(a.sum1 - b.sum) > dparts) dparts = fabs(a.sum1 - b.sum);
            if (b.n != a.n1) bad = 1;
            if (l.len != lr.len) {
                dreal = 1.0;
            } else {
                for (int t2 = 0; t2 < l.len; t2++) {
                    if (memcmp(&l.v[t2], &lr.v[t2], 5 * sizeof(int))) {
                        dreal = 1.0;
                        break;
                    }
                    double d = fabs(l.v[t2].coef - lr.v[t2].coef);
                    if (d > dreal) dreal = d;
                }
            }
        }
    ac_expr_free(&l);
    ac_expr_free(&lr);
    int ok = !bad && dterm <= TOL && dmerged <= TOL && dparts <= TOL && dreal <= TOL && nref > 0;
    printf("%s %-24s csf %4d ref %6ld raw %7ld merged %7ld | terms %.1e merged %.1e parts %.1e "
           "real %.1e\n",
           ok ? "ok  " : "FAIL", s->name, n, nref, raw, merged, dterm, dmerged, dparts, dreal);
    free(val);
    free(csf);
    return !ok;
}

int main(int argc, char **argv)
{
    ac_cfptab t;
    char err[256];
    FILE *ref = argc == 2 ? fopen(argv[1], "r") : NULL;
    if (!ref) {
        fprintf(stderr, "usage: test_terms REF\n");
        return 2;
    }
    if (ac_cfp_generate(&t, err, sizeof err)) {
        fprintf(stderr, "test_terms: %s\n", err);
        return 2;
    }
    int bad = 0;
    for (int i = 0; i < NSPACES; i++)
        bad += run_space(&t, &spaces[i], ref);
    fclose(ref);
    ac_cfp_free(&t);
    if (bad) {
        printf("FAILED: %d spaces\n", bad);
        return 1;
    }
    printf("passed\n");
    return 0;
}
