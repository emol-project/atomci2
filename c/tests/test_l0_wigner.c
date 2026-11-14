/*
 * test_l0_wigner.c -- L0: ac_wigner against the Fortran XWIG6J, XWIG9X,
 * DELTA, F3L and FSUBK, called in the same process.
 *
 * The Fortran routines take multiplicities (2j+1); the C ones take 2j.
 *   6j        all 2j in [0, TJ6MAX]^6, valid or not
 *   9j        all 2j in [0, TJ9MAX]^9 whose rows and columns are triangles
 *   DELTA     all (2a, 2b, 2c) in [0, TJ6MAX]^3 with integer perimeter
 *   F(a,b,c)  all integer a, b, c in [0, LMAX] with a+b+c even
 *             (the Fortran F3L does not test parity; odd sums are skipped)
 *   H^k       all k, l1..l4 in [0, LMAX], against the FSUBK statement
 *             function of src/smat.f evaluated with the Fortran F3L
 * Prints the number of comparisons and the largest |C - Fortran| per kind;
 * exits 1 if any difference exceeds TOL.
 */
#include "ac_wigner.h"
#include "ftn_decl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TJ6MAX 12
#define TJ9MAX 8
#define LMAX 9
#define TOL 1e-13

typedef struct {
    const char *name;
    long n;
    double maxdiff;
    long nbad;
} stat;

static void record(stat *s, double c, double f, const char *what)
{
    double d = fabs(c - f);
    s->n++;
    if (d > s->maxdiff || isnan(d))
        s->maxdiff = isnan(d) ? INFINITY : d;
    if (!(d <= TOL)) {
        if (s->nbad < 5)
            fprintf(stderr, "%s mismatch %s: C %.17g Fortran %.17g\n", s->name,
                    what, c, f);
        s->nbad++;
    }
}

static int tri(int a, int b, int c)
{
    return (a + b + c) % 2 == 0 && c <= a + b && c >= abs(a - b);
}

/* FSUBK of src/smat.f, with multiplicity arguments, using the Fortran F3L */
static double fsubk_ftn(int32_t L, int32_t L1, int32_t L2, int32_t L3, int32_t L4)
{
    int m = ((L1 - L2 + L3 - L4 + 88) / 4) % 2;
    double ph = 1.0 - 2.0 * abs(m);
    return ph * sqrt((double)L1 * L2 * L3 * L4 / L) * f3l_(&L, &L1, &L2) *
           f3l_(&L, &L3, &L4);
}

int main(void)
{
    setwig_();
    stat s6 = {"6j", 0, 0, 0}, s9 = {"9j", 0, 0, 0}, sd = {"DELTA", 0, 0, 0},
         sf = {"F(a,b,c)", 0, 0, 0}, sh = {"H^k", 0, 0, 0};
    char what[128];

    for (int a = 0; a <= TJ6MAX; a++)
        for (int b = 0; b <= TJ6MAX; b++)
            for (int c = 0; c <= TJ6MAX; c++)
                for (int d = 0; d <= TJ6MAX; d++)
                    for (int e = 0; e <= TJ6MAX; e++)
                        for (int f = 0; f <= TJ6MAX; f++) {
                            int32_t m[6] = {a + 1, b + 1, c + 1, d + 1, e + 1, f + 1};
                            double vf = xwig6j_(&m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
                            double vc = ac_sixj(a, b, c, d, e, f);
                            snprintf(what, sizeof what, "{%d %d %d; %d %d %d}/2",
                                     a, b, c, d, e, f);
                            record(&s6, vc, vf, what);
                        }

    for (int a = 0; a <= TJ9MAX; a++)
        for (int b = 0; b <= TJ9MAX; b++)
            for (int c = 0; c <= TJ9MAX; c++) {
                if (!tri(a, b, c))
                    continue;
                for (int d = 0; d <= TJ9MAX; d++)
                    for (int e = 0; e <= TJ9MAX; e++)
                        for (int f = 0; f <= TJ9MAX; f++) {
                            if (!tri(d, e, f))
                                continue;
                            for (int g = 0; g <= TJ9MAX; g++) {
                                if (!tri(a, d, g))
                                    continue;
                                for (int h = 0; h <= TJ9MAX; h++) {
                                    if (!tri(b, e, h))
                                        continue;
                                    for (int i = 0; i <= TJ9MAX; i++) {
                                        if (!tri(g, h, i) || !tri(c, f, i))
                                            continue;
                                        int32_t m[9] = {a + 1, b + 1, c + 1, d + 1, e + 1,
                                                        f + 1, g + 1, h + 1, i + 1};
                                        double vf = xwig9x_(&m[0], &m[1], &m[2], &m[3],
                                                            &m[4], &m[5], &m[6], &m[7], &m[8]);
                                        double vc = ac_ninej(a, b, c, d, e, f, g, h, i);
                                        snprintf(what, sizeof what,
                                                 "{%d %d %d; %d %d %d; %d %d %d}/2", a, b,
                                                 c, d, e, f, g, h, i);
                                        record(&s9, vc, vf, what);
                                    }
                                }
                            }
                        }
            }

    for (int a = 0; a <= TJ6MAX; a++)
        for (int b = 0; b <= TJ6MAX; b++)
            for (int c = 0; c <= TJ6MAX; c++) {
                if ((a + b + c) % 2)
                    continue;
                int32_t m[3] = {a + 1, b + 1, c + 1};
                snprintf(what, sizeof what, "(%d %d %d)/2", a, b, c);
                record(&sd, ac_tri_delta(a, b, c), delta_(&m[0], &m[1], &m[2]), what);
            }

    for (int a = 0; a <= 2 * LMAX; a++)
        for (int b = 0; b <= LMAX; b++)
            for (int c = 0; c <= LMAX; c++) {
                if ((a + b + c) % 2)
                    continue;
                int32_t m[3] = {2 * a + 1, 2 * b + 1, 2 * c + 1};
                snprintf(what, sizeof what, "F(%d,%d,%d)", a, b, c);
                record(&sf, ac_fabc(a, b, c), f3l_(&m[0], &m[1], &m[2]), what);
            }

    for (int k = 0; k <= 2 * LMAX; k++)
        for (int l1 = 0; l1 <= LMAX; l1++)
            for (int l2 = 0; l2 <= LMAX; l2++)
                for (int l3 = 0; l3 <= LMAX; l3++)
                    for (int l4 = 0; l4 <= LMAX; l4++) {
                        /* FSUBK is only ever called with both parities even
                           (k stepped by 2); outside that F3L is not F */
                        if ((k + l1 + l2) % 2 || (k + l3 + l4) % 2)
                            continue;
                        double vf = fsubk_ftn(2 * k + 1, 2 * l1 + 1, 2 * l2 + 1,
                                              2 * l3 + 1, 2 * l4 + 1);
                        snprintf(what, sizeof what, "H^%d(%d,%d,%d,%d)", k, l1, l2,
                                 l3, l4);
                        record(&sh, ac_hk(k, l1, l2, l3, l4), vf, what);
                    }

    stat *all[] = {&s6, &s9, &sd, &sf, &sh};
    long bad = 0;
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) {
        printf("%-9s %10ld compared   max |C-F| = %.3e   over %.0e: %ld\n",
               all[i]->name, all[i]->n, all[i]->maxdiff, TOL, all[i]->nbad);
        bad += all[i]->nbad;
    }
    printf("L0 %s\n", bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}
