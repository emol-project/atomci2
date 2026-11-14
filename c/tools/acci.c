/*
 * acci.c -- the non-relativistic CI of stage 2 on the CSFs of a Fortran
 * energy-expression tape and the integrals of its NFTRI (S2-M2, check V2a).
 *
 * usage: acci CFPTBL nfte nftri [nroots [cmp]]
 *
 * Builds H with closed shells as explicit slots and the general-form engine
 * (no Fortran energy expression is used), diagonalizes it and prints the
 * lowest eigenvalues with 15 digits, for comparison with the Fortran CI on
 * the same CSFs and integrals.
 *
 * With "cmp", H is also built from the Fortran energy expression on the
 * tape (the coefficients times the same integrals), diagonalized, and
 * compared element by element with the engine's H: the largest difference,
 * and how much of it sits on elements the tape has no term for.
 */
#include "ac_cfp.h"
#include "ac_ci.h"
#include "ac_ftnio.h"
#include "ac_ints.h"
#include "ac_tape.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s CFPTBL nfte nftri [nroots]\n", argv[0]);
        return 2;
    }
    int nroots = argc > 4 ? atoi(argv[4]) : 3;
    char err[256];
    ac_cfptab t;
    ac_ints ints;
    ac_nfte_head h;
    ac_ftn_file f;
    if (ac_cfp_load(&t, argv[1], err, sizeof err) ||
        ac_ints_load(&ints, argv[3], err, sizeof err)) {
        fprintf(stderr, "acci: %s\n", err);
        return 1;
    }
    if (ac_ftn_open(&f, argv[2], "rb") || ac_nfte_read_head(&f, &h, err, sizeof err)) {
        fprintf(stderr, "acci: %s: %s\n", argv[2], err);
        return 1;
    }
    ac_ftn_close(&f);
    ac_ci ci;
    if (ac_ci_from_nfte(&ci, &t, &ints, &h, err, sizeof err)) {
        fprintf(stderr, "acci: %s\n", err);
        return 1;
    }
    int n = ci.ncsf;
    printf("acci: %d CSFs, 2S+1 = %d, 2L+1 = %d, %ld integrals (%ld one-electron)\n", n,
           h.ispin, h.iorb, ints.n, ints.n1);

    double *H = calloc((size_t)n * n, sizeof *H), *w = malloc((size_t)n * sizeof *w);
    if (!H || !w)
        return 1;
    clock_t t0 = clock();
    for (int j = 0; j < n; j++)
        for (int i = j; i < n; i++) {
            int e;
            double v = ac_ci_hij(&ci, i, j, &e);
            if (e) {
                fprintf(stderr, "acci: error %d for pair %d %d\n", e, i, j);
                return 1;
            }
            H[(size_t)j * n + i] = v;
        }
    double tb = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("H(1,1) = %.15f\n", H[0]);
    printf("built in %.1f s, %ld operator evaluations\n", tb, ci.ops_evaluated);

    if (argc > 5 && argv[5][0] == 'c') {
        /* H from the Fortran energy expression on the tape */
        double *T = calloc((size_t)n * n, sizeof *T);
        char *has = calloc((size_t)n * n, 1);
        ac_ftn_file g;
        ac_nfte_head h2;
        if (!T || !has || ac_ftn_open(&g, argv[2], "rb") ||
            ac_nfte_read_head(&g, &h2, err, sizeof err))
            return 1;
        ac_nfte_rec r;
        memset(&r, 0, sizeof r);
        while (ac_nfte_read_rec(&g, &r) == AC_FTN_OK) {
            if (r.noe == 0)
                continue;
            for (int p = 0; p < r.nij; p++) {
                int ii = r.indexi[p] - 1, jj = (r.nii ? r.indexj[p] : r.indexi[p]) - 1;
                double v = 0.0;
                for (int e = 0; e < r.noe; e++)
                    v += r.wt[e] * ints.x[r.isqn[p * r.noe + e]];
                size_t at = (size_t)(ii > jj ? jj : ii) * n + (ii > jj ? ii : jj);
                T[at] += v;
                has[at] = 1;
            }
        }
        ac_ftn_close(&g);
        double dmax = 0.0, dmax_missing = 0.0, dsum_missing = 0.0;
        long nmissing = 0;
        for (int j = 0; j < n; j++)
            for (int i = j; i < n; i++) {
                size_t at = (size_t)j * n + i;
                double d = fabs(H[at] - T[at]);
                if (d > dmax)
                    dmax = d;
                if (!has[at] && H[at] != 0.0) {
                    nmissing++;
                    dsum_missing += fabs(H[at]);
                    if (fabs(H[at]) > dmax_missing)
                        dmax_missing = fabs(H[at]);
                }
            }
        printf("cmp: max |H_engine - H_tape| = %.3e\n", dmax);
        printf("cmp: %ld elements nonzero in the engine but absent from the tape, "
               "largest %.3e, sum %.3e\n", nmissing, dmax_missing, dsum_missing);
        double *wt = malloc((size_t)n * sizeof *wt);
        if (ac_ci_eigvals(n, T, wt))
            return 1;
        for (int k = 0; k < nroots && k < n; k++)
            printf("E%d(tape) = %.15f\n", k + 1, wt[k]);
        free(wt);
        free(T);
        free(has);
        ac_nfte_rec_free(&r);
        ac_nfte_head_free(&h2);
    }
    if (ac_ci_eigvals(n, H, w)) {
        fprintf(stderr, "acci: dsyev failed\n");
        return 1;
    }
    for (int r = 0; r < nroots && r < n; r++)
        printf("E%d = %.15f\n", r + 1, w[r]);
    free(H);
    free(w);
    ac_ci_free(&ci);
    ac_nfte_head_free(&h);
    ac_ints_free(&ints);
    ac_cfp_free(&t);
    return 0;
}
