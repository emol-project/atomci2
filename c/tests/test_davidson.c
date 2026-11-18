/*
 * test_davidson.c -- ac_ci_davidson against the full dsyev (ac_ci_eigvecs)
 * on random symmetric matrices shaped like CI matrices (a spread diagonal,
 * weaker couplings):
 *   k lowest eigenvalues within 1e-12 (relative to max(1, |w|)), each
 *   eigenvector |<x|x_ref>| within 1e-11 of 1, the residuals below tol;
 *   a nearly degenerate lowest pair (gap 1e-7): the eigenvalues and the
 *   projector on the pair's span; the matrix is left unchanged.
 * ac_ci_count_below (the eigenvalues below a threshold, from the inertia of
 * L D L^T) against the eigenvalues of dsyev.
 *
 * usage: test_davidson
 */
#include "ac_ci.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nfail;

static void report(const char *name, double dev, double tol)
{
    int ok = dev <= tol;
    printf("%-44s max dev %.3e  %s\n", name, dev, ok ? "ok" : "FAIL");
    nfail += !ok;
}

static unsigned long long rng = 88172645463325252ULL;

static double urand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (double)(rng >> 11) / 9007199254740992.0;
}

/* n x n symmetric, column-major: diagonal 0.01 i + noise, couplings c (2u - 1),
   two lowest diagonal elements gap apart when gap >= 0 */
static double *make(int n, double c, double gap)
{
    double *H = malloc((size_t)n * n * sizeof *H);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++) {
            double v = i == j ? -3.0 + 0.01 * i + 0.005 * urand() : c * (2.0 * urand() - 1.0);
            H[(size_t)j * n + i] = H[(size_t)i * n + j] = v;
        }
    if (gap >= 0.0) { /* rows 0, 1 decoupled from each other, equal diagonal + gap */
        H[0] = -3.0;
        H[(size_t)n + 1] = -3.0 + gap;
        H[n] = H[1] = 0.0;
    }
    return H;
}

static void run(const char *name, int n, int k, double c, double gap)
{
    double *H = make(n, c, gap), *R = malloc((size_t)n * n * sizeof *R);
    double *H0 = malloc((size_t)n * n * sizeof *H0);
    double *wr = malloc((size_t)n * sizeof *wr), *w = malloc((size_t)k * sizeof *w);
    double *x = malloc((size_t)n * k * sizeof *x);
    memcpy(R, H, (size_t)n * n * sizeof *R);
    memcpy(H0, H, (size_t)n * n * sizeof *H0);
    if (ac_ci_eigvecs(n, R, wr)) {
        printf("%s: dsyev failed\n", name);
        nfail++;
        return;
    }
    int it, rc = ac_ci_davidson(n, H, k, 1e-11, w, x, &it);
    char lab[96];
    snprintf(lab, sizeof lab, "%s: converged (%d iterations)", name, it);
    report(lab, rc != 0, 0);
    double de = 0.0, dv = 0.0, dr = 0.0, dh = 0.0;
    for (int j = 0; j < k; j++) {
        de = fmax(de, fabs(w[j] - wr[j]) / fmax(1.0, fabs(wr[j])));
        double r2 = 0.0;
        for (int i = 0; i < n; i++) {
            double s = -w[j] * x[(size_t)j * n + i];
            for (int q = 0; q < n; q++)
                s += H[(size_t)q * n + i] * x[(size_t)j * n + q];
            r2 += s * s;
        }
        dr = fmax(dr, sqrt(r2));
    }
    for (size_t i = 0; i < (size_t)n * n; i++)
        dh = fmax(dh, fabs(H[i] - H0[i]));
    if (gap < 0.0) {
        for (int j = 0; j < k; j++) {
            double o = 0.0;
            for (int i = 0; i < n; i++)
                o += x[(size_t)j * n + i] * R[(size_t)j * n + i];
            dv = fmax(dv, 1.0 - fabs(o));
        }
    } else { /* projector on the two lowest */
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++) {
                double p = 0.0, q = 0.0;
                for (int j = 0; j < 2; j++) {
                    p += x[(size_t)j * n + a] * x[(size_t)j * n + b];
                    q += R[(size_t)j * n + a] * R[(size_t)j * n + b];
                }
                dv = fmax(dv, fabs(p - q));
            }
    }
    snprintf(lab, sizeof lab, "%s: eigenvalues", name);
    report(lab, de, 1e-12);
    snprintf(lab, sizeof lab, gap < 0.0 ? "%s: eigenvectors" : "%s: projector of the pair", name);
    report(lab, dv, gap < 0.0 ? 1e-11 : 1e-9);
    snprintf(lab, sizeof lab, "%s: residuals", name);
    report(lab, dr, 1e-11);
    snprintf(lab, sizeof lab, "%s: H unchanged", name);
    report(lab, dh, 0.0);
    free(H);
    free(H0);
    free(R);
    free(wr);
    free(w);
    free(x);
}

/* ac_ci_count_below against the eigenvalues of dsyev, for thresholds between them */
static void count(const char *name, int n, double c)
{
    double *H = make(n, c, -1.0), *R = malloc((size_t)n * n * sizeof *R);
    double *wr = malloc((size_t)n * sizeof *wr);
    memcpy(R, H, (size_t)n * n * sizeof *R);
    int bad = ac_ci_eigvals(n, R, wr) != 0;
    static const int at[] = {0, 1, 2, 7, 100, 333};
    for (int t = 0; t <= 6 && !bad; t++) {
        int want = t < 6 ? at[t] : n, got = -1;
        if (want > n)
            continue;
        double sigma = want == 0 ? wr[0] - 1.0 : want == n ? wr[n - 1] + 1.0
                                                           : 0.5 * (wr[want - 1] + wr[want]);
        memcpy(R, H, (size_t)n * n * sizeof *R);
        bad = ac_ci_count_below(n, R, sigma, &got) != 0 || got != want;
    }
    char lab[96];
    snprintf(lab, sizeof lab, "%s: eigenvalues below a threshold counted", name);
    report(lab, bad, 0);
    free(H);
    free(R);
    free(wr);
}

int main(void)
{
    run("n 600, k 1", 600, 1, 0.02, -1.0);
    run("n 600, k 4", 600, 4, 0.02, -1.0);
    run("n 1200, k 3, strong coupling", 1200, 3, 0.1, -1.0);
    run("n 800, k 2, gap 1e-7", 800, 2, 0.02, 1e-7);
    run("n 150 (below the start block), k 2", 150, 2, 0.05, -1.0);
    count("n 600", 600, 0.02);
    count("n 500, strong coupling", 500, 0.3);
    count("n 3", 3, 0.5);
    printf("%s\n", nfail ? "test_davidson: FAILED" : "test_davidson: all ok");
    return nfail != 0;
}
