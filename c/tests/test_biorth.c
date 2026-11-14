/*
 * test_biorth.c -- algebraic checks of ac_biorth (S2-M9), no radial
 * functions involved:
 *   lu      <A|B> = 1 for random overlaps (square and not, three pivot
 *           splittings), t = c^-1, upper triangular
 *   eij     <i|E_ab|j> = <j|E_ba|i>, sum_k E_kk = N, [E_ab, E_bc] = E_ac
 *           on a full CI space
 *   1e      one electron: c^A = t c^X
 *   s2      two s electrons, singlet: the spatial coefficients M^A = t M t^T
 *   group   three electrons, full CI: t1 then t2 equals t2 t1; t then t^-1
 *           is the identity
 *   closure a full CI is closed; removing a configuration or one CSF of a
 *           configuration breaks it
 *
 * usage: test_biorth CFPTBL
 */
#include "ac_biorth.h"
#include "ac_cfp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nfail;

static void report(const char *name, double dev, double tol)
{
    int ok = dev <= tol;
    printf("%-34s max dev %.3e  %s\n", name, dev, ok ? "ok" : "FAIL");
    nfail += !ok;
}

static unsigned long long rng = 88172645463325252ULL;

static double urand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (double)(rng >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

/* every occupation of nwt orbitals (l given) with nel electrons */
static int all_configs(int nwt, const int *l, int nel, int *occ, int max)
{
    int n = 0, cur[16] = {0};
    for (;;) {
        int s = 0;
        for (int i = 0; i < nwt; i++)
            s += cur[i];
        if (s == nel && n < max)
            memcpy(occ + (size_t)n++ * nwt, cur, (size_t)nwt * sizeof *cur);
        int i = 0;
        while (i < nwt && ++cur[i] > 4 * l[i] + 2)
            cur[i++] = 0;
        if (i == nwt)
            return n;
    }
}

static void test_lu(void)
{
    const int nm[][2] = {{3, 3}, {2, 4}, {4, 2}, {0, 2}, {3, 0}, {1, 1}, {5, 5}};
    double dev = 0.0;
    for (size_t c = 0; c < sizeof nm / sizeof *nm; c++)
        for (int mode = 0; mode < 3; mode++) {
            int n = nm[c][0], m = nm[c][1];
            double S[25], cx[25], cy[25], tx[25], ty[25];
            for (int p = 0; p < n; p++)
                for (int q = 0; q < m; q++)
                    S[p * m + q] = (p == q ? 1.0 : 0.0) + 0.3 * urand();
            char err[128];
            if (ac_biorth_lu(n, m, S, mode, 1e-8, cx, cy, tx, ty, err, sizeof err)) {
                printf("lu: %s\n", err);
                nfail++;
                continue;
            }
            int mn = n < m ? n : m;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) {
                    double s = 0.0;
                    for (int p = 0; p < n; p++)
                        for (int q = 0; q < m; q++)
                            s += cx[p * n + i] * S[p * m + q] * cy[q * m + j];
                    double want = i == j && i < mn ? 1.0 : 0.0;
                    dev = fmax(dev, fabs(s - want));
                }
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double s = 0.0;
                    for (int k = 0; k < n; k++)
                        s += tx[i * n + k] * cx[k * n + j];
                    dev = fmax(dev, fabs(s - (i == j)));
                    if (i > j)
                        dev = fmax(dev, fabs(cx[i * n + j]) + fabs(tx[i * n + j]));
                }
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++) {
                    double s = 0.0;
                    for (int k = 0; k < m; k++)
                        s += ty[i * m + k] * cy[k * m + j];
                    dev = fmax(dev, fabs(s - (i == j)));
                    if (i > j)
                        dev = fmax(dev, fabs(cy[i * m + j]) + fabs(ty[i * m + j]));
                }
        }
    report("lu: biorthonormal, t = c^-1", dev, 1e-13);
}

/* a random upper-triangular t over sequence orbitals (zero across l) */
static void rand_t(const ac_ci *ci, double *t)
{
    int w = ci->nwt + 1;
    memset(t, 0, (size_t)w * w * sizeof *t);
    for (int a = 1; a <= ci->nwt; a++)
        for (int k = a; k <= ci->nwt; k++)
            if (ci->lorb[a] == ci->lorb[k])
                t[a * w + k] = a == k ? 1.0 + 0.4 * urand() : 0.4 * urand();
}

static void matmul_t(int w, const double *x, const double *y, double *z)
{
    for (int i = 0; i < w; i++)
        for (int j = 0; j < w; j++) {
            double s = 0.0;
            for (int k = 0; k < w; k++)
                s += x[i * w + k] * y[k * w + j];
            z[i * w + j] = s;
        }
}

/* inverse of the upper-triangular t (1-based block) */
static void inv_t(int w, const double *t, double *ti)
{
    memset(ti, 0, (size_t)w * w * sizeof *ti);
    for (int c = 1; c < w; c++)
        for (int i = c; i >= 1; i--) {
            double s = i == c ? 1.0 : 0.0;
            for (int k = i + 1; k <= c; k++)
                s -= t[i * w + k] * ti[k * w + c];
            ti[i * w + c] = s / t[i * w + i];
        }
}

static int build(ac_ci *ci, const ac_cfptab *tab, int nwt, const int *l, int nel, int *ncfg)
{
    static int occ[4096 * 8];
    int n = all_configs(nwt, l, nel, occ, 4096);
    if (ncfg)
        *ncfg = n;
    char err[256];
    if (ac_ci_from_orbitals(ci, tab, nwt, l, n, occ, err, sizeof err)) {
        printf("ci: %s\n", err);
        return -1;
    }
    return 0;
}

static void test_eij(const ac_cfptab *tab)
{
    const int l[] = {0, 0, 0, 1, 1};
    int nwt = 5, nel = 3;
    ac_ci ci;
    if (build(&ci, tab, nwt, l, nel, NULL)) {
        nfail++;
        return;
    }
    int n = ci.ncsf, e = 0;
    double dsym = 0.0, dn = 0.0, dcom = 0.0;
    double *E = malloc((size_t)(nwt + 1) * (nwt + 1) * n * n * sizeof *E);
#define EL(a, b, i, j) E[((((size_t)(a) * (nwt + 1) + (b)) * n) + (i)) * n + (j)]
    for (int a = 1; a <= nwt; a++)
        for (int b = 1; b <= nwt; b++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    EL(a, b, i, j) = l[a - 1] == l[b - 1] ? ac_biorth_eij(&ci, i, j, a, b, &e) : 0.0;
    for (int a = 1; a <= nwt; a++)
        for (int b = 1; b <= nwt; b++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    dsym = fmax(dsym, fabs(EL(a, b, i, j) - EL(b, a, j, i)));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 1; k <= nwt; k++)
                s += EL(k, k, i, j);
            dn = fmax(dn, fabs(s - (i == j ? nel : 0)));
        }
    for (int a = 1; a <= nwt; a++)
        for (int b = 1; b <= nwt; b++)
            for (int c = 1; c <= nwt; c++) {
                if (l[a - 1] != l[b - 1] || l[b - 1] != l[c - 1] || a == c)
                    continue;
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        double s = 0.0;
                        for (int m = 0; m < n; m++)
                            s += EL(a, b, i, m) * EL(b, c, m, j) - EL(b, c, i, m) * EL(a, b, m, j);
                        dcom = fmax(dcom, fabs(s - EL(a, c, i, j)));
                    }
            }
#undef EL
    free(E);
    if (e)
        nfail++;
    report("eij: <i|E_ab|j> = <j|E_ba|i>", dsym, 1e-13);
    report("eij: sum_k E_kk = N", dn, 1e-13);
    report("eij: [E_ab, E_bc] = E_ac", dcom, 1e-12);
    ac_ci_free(&ci);
}

static void test_1e(const ac_cfptab *tab)
{
    const int l[] = {0, 1, 0, 1, 0, 1};
    int nwt = 6;
    ac_ci ci;
    if (build(&ci, tab, nwt, l, 1, NULL)) {
        nfail++;
        return;
    }
    ac_biorth_ops o;
    char err[256];
    if (ac_biorth_ops_build(&o, &ci, err, sizeof err)) {
        printf("1e: %s\n", err);
        nfail++;
        return;
    }
    int w = nwt + 1, n = ci.ncsf;
    double t[49], c[16], cx[16], dev = 0.0;
    for (int trial = 0; trial < 5; trial++) {
        rand_t(&ci, t);
        for (int i = 0; i < n; i++)
            cx[i] = c[i] = urand();
        if (ac_biorth_transform(&o, &ci, t, 1, c, err, sizeof err)) {
            printf("1e: %s\n", err);
            nfail++;
            break;
        }
        /* CSF i is one electron in orbital orb(i), with phase 1 */
        for (int i = 0; i < n; i++) {
            int oi = 0;
            for (int k = 1; k <= nwt; k++)
                if (o.occ[(size_t)i * w + k])
                    oi = k;
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                int oj = 0;
                for (int k = 1; k <= nwt; k++)
                    if (o.occ[(size_t)j * w + k])
                        oj = k;
                if (l[oi - 1] == l[oj - 1])
                    s += t[oi * w + oj] * cx[j];
            }
            dev = fmax(dev, fabs(c[i] - s));
        }
    }
    report("1e: c^A = t c^X", dev, 1e-14);
    ac_biorth_ops_free(&o);
    ac_ci_free(&ci);
}

static void test_s2(const ac_cfptab *tab)
{
    const int l[] = {0, 0, 0};
    int nwt = 3, w = 4;
    ac_ci ci;
    if (build(&ci, tab, nwt, l, 2, NULL)) {
        nfail++;
        return;
    }
    ac_biorth_ops o;
    char err[256];
    if (ac_biorth_ops_build(&o, &ci, err, sizeof err)) {
        printf("s2: %s\n", err);
        nfail++;
        return;
    }
    /* singlets: s_a^2 -> M_aa = c; s_a s_b -> M_ab = M_ba = sigma c / sqrt2 */
    int n = ci.ncsf, pa[16], pb[16], singlet[16];
    for (int i = 0; i < n; i++) {
        int k1 = 0, k2 = 0;
        for (int k = 1; k <= nwt; k++) {
            int q = o.occ[(size_t)i * w + k];
            if (q == 2)
                k1 = k2 = k;
            else if (q == 1)
                (k1 ? (k2 = k) : (k1 = k));
        }
        pa[i] = k1;
        pb[i] = k2;
        const ac_csf *f = &ci.csf[i].c;
        singlet[i] = f->cpl[f->nsh].ts == 0;
    }
    /* sigma from <s_a s_b|E_ab|s_b^2> = sqrt2 sigma */
    double sigma[16];
    int e = 0;
    for (int i = 0; i < n; i++) {
        sigma[i] = 1.0;
        if (!singlet[i] || pa[i] == pb[i])
            continue;
        for (int j = 0; j < n; j++)
            if (singlet[j] && pa[j] == pb[i] && pb[j] == pb[i])
                sigma[i] = ac_biorth_eij(&ci, i, j, pa[i], pb[i], &e) / sqrt(2.0);
    }
    double dev = 0.0, dsig = 0.0;
    for (int i = 0; i < n; i++)
        if (singlet[i] && pa[i] != pb[i])
            dsig = fmax(dsig, fabs(fabs(sigma[i]) - 1.0));
    for (int trial = 0; trial < 5; trial++) {
        double t[16], c[16], M[16] = {0}, MA[16] = {0};
        rand_t(&ci, t);
        for (int i = 0; i < n; i++)
            c[i] = singlet[i] ? urand() : 0.0;
        for (int i = 0; i < n; i++) {
            if (!singlet[i])
                continue;
            int a = pa[i], b = pb[i];
            if (a == b)
                M[a * w + a] += c[i];
            else {
                M[a * w + b] += sigma[i] * c[i] / sqrt(2.0);
                M[b * w + a] += sigma[i] * c[i] / sqrt(2.0);
            }
        }
        for (int a = 1; a <= nwt; a++)
            for (int b = 1; b <= nwt; b++)
                for (int p = 1; p <= nwt; p++)
                    for (int q = 1; q <= nwt; q++)
                        MA[a * w + b] += t[a * w + p] * M[p * w + q] * t[b * w + q];
        if (ac_biorth_transform(&o, &ci, t, 1, c, err, sizeof err)) {
            printf("s2: %s\n", err);
            nfail++;
            break;
        }
        for (int i = 0; i < n; i++) {
            if (!singlet[i])
                continue;
            int a = pa[i], b = pb[i];
            double want = a == b ? MA[a * w + a] : sigma[i] * sqrt(2.0) * MA[a * w + b];
            dev = fmax(dev, fabs(c[i] - want));
        }
    }
    report("s2: CSF phases |sigma| = 1", dsig, 1e-14);
    report("s2: M^A = t M t^T", dev, 1e-14);
    ac_biorth_ops_free(&o);
    ac_ci_free(&ci);
}

static void test_group(const ac_cfptab *tab, const int *l, int nwt, int nel, const char *name)
{
    ac_ci ci;
    if (build(&ci, tab, nwt, l, nel, NULL)) {
        nfail++;
        return;
    }
    ac_biorth_ops o;
    char err[256];
    if (ac_biorth_ops_build(&o, &ci, err, sizeof err)) {
        printf("group: %s\n", err);
        nfail++;
        return;
    }
    int n = ci.ncsf, w = nwt + 1;
    double *t1 = malloc((size_t)w * w * sizeof *t1), *t2 = malloc((size_t)w * w * sizeof *t2);
    double *t12 = malloc((size_t)w * w * sizeof *t12), *ti = malloc((size_t)w * w * sizeof *ti);
    double *c0 = malloc((size_t)n * sizeof *c0), *ca = malloc((size_t)n * sizeof *ca);
    double *cb = malloc((size_t)n * sizeof *cb);
    double dev = 0.0, dinv = 0.0;
    for (int trial = 0; trial < 3; trial++) {
        rand_t(&ci, t1);
        rand_t(&ci, t2);
        matmul_t(w, t2, t1, t12);
        inv_t(w, t1, ti);
        for (int i = 0; i < n; i++)
            c0[i] = ca[i] = cb[i] = urand();
        if (ac_biorth_transform(&o, &ci, t1, 1, ca, err, sizeof err) ||
            ac_biorth_transform(&o, &ci, t2, 1, ca, err, sizeof err) ||
            ac_biorth_transform(&o, &ci, t12, 1, cb, err, sizeof err)) {
            printf("group: %s\n", err);
            nfail++;
            break;
        }
        for (int i = 0; i < n; i++)
            dev = fmax(dev, fabs(ca[i] - cb[i]));
        memcpy(ca, c0, (size_t)n * sizeof *ca);
        ac_biorth_transform(&o, &ci, t1, 1, ca, err, sizeof err);
        ac_biorth_transform(&o, &ci, ti, 1, ca, err, sizeof err);
        for (int i = 0; i < n; i++)
            dinv = fmax(dinv, fabs(ca[i] - c0[i]));
    }
    char lab[64];
    snprintf(lab, sizeof lab, "group %s: t2 after t1 = t2 t1", name);
    report(lab, dev, 1e-12);
    snprintf(lab, sizeof lab, "group %s: t^-1 after t = 1", name);
    report(lab, dinv, 1e-12);
    printf("    (%d CSFs, %d elements of E_ak)\n", n, o.n);
    free(t1);
    free(t2);
    free(t12);
    free(ti);
    free(c0);
    free(ca);
    free(cb);
    ac_biorth_ops_free(&o);
    ac_ci_free(&ci);
}

static void test_closure(const ac_cfptab *tab)
{
    const int l[] = {0, 0, 0, 1, 1};
    int nwt = 5, nel = 3;
    ac_ci ci;
    if (build(&ci, tab, nwt, l, nel, NULL)) {
        nfail++;
        return;
    }
    char err[256], what[160];
    double dev;
    if (ac_biorth_closure(&ci, &dev, what, sizeof what, err, sizeof err)) {
        printf("closure: %s\n", err);
        nfail++;
        return;
    }
    report("closure: full CI", dev, 1e-24);
    /* drop the configuration 1s 2s 3s, which 2s 3s^2 reaches by 3s -> 1s */
    int n = ci.ncsf, keep = 0, w = nwt + 1, removed = -1;
    unsigned char occ[8];
    ac_ci_csf *all = ci.csf;
    ac_ci_csf *sub = calloc((size_t)n, sizeof *sub);
    for (int i = 0; i < n; i++) {
        memset(occ, 0, sizeof occ);
        for (int s = 1; s <= all[i].c.nsh; s++)
            occ[all[i].seq[s]] = (unsigned char)(occ[all[i].seq[s]] + all[i].c.nel[s]);
        int is123 = occ[1] == 1 && occ[2] == 1 && occ[3] == 1;
        if (!is123)
            sub[keep++] = all[i];
    }
    (void)w;
    ci.csf = sub;
    ci.ncsf = keep;
    ac_biorth_closure(&ci, &dev, what, sizeof what, err, sizeof err);
    printf("    without 1s 2s 3s: %s\n", what);
    report("closure: missing configuration", dev > 0.1 ? 0.0 : 1.0, 0.5);
    /* drop one doublet of 1s 2s 3s */
    keep = 0;
    for (int i = 0; i < n; i++) {
        memset(occ, 0, sizeof occ);
        for (int s = 1; s <= all[i].c.nsh; s++)
            occ[all[i].seq[s]] = (unsigned char)(occ[all[i].seq[s]] + all[i].c.nel[s]);
        int is123 = occ[1] == 1 && occ[2] == 1 && occ[3] == 1;
        const ac_csf *f = &all[i].c;
        if (is123 && f->cpl[f->nsh].ts == 1 && removed < 0) {
            removed = i;
            continue;
        }
        sub[keep++] = all[i];
    }
    ci.ncsf = keep;
    ac_biorth_closure(&ci, &dev, what, sizeof what, err, sizeof err);
    printf("    without CSF %d: %s\n", removed, what);
    report("closure: missing CSF", dev > 1e-3 ? 0.0 : 1.0, 0.5);
    ci.csf = all;
    ci.ncsf = n;
    free(sub);
    ac_ci_free(&ci);
}

int main(int argc, char **argv)
{
    ac_cfptab t;
    char err[256];
    if (argc != 2 || ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "usage: %s CFPTBL\n", argv[0]);
        return 2;
    }
    test_lu();
    test_eij(&t);
    test_1e(&t);
    test_s2(&t);
    const int la[] = {0, 0, 0, 1, 1};
    test_group(&t, la, 5, 3, "s3p2 N=3");
    const int lb[] = {0, 0, 1, 1, 2, 2};
    test_group(&t, lb, 6, 3, "s2p2d2 N=3");
    const int lc[] = {1, 1, 1};
    test_group(&t, lc, 3, 4, "p3 N=4");
    test_closure(&t);
    ac_cfp_free(&t);
    printf(nfail ? "test_biorth: %d FAILED\n" : "test_biorth: all passed\n", nfail);
    return nfail != 0;
}
