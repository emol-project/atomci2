/*
 * test_l2_shms.c -- L2: recompute every SHMS call recorded in M0 with
 * ac_factor_shms and compare with the Fortran's W.
 *
 * usage: test_l2_shms CFPTBL shms.txt [shms.txt ...]
 *
 * Each record (c/tests/trace/trace_wrap.c) holds SHMS's complete input and
 * its W. Both the plain product and the normal-ordered product are computed;
 * a record passes when the Fortran W equals the normal-ordered value within
 * TOL (the Hamiltonian, METECC eq. 45, is normal ordered). The summary is
 * grouped by the order pattern of the operator shells (C1, A1, C2, A2),
 * the cases SHMSRC distinguishes, and says for each pattern how many records
 * matched the plain and the normal-ordered value.
 */
#include "ac_cfp.h"
#include "ac_factor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-12
#define MAXPAT 64

typedef struct {
    char name[40];
    long n, plain_ok, normal_ok, both_ok, err;
    double maxdiff_normal;
} pattern;

static pattern pats[MAXPAT];
static int npat;

static pattern *pat_of(const char *name)
{
    for (int i = 0; i < npat; i++)
        if (!strcmp(pats[i].name, name))
            return &pats[i];
    if (npat == MAXPAT) {
        fprintf(stderr, "too many patterns\n");
        exit(2);
    }
    pattern *p = &pats[npat++];
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", name);
    return p;
}

/* e.g. "C1<A1=C2<A2": the operators sorted by shell, '=' for equal shells */
static void pattern_name(char *out, size_t n, int two_body, int c1, int a1,
                         int c2, int a2)
{
    const char *lab[4] = {"C1", "A1", "C2", "A2"};
    int v[4] = {c1, a1, c2, a2}, ord[4] = {0, 1, 2, 3};
    int m = two_body ? 4 : 2;
    for (int i = 1; i < m; i++) /* stable insertion sort by shell */
        for (int j = i; j > 0 && v[ord[j - 1]] > v[ord[j]]; j--) {
            int tmp = ord[j];
            ord[j] = ord[j - 1];
            ord[j - 1] = tmp;
        }
    size_t k = (size_t)snprintf(out, n, "%s", two_body ? "" : "1el ");
    for (int i = 0; i < m && k < n; i++) {
        if (i)
            k += (size_t)snprintf(out + k, n - k, "%s",
                                  v[ord[i - 1]] == v[ord[i]] ? "=" : "<");
        k += (size_t)snprintf(out + k, n - k, "%s", lab[ord[i]]);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s CFPTBL shms.txt ...\n", argv[0]);
        return 2;
    }
    ac_cfptab t;
    char err[256];
    if (ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "ac_cfp_load: %s\n", err);
        return 2;
    }

    long total = 0, fails = 0, shown = 0;
    for (int a = 2; a < argc; a++) {
        FILE *fp = fopen(argv[a], "r");
        if (!fp) {
            perror(argv[a]);
            return 2;
        }
        long nfile = 0;
        int32_t arr[8][12];
        for (;;) {
            int ok = 1;
            for (int i = 0; i < 8 && ok; i++)
                for (int j = 0; j < 12 && ok; j++)
                    ok = fscanf(fp, "%d", &arr[i][j]) == 1;
            int s[12];
            for (int j = 0; j < 12 && ok; j++)
                ok = fscanf(fp, "%d", &s[j]) == 1;
            double w;
            if (!ok || fscanf(fp, "%lf", &w) != 1)
                break;
            /* s: idummy c1 a1 lsi lri c2 a2 lsf lrf lst lrt nht */
            int c1 = s[1], a1 = s[2], lsi = s[3], lri = s[4], c2 = s[5], a2 = s[6];
            int lsf = s[7], lrf = s[8], lst = s[9], lrt = s[10], nht = s[11];
            int two = c2 <= nht;
            char name[40];
            pattern_name(name, sizeof name, two, c1, a1, c2, a2);
            pattern *p = pat_of(name);
            p->n++;
            nfile++;

            int e1, e2;
            double vp = ac_factor_shms(&t, nht, arr[0], arr[2], arr[3], arr[4],
                                       arr[5], arr[6], arr[7], c1, a1, lsi, lri,
                                       c2, a2, lsf, lrf, lst, lrt, 0, &e1);
            double vn = ac_factor_shms(&t, nht, arr[0], arr[2], arr[3], arr[4],
                                       arr[5], arr[6], arr[7], c1, a1, lsi, lri,
                                       c2, a2, lsf, lrf, lst, lrt, 1, &e2);
            if (e1 || e2) {
                p->err++;
                fails++;
                if (shown++ < 10)
                    fprintf(stderr, "error %d/%d in %s, record %ld of %s\n", e1, e2,
                            name, nfile, argv[a]);
                continue;
            }
            double dp = fabs(vp - w), dn = fabs(vn - w);
            int okp = dp <= TOL, okn = dn <= TOL;
            p->plain_ok += okp;
            p->normal_ok += okn;
            p->both_ok += okp && okn;
            if (dn > p->maxdiff_normal)
                p->maxdiff_normal = dn;
            if (!okn) {
                fails++;
                if (shown++ < 10)
                    fprintf(stderr,
                            "%s record %ld of %s: Fortran %.15g plain %.15g "
                            "normal %.15g (lsi lri lsf lrf = %d %d %d %d)\n",
                            name, nfile, argv[a], w, vp, vn, lsi, lri, lsf, lrf);
            }
        }
        fclose(fp);
        total += nfile;
        printf("read %ld records from %s\n", nfile, argv[a]);
    }

    printf("\n%-18s %8s %9s %9s %9s %6s %12s\n", "pattern", "records", "plain=W",
           "normal=W", "both", "error", "max|n-W|");
    for (int i = 0; i < npat; i++)
        printf("%-18s %8ld %9ld %9ld %9ld %6ld %12.3e\n", pats[i].name, pats[i].n,
               pats[i].plain_ok, pats[i].normal_ok, pats[i].both_ok, pats[i].err,
               pats[i].maxdiff_normal);
    printf("\n%ld records, %ld not matching the normal-ordered value\n", total, fails);
    printf("L2 %s\n", fails ? "FAILED" : "passed");
    ac_cfp_free(&t);
    return fails ? 1 : 0;
}
