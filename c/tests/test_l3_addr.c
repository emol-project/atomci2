/*
 * test_l3_addr.c -- ac_addr against the Fortran SETIAD/SETKAD/INT1AD/INT2AD,
 * called in the same process.
 *
 * usage: test_l3_addr CFPTBL smat.txt [smat.txt ...]
 *
 * The orbital sequences (nwt, nwth) are the distinct ones found in the M0
 * SMAT recordings. Each is set up four ways: SETIAD and SETKAD, each with
 * NWSQ numbered automatically and with the order inside every l reversed.
 * Compared: LMAX, ISSH, LISLU, the IGCD/ISCD tables, INT1AD for every pair
 * of orbitals of equal l, and INT2AD for every k and every (m1 m2; m3 m4)
 * whose pairs satisfy the triangle and parity rules of R^k.
 */
#include "ac_addr.h"
#include "ac_cfp.h"
#include "ftn_decl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXSEQ 8

static long nbad;

static void eqi(const char *what, long c, long f)
{
    if (c != f) {
        if (nbad < 10)
            fprintf(stderr, "%s: C %ld Fortran %ld\n", what, c, f);
        nbad++;
    }
}

/* skip the rest of an S record after nwth/nwf/nwtc, and its T/E lines */
static int read_seq(FILE *fp, int *nwt, int32_t *nwth)
{
    char tag[4];
    while (fscanf(fp, "%3s", tag) == 1) {
        if (strcmp(tag, "S")) {
            int c;
            while ((c = fgetc(fp)) != '\n' && c != EOF)
                ;
            continue;
        }
        long seq;
        int n, x;
        if (fscanf(fp, "%ld %d", &seq, &n) != 2)
            return -1;
        for (int i = 0; i < 4 * n; i++)
            if (fscanf(fp, "%d", &x) != 1) return -1;
        if (fscanf(fp, "%d", &n) != 1)
            return -1;
        for (int i = 0; i < 4 * n; i++)
            if (fscanf(fp, "%d", &x) != 1) return -1;
        if (fscanf(fp, "%d", nwt) != 1 || *nwt > AC_ADDR_MAXORB)
            return -1;
        for (int i = 0; i < *nwt; i++)
            if (fscanf(fp, "%d", &nwth[i]) != 1) return -1;
        int c;
        while ((c = fgetc(fp)) != '\n' && c != EOF)
            ;
        return 1;
    }
    return 0;
}

static void check_setup(const ac_cfptab *t, int kad, int nwt, const int32_t *nwth,
                        const int32_t *nwlf, const int32_t *nwsq, long *n1, long *n2)
{
    ac_addr a;
    if (ac_addr_setup(&a, t, kad, nwt, nwth, nwlf, nwsq)) {
        eqi("ac_addr_setup failed", 1, 0);
        return;
    }
    int32_t n = nwt, issh[AC_ADDR_MAXL + 1] = {0}, lmax = 0, zero = 0;
    comstr_.lisl = AC_ADDR_LISL;
    if (kad)
        setkad_(&n, nwlf, issh, &zero, nwsq, &lmax);
    else
        setiad_(&n, nwth, issh, &zero, nwsq, &lmax);
    eqi("LMAX", a.lmax, lmax);
    for (int l = 1; l <= a.lmax; l++)
        eqi("ISSH", a.issh[l], issh[l - 1]);
    eqi("LISLU", a.lislu, comstr_.lislu);
    for (int i = 1; i <= a.lislu; i++) {
        eqi("IGCD", a.igcd[i], comstr_.igcd[i - 1]);
        eqi("ISCD", a.iscd[i], comstr_.iscd[i - 1]);
    }

    for (int32_t m1 = 1; m1 <= nwt; m1++)
        for (int32_t m2 = 1; m2 <= nwt; m2++) {
            if (a.nwlf[m1] != a.nwlf[m2])
                continue;
            int32_t f;
            int1ad_(&m1, &m2, &f);
            eqi("INT1AD", ac_addr_one(&a, m1, m2), f);
            ++*n1;
        }

    for (int32_t m1 = 1; m1 <= nwt; m1++)
        for (int32_t m2 = 1; m2 <= nwt; m2++) {
            int l1 = a.nwlf[m1] - 1, l2 = a.nwlf[m2] - 1;
            for (int32_t m3 = 1; m3 <= nwt; m3++)
                for (int32_t m4 = 1; m4 <= nwt; m4++) {
                    int l3 = a.nwlf[m3] - 1, l4 = a.nwlf[m4] - 1;
                    for (int k = abs(l1 - l2); k <= l1 + l2; k += 2) {
                        if (k < abs(l3 - l4) || k > l3 + l4 || (k + l3 + l4) % 2)
                            continue;
                        int32_t kxx = 2 * k + 1, f;
                        int2ad_(&kxx, &m1, &m2, &m3, &m4, &f);
                        int c = ac_addr_two(&a, kxx, m1, m2, m3, m4);
                        if (c != f) {
                            if (nbad < 10)
                                fprintf(stderr, "INT2AD(k=%d, %d %d %d %d) kad=%d: C %d F %d\n",
                                        k, m1, m2, m3, m4, kad, c, f);
                            nbad++;
                        }
                        ++*n2;
                    }
                }
        }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s CFPTBL smat.txt ...\n", argv[0]);
        return 2;
    }
    ac_cfptab t;
    char err[256];
    if (ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "ac_cfp_load: %s\n", err);
        return 2;
    }
    ftnload_(argv[1], strlen(argv[1]));

    static int32_t seqs[MAXSEQ][AC_ADDR_MAXORB];
    int nseq = 0, seqn[MAXSEQ];
    for (int a = 2; a < argc; a++) {
        FILE *fp = fopen(argv[a], "r");
        if (!fp) {
            perror(argv[a]);
            return 2;
        }
        int nwt, rc;
        int32_t nwth[AC_ADDR_MAXORB];
        while ((rc = read_seq(fp, &nwt, nwth)) == 1) {
            int known = 0;
            for (int s = 0; s < nseq && !known; s++)
                known = seqn[s] == nwt && !memcmp(seqs[s], nwth, (size_t)nwt * sizeof *nwth);
            if (!known && nseq < MAXSEQ) {
                seqn[nseq] = nwt;
                memcpy(seqs[nseq], nwth, (size_t)nwt * sizeof *nwth);
                nseq++;
            }
        }
        fclose(fp);
        if (rc < 0) {
            fprintf(stderr, "%s: bad record\n", argv[a]);
            return 2;
        }
    }
    printf("%d distinct orbital sequences\n", nseq);

    long n1 = 0, n2 = 0, nsetup = 0;
    for (int s = 0; s < nseq; s++) {
        int nwt = seqn[s];
        int32_t nwlf[AC_ADDR_MAXORB], auto0[AC_ADDR_MAXORB], rev[AC_ADDR_MAXORB];
        int cnt[AC_ADDR_MAXL + 2] = {0}, tot[AC_ADDR_MAXL + 2] = {0};
        for (int i = 0; i < nwt; i++) {
            nwlf[i] = t.iconnt[seqs[s][i] / 256 - 1][0] / 2 + 1;
            tot[nwlf[i]]++;
            auto0[i] = 0;
        }
        for (int i = 0; i < nwt; i++) /* reverse the order inside each l */
            rev[i] = tot[nwlf[i]] - cnt[nwlf[i]]++;
        for (int kad = 0; kad <= 1; kad++) {
            check_setup(&t, kad, nwt, seqs[s], nwlf, auto0, &n1, &n2);
            check_setup(&t, kad, nwt, seqs[s], nwlf, rev, &n1, &n2);
            nsetup += 2;
        }
        printf("sequence %d: nwt = %d\n", s + 1, nwt);
    }
    printf("%ld setups, %ld INT1AD and %ld INT2AD addresses compared\n", nsetup, n1, n2);
    printf("L3 addr %s\n", nbad ? "FAILED" : "passed");
    ac_cfp_free(&t);
    return nbad ? 1 : 0;
}
