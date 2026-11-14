/*
 * test_l3_smat.c -- L3: replay every SMAT call recorded in M0 through ac_smat
 * and compare the terms and outputs with the Fortran's.
 *
 * usage: test_l3_smat CFPTBL smat.txt [smat.txt ...]
 *
 * Record format (c/tests/trace/trace_wrap.c):
 *   S seq nws nwen[nws] nweq[nws] nwesl[nws] nwrps[nws]
 *         nvs nven[nvs] nveq[nvs] nvesl[nvs] nvrps[nvs] nwt nwth[nwt] nwf nwtc[nwf]
 *   T km i1 i2 i3 i4 noe wt      (one per ISQT inside the call)
 *   E noe n1oe nhtr nfsr nhdtx nwst nvst
 * A call passes when the seven outputs agree exactly and the terms agree in
 * number, order, KM and slots exactly and in WT within TOL (relative to
 * max(1, |WT|)).
 */
#include "ac_cfp.h"
#include "ac_smat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-12
#define MAXT 4096

static int rd_i16(FILE *fp, int16_t *v, int n)
{
    for (int i = 0; i < n; i++) {
        int x;
        if (fscanf(fp, "%d", &x) != 1)
            return -1;
        v[i] = (int16_t)x;
    }
    return 0;
}

static int rd_i32(FILE *fp, int32_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (fscanf(fp, "%d", &v[i]) != 1)
            return -1;
    return 0;
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

    long ncall = 0, nterm = 0, nbad = 0, shown = 0;
    double maxd = 0.0;
    static ac_term ft[MAXT];
    for (int a = 2; a < argc; a++) {
        FILE *fp = fopen(argv[a], "r");
        if (!fp) {
            perror(argv[a]);
            return 2;
        }
        long ncall_file = 0;
        char tag[4];
        while (fscanf(fp, "%3s", tag) == 1) {
            if (strcmp(tag, "S") != 0) {
                fprintf(stderr, "%s: expected S, got %s\n", argv[a], tag);
                return 2;
            }
            long seq;
            int nws, nvs, nwt, nwf;
            int16_t wen[32], weq[32], wesl[32], wrps[32];
            int16_t ven[32], veq[32], vesl[32], vrps[32];
            int32_t nwth[256], nwtc[64];
            if (fscanf(fp, "%ld %d", &seq, &nws) != 2 || nws > 32 ||
                rd_i16(fp, wen, nws) || rd_i16(fp, weq, nws) ||
                rd_i16(fp, wesl, nws) || rd_i16(fp, wrps, nws) ||
                fscanf(fp, "%d", &nvs) != 1 || nvs > 32 || rd_i16(fp, ven, nvs) ||
                rd_i16(fp, veq, nvs) || rd_i16(fp, vesl, nvs) ||
                rd_i16(fp, vrps, nvs) || fscanf(fp, "%d", &nwt) != 1 ||
                nwt > 256 || rd_i32(fp, nwth, nwt) || fscanf(fp, "%d", &nwf) != 1 ||
                nwf > 64 || rd_i32(fp, nwtc, nwf)) {
                fprintf(stderr, "%s: bad S record near seq %ld\n", argv[a], seq);
                return 2;
            }
            /* Fortran terms, then the E line */
            int nft = 0, fo[7];
            for (;;) {
                if (fscanf(fp, "%3s", tag) != 1) {
                    fprintf(stderr, "%s: truncated at seq %ld\n", argv[a], seq);
                    return 2;
                }
                if (!strcmp(tag, "T")) {
                    int noe;
                    ac_term *x = &ft[nft < MAXT ? nft : MAXT - 1];
                    if (fscanf(fp, "%d %d %d %d %d %d %lf", &x->km, &x->i1, &x->i2,
                               &x->i3, &x->i4, &noe, &x->wt) != 7)
                        return 2;
                    if (noe != nft + 1)
                        fprintf(stderr, "seq %ld: Fortran NOE %d out of order\n", seq, noe);
                    nft++;
                } else if (!strcmp(tag, "E")) {
                    if (rd_i32(fp, (int32_t *)fo, 7))
                        return 2;
                    break;
                } else {
                    fprintf(stderr, "%s: unexpected %s at seq %ld\n", argv[a], tag, seq);
                    return 2;
                }
            }

            ac_terms ct;
            ac_terms_init(&ct);
            ac_smat_out o;
            int rc = ac_smat(&t, nws, wen, weq, wesl, wrps, nvs, ven, veq, vesl, vrps,
                             nwt, nwth, nwf, nwtc, &ct, &o);
            int co[7] = {o.noe, o.n1oe, o.nhtr, o.nfsr, o.nhdtx, o.nwst, o.nvst};
            int bad = rc != 0 || memcmp(co, fo, sizeof co) != 0 || ct.len != nft;
            for (int i = 0; !bad && i < nft; i++) {
                const ac_term *c = &ct.v[i], *f = &ft[i];
                double d = fabs(c->wt - f->wt);
                double lim = TOL * (fabs(f->wt) > 1 ? fabs(f->wt) : 1);
                if (d > maxd)
                    maxd = d;
                if (c->km != f->km || c->i1 != f->i1 || c->i2 != f->i2 ||
                    c->i3 != f->i3 || c->i4 != f->i4 || d > lim)
                    bad = 1;
            }
            if (bad) {
                nbad++;
                if (shown++ < 8) {
                    fprintf(stderr, "seq %ld of %s: rc=%d outputs C", seq, argv[a], rc);
                    for (int i = 0; i < 7; i++)
                        fprintf(stderr, " %d", co[i]);
                    fprintf(stderr, " / F");
                    for (int i = 0; i < 7; i++)
                        fprintf(stderr, " %d", fo[i]);
                    fprintf(stderr, "; terms C %d F %d\n", ct.len, nft);
                    int n = ct.len > nft ? ct.len : nft;
                    for (int i = 0; i < n && i < 40; i++) {
                        if (i < ct.len)
                            fprintf(stderr, "   C %3d: %3d %3d %3d %3d %3d %.15g", i + 1,
                                    ct.v[i].km, ct.v[i].i1, ct.v[i].i2, ct.v[i].i3,
                                    ct.v[i].i4, ct.v[i].wt);
                        else
                            fprintf(stderr, "   C    : -");
                        if (i < nft)
                            fprintf(stderr, "   | F: %3d %3d %3d %3d %3d %.15g\n", ft[i].km,
                                    ft[i].i1, ft[i].i2, ft[i].i3, ft[i].i4, ft[i].wt);
                        else
                            fprintf(stderr, "   | F: -\n");
                    }
                }
            }
            nterm += nft;
            ncall_file++;
            ac_terms_free(&ct);
        }
        fclose(fp);
        ncall += ncall_file;
        printf("replayed %ld SMAT calls from %s\n", ncall_file, argv[a]);
    }
    printf("%ld calls, %ld Fortran terms, max |WT_C - WT_F| = %.3e, %ld calls differing\n",
           ncall, nterm, maxd, nbad);
    printf("L3 %s\n", nbad ? "FAILED" : "passed");
    ac_cfp_free(&t);
    return nbad ? 1 : 0;
}
