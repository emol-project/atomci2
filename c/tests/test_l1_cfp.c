/*
 * test_l1_cfp.c -- L1: ac_cfp against the Fortran SETCFP/CFPSET/ISP/IOR/
 * LSQN/XCFP, called in the same process, plus a replay of the XCFP calls
 * recorded in M0.
 *
 * usage: test_l1_cfp CFPTBL [xcfp.txt ...]
 *
 *  1. tables: every field of the five CFPTBL records, C vs /COMCFP/
 *  2. lists:  the shell-state list of every (l, N), C vs /COMWFA/
 *             (TOPADR/NEL/NSHL/WFADR/WF), and WS/WR vs the decoded codes
 *  3. codes:  every valid code: ISP, IQS, ILFO, IOR vs ac_cfp_decode;
 *             LSQN of every block vs its row
 *  4. XCFP:   every pair of states of one shell with N and N-1 electrons,
 *             id = +1 (CFP(f_N, f_N-1, N-1, +1)) and id = -1
 *             (CFP(f_N-1, f_N, N, -1)), as SMAT/CFACRC call it
 *  5. replay: each recorded "im km kmp id value" line
 * Exits 1 on any mismatch. Table fields must agree exactly; XCFP within TOL.
 */
#include "ac_cfp.h"
#include "ftn_decl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-14

static long nbad;

static void bad(const char *fmt, const char *what, long a, long b)
{
    if (nbad < 10) {
        fprintf(stderr, fmt, what, a, b);
        fputc('\n', stderr);
    }
    nbad++;
}

static void eqi(const char *what, long c, long f)
{
    if (c != f)
        bad("%s: C %ld Fortran %ld", what, c, f);
}

static double xmax;
static long xn;

static void eqx(int im, int km, int kmp, int id, double c, double f)
{
    double d = fabs(c - f);
    xn++;
    if (d > xmax)
        xmax = d;
    if (!(d <= TOL * (fabs(f) > 1 ? fabs(f) : 1))) {
        if (nbad < 10)
            fprintf(stderr, "XCFP(%d,%d,%d,%d): C %.17g Fortran %.17g\n", im, km,
                    kmp, id, c, f);
        nbad++;
    }
}

/* the ICONNT link XCFP would follow, so links to ungenerated levels (-1),
   which the Fortran would index out of bounds, are not passed to it */
static int link_of(const ac_cfptab *t, int im, int km)
{
    const int *ri = t->iconnt[im / 256 - 1], *rk = t->iconnt[km / 256 - 1];
    int dq = rk[1] - ri[1], ds = ri[2] - rk[2];
    if (dq == 1 && ds == 1) return rk[5];
    if (dq == 1 && ds == -1) return rk[6];
    if (dq == -1 && ds == 1) return rk[7];
    if (dq == -1 && ds == -1) return rk[8];
    return 0;
}

static double ftn_xcfp(int im, int km, int kmp, int id)
{
    int32_t a = im, b = km, c = kmp, d = id;
    return xcfp_(&a, &b, &c, &d);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s CFPTBL [xcfp.txt ...]\n", argv[0]);
        return 2;
    }
    ac_cfptab t;
    char err[256];
    if (ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "ac_cfp_load: %s\n", err);
        return 2;
    }
    ftnload_(argv[1], strlen(argv[1]));

    /* 1. tables */
    eqi("IBEL", t.ibel, comcfp_.ibel);
    eqi("ICNN", t.icnn, comcfp_.icnn);
    eqi("ILFN", t.ilfn, comcfp_.ilfn);
    eqi("ICMTN", t.icmtn, comcfp_.icmtn);
    eqi("ICDX", t.icdx, comcfp_.icdx);
    for (int i = 0; i < 18; i++) {
        eqi("INAME", t.name[i], comcfp_.iname[i]);
        eqi("IMMINT", t.immint[i], comcfp_.immint[i]);
        eqi("ILSTAD", t.ilstad[i], comcfp_.ilstad[i]);
    }
    for (int j = 0; j < t.icnn; j++)
        for (int i = 0; i < 9; i++)
            eqi("ICONNT", t.iconnt[j][i], comcfp_.iconnt[j][i]);
    for (int j = 0; j < t.icmtn; j++)
        for (int i = 0; i < 5; i++)
            eqi("ICMTDX", t.icmtdx[j][i], comcfp_.icmtdx[j][i]);
    for (int i = 0; i < t.ilfn; i++)
        eqi("ILFUNC", t.ilfunc[i], comcfp_.ilfunc[i]);
    long ncmt_diff = 0;
    for (int i = 0; i < t.icdx; i++)
        if (memcmp(&t.cmttbl[i], &comcfp_.cmttbl[i], sizeof(double)))
            ncmt_diff++;
    eqi("CMTTBL entries differing bitwise", ncmt_diff, 0);
    printf("tables    ibel=%d icnn=%d ilfn=%d icmtn=%d icdx=%d   compared\n",
           t.ibel, t.icnn, t.ilfn, t.icmtn, t.icdx);

    /* 2. lists */
    long nlist = 0, nstate = 0;
    for (int nl = 1; nl <= t.ibel; nl++) {
        int l = nl - 1;
        /* CFPSET builds N = 0..4l+2 for complete shells (IMMINT = 1) and
           N = 0..2nl-IMMINT for truncated ones */
        int nlists = t.immint[l] == 1 ? 4 * l + 3 : 2 * nl - t.immint[l] + 1;
        if (nl < t.ibel)
            eqi("lists per shell", nlists, comwfa_.topadr[nl] - comwfa_.topadr[l]);
        for (int tad = comwfa_.topadr[l]; tad < comwfa_.topadr[l] + nlists; tad++) {
            int ne = comwfa_.nel[tad - 1];
            eqi("NEL", ne, tad - comwfa_.topadr[l]);
            const int *codes;
            int n = ac_cfp_states(&t, l, ne, &codes);
            eqi("list length", n, comwfa_.nshl[tad - 1]);
            for (int s = 0; s < n && s < comwfa_.nshl[tad - 1]; s++) {
                int w = comwfa_.wfadr[tad - 1] + s - 1;
                eqi("list code", codes[s], comwfa_.wf[w]);
                ac_shell_qn q;
                if (ac_cfp_decode(&t, codes[s], &q) == 0) {
                    eqi("WS", q.ts + 1, comwfa_.ws[w]);
                    eqi("WR", q.tL + 1, comwfa_.wr[w]);
                }
                nstate++;
            }
            nlist++;
        }
        /* lists C did not build must be the ones CFPSET did not build */
        for (int ne = nlists; ne <= 4 * l + 2; ne++) {
            const int *codes;
            eqi("unbuilt list", ac_cfp_states(&t, l, ne, &codes), -1);
        }
    }
    printf("lists     %ld (l,N) lists, %ld entries   compared\n", nlist, nstate);

    /* 3. codes */
    long ncode = 0, nlsqn = 0;
    for (int isq = 1; isq <= t.icnn; isq++) {
        for (int lsub = 1; lsub <= t.iconnt[isq - 1][3]; lsub++) {
            int32_t code = 256 * isq + lsub;
            ac_shell_qn q;
            if (ac_cfp_decode(&t, code, &q)) {
                eqi("decode failed for code", code, 0);
                continue;
            }
            eqi("ISP", q.ts + 1, isp_(&code));
            eqi("IQS", q.tq + 1, iqs_(&code));
            eqi("ILFO", q.tl + 1, ilfo_(&code));
            eqi("IOR", q.tL + 1, ior_(&code));
            ncode++;
        }
        const int *r = t.iconnt[isq - 1];
        int32_t L = r[0], IQ = r[1], IS = r[2];
        eqi("LSQN (Fortran) vs row", lsqn_(&L, &IQ, &IS), isq);
        eqi("LSQN (C) vs row", ac_cfp_lsqn(&t, r[0] - 1, r[1] - 1, r[2] - 1), isq);
        nlsqn++;
    }
    printf("codes     %ld codes, %ld blocks   compared\n", ncode, nlsqn);

    /* 4. XCFP over all connected pairs */
    long nskip = 0;
    for (int l = 0; l < t.ibel; l++)
        for (int ne = 1; ne <= 4 * l + 2; ne++) {
            const int *hi, *lo;
            int nh = ac_cfp_states(&t, l, ne, &hi);
            int nlo = ac_cfp_states(&t, l, ne - 1, &lo);
            if (nh <= 0 || nlo <= 0)
                continue;
            for (int a = 0; a < nh; a++)
                for (int b = 0; b < nlo; b++) {
                    if (link_of(&t, hi[a], lo[b]) < 0 || link_of(&t, lo[b], hi[a]) < 0) {
                        nskip++;
                        continue;
                    }
                    eqx(hi[a], lo[b], ne - 1, 1,
                        ac_cfp_xcfp(&t, hi[a], lo[b], ne - 1, 1),
                        ftn_xcfp(hi[a], lo[b], ne - 1, 1));
                    eqx(lo[b], hi[a], ne, -1, ac_cfp_xcfp(&t, lo[b], hi[a], ne, -1),
                        ftn_xcfp(lo[b], hi[a], ne, -1));
                }
        }
    printf("XCFP      %ld compared (%ld pairs skipped: link to an ungenerated level)"
           "   max |C-F| = %.3e\n", xn, nskip, xmax);

    /* 5. replay of recorded calls */
    for (int a = 2; a < argc; a++) {
        FILE *fp = fopen(argv[a], "r");
        if (!fp) {
            perror(argv[a]);
            return 2;
        }
        long n0 = xn;
        int im, km, kmp, id;
        double v;
        while (fscanf(fp, "%d %d %d %d %lf", &im, &km, &kmp, &id, &v) == 5)
            eqx(im, km, kmp, id, ac_cfp_xcfp(&t, im, km, kmp, id), v);
        fclose(fp);
        printf("replay    %ld recorded XCFP calls from %s\n", xn - n0, argv[a]);
    }

    ac_cfp_free(&t);
    printf("L1 %s\n", nbad ? "FAILED" : "passed");
    return nbad ? 1 : 0;
}
