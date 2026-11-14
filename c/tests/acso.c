/*
 * acso.c -- LSJ levels of Coulomb H + H_SO1 for given integrals (stage 2,
 * S2-M4; driven by c/python/test_s2m4.py).
 *
 * usage: acso CFPTBL < spec
 *
 * spec (orbitals 1-based):
 *   orbitals l1 l2 ...
 *   config n1 n2 ...            (one line per configuration)
 *   h a b value                 h(a,b) = h(b,a)
 *   r k a b c d value           R^k(ab;cd), a, b on electron 1; every
 *                               ordering that is needed must be given
 *   z a b value                 zeta(a,b) = zeta(b,a)
 *   t a b value                 kinetic T(a,b), for Delta g_RMC
 *   u a b value                 Z <a|1/r|b>, for Delta g_GC
 *   end
 * output:
 *   csf i tS tL                 the CSFs (i from 0)
 *   asym tJ value               max |H - H^T| of the J block
 *   diag tJ i value             its diagonal element of CSF i
 *   level tJ value              eigenvalues, ascending, per 2J
 *   g tJ E gL gS rmc gc ormc    its Zeeman factors (2J > 0; ac_lsj_gfactor)
 */
#include "ac_cfp.h"
#include "ac_ci.h"
#include "ac_lsj.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXO 8
#define MAXK 9
#define MAXC 64

static double hv[MAXO + 1][MAXO + 1], zv[MAXO + 1][MAXO + 1];
static double tv[MAXO + 1][MAXO + 1], uv[MAXO + 1][MAXO + 1];

static double get_t(void *ctx, int a, int b)
{
    (void)ctx;
    return tv[a][b];
}

static double get_u(void *ctx, int a, int b)
{
    (void)ctx;
    return uv[a][b];
}
static double rv[MAXK][MAXO + 1][MAXO + 1][MAXO + 1][MAXO + 1];

static double get_h(void *ctx, int a, int b)
{
    (void)ctx;
    return hv[a][b];
}

static double get_rk(void *ctx, int k, int a, int b, int c, int d)
{
    (void)ctx;
    return k < MAXK ? rv[k][a][b][c][d] : 0.0;
}

static double get_zeta(void *ctx, int a, int b)
{
    (void)ctx;
    return zv[a][b];
}

int main(int argc, char **argv)
{
    ac_cfptab t;
    char err[256];
    if (argc != 2 || ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "usage: %s CFPTBL < spec\n", argv[0]);
        return 2;
    }
    int nwt = 0, lorb[MAXO], ncfg = 0, occ[MAXC * MAXO], nz = 0;
    char word[32];
    while (scanf("%31s", word) == 1) {
        if (!strcmp(word, "orbitals")) {
            int x;
            while (nwt < MAXO && scanf("%d", &x) == 1)
                lorb[nwt++] = x;
        } else if (!strcmp(word, "config") && ncfg < MAXC) {
            for (int s = 0; s < nwt; s++)
                if (scanf("%d", &occ[ncfg * nwt + s]) != 1)
                    return 2;
            ncfg++;
        } else if (!strcmp(word, "h")) {
            int a, b;
            double v;
            if (scanf("%d %d %lf", &a, &b, &v) != 3 || a < 1 || b < 1 || a > nwt || b > nwt)
                return 2;
            hv[a][b] = hv[b][a] = v;
        } else if (!strcmp(word, "z")) {
            int a, b;
            double v;
            if (scanf("%d %d %lf", &a, &b, &v) != 3 || a < 1 || b < 1 || a > nwt || b > nwt)
                return 2;
            zv[a][b] = zv[b][a] = v;
            nz++;
        } else if (!strcmp(word, "t") || !strcmp(word, "u")) {
            int a, b;
            double v;
            if (scanf("%d %d %lf", &a, &b, &v) != 3 || a < 1 || b < 1 || a > nwt || b > nwt)
                return 2;
            if (word[0] == 't')
                tv[a][b] = tv[b][a] = v;
            else
                uv[a][b] = uv[b][a] = v;
        } else if (!strcmp(word, "r")) {
            int k, a, b, c, d;
            double v;
            if (scanf("%d %d %d %d %d %lf", &k, &a, &b, &c, &d, &v) != 6 || k < 0 || k >= MAXK ||
                a < 1 || b < 1 || c < 1 || d < 1 || a > nwt || b > nwt || c > nwt || d > nwt)
                return 2;
            rv[k][a][b][c][d] = v;
        } else if (!strcmp(word, "end")) {
            break;
        } else {
            fprintf(stderr, "acso: unknown keyword %s\n", word);
            return 2;
        }
    }

    ac_ci ci;
    if (ac_ci_from_orbitals(&ci, &t, nwt, lorb, ncfg, occ, err, sizeof err)) {
        fprintf(stderr, "acso: %s\n", err);
        return 1;
    }
    ci.in.h = get_h;
    ci.in.rk = get_rk;
    ci.in.zeta = nz ? get_zeta : NULL;
    int tjmax = 0, nel = 0;
    for (int s = 0; s < nwt; s++)
        nel += occ[s];
    for (int i = 0; i < ci.ncsf; i++) {
        ac_lab a = ci.csf[i].c.cpl[nwt];
        printf("csf %d %d %d\n", i, a.ts, a.tl);
        if (a.ts + a.tl > tjmax)
            tjmax = a.ts + a.tl;
    }
    ac_lsj m;
    if (ac_lsj_prepare(&m, &ci, err, sizeof err)) {
        fprintf(stderr, "acso: %s\n", err);
        return 1;
    }
    int *idx = malloc((size_t)ci.ncsf * sizeof *idx);
    double *H = malloc((size_t)ci.ncsf * ci.ncsf * sizeof *H);
    double *w = malloc((size_t)ci.ncsf * sizeof *w);
    for (int tJ = nel % 2; tJ <= tjmax; tJ += 2) {
        double asym;
        int dim = ac_lsj_matrix(&m, &ci, tJ, idx, H, &asym);
        if (dim == 0)
            continue;
        printf("asym %d %.3e\n", tJ, asym);
        for (int p = 0; p < dim; p++)
            printf("diag %d %d %.17g\n", tJ, idx[p], H[(size_t)p * dim + p]);
        if (ac_ci_eigvecs(dim, H, w)) {
            fprintf(stderr, "acso: dsyev failed\n");
            return 1;
        }
        for (int p = 0; p < dim; p++) {
            printf("level %d %.17g\n", tJ, w[p]);
            ac_lsj_g g;
            int e;
            if (tJ > 0 && !ac_lsj_gfactor(&ci, dim, idx, tJ, H + (size_t)p * dim, get_t, get_u,
                                          NULL, &g, &e))
                printf("g %d %.17g %.17g %.17g %.17g %.17g %.17g\n", tJ, w[p], g.gL, g.gS, g.rmc,
                       g.gc, g.ormc);
            else if (tJ > 0) {
                fprintf(stderr, "acso: g factor: engine error %d\n", e);
                return 1;
            }
        }
    }
    free(idx);
    free(H);
    free(w);
    ac_lsj_free(&m);
    ac_ci_free(&ci);
    ac_cfp_free(&t);
    return 0;
}
