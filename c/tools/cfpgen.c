/*
 * cfpgen -- write the CFP table generated in C (ac_cfp_generate) as CFPTBL,
 * or compare it with an existing CFPTBL.
 *
 *   cfpgen OUT             write the table to OUT
 *   cfpgen -c CFPTBL       compare with CFPTBL (the Fortran cfpa + cfpb):
 *                          every integer exactly and every CFP value; prints
 *                          the largest difference, exit 0 when they agree
 *                          bit for bit
 */
#include "ac_cfp.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int compare(const ac_cfptab *g, const ac_cfptab *f)
{
    int bad = 0;
#define CMPI(name, a, b)                                                                  \
    if ((a) != (b)) {                                                                     \
        if (bad++ < 20)                                                                   \
            printf("  %s: generated %d, CFPTBL %d\n", name, (int)(a), (int)(b));         \
    }
    CMPI("ibel", g->ibel, f->ibel);
    CMPI("icnn", g->icnn, f->icnn);
    CMPI("ilfn", g->ilfn, f->ilfn);
    CMPI("icmtn", g->icmtn, f->icmtn);
    CMPI("icdx", g->icdx, f->icdx);
    if (bad)
        return bad;
    for (int i = 0; i < 18; i++) {
        CMPI("name", g->name[i], f->name[i]);
        CMPI("immint", g->immint[i], f->immint[i]);
        CMPI("ilstad", g->ilstad[i], f->ilstad[i]);
    }
    for (int j = 0; j < g->icnn; j++)
        for (int i = 0; i < 9; i++)
            CMPI("iconnt", g->iconnt[j][i], f->iconnt[j][i]);
    for (int j = 0; j < g->icmtn; j++)
        for (int i = 0; i < 5; i++)
            CMPI("icmtdx", g->icmtdx[j][i], f->icmtdx[j][i]);
    for (int i = 0; i < g->ilfn; i++)
        CMPI("ilfunc", g->ilfunc[i], f->ilfunc[i]);
#undef CMPI
    int ndiff = 0, worst = -1;
    double dmax = 0.0;
    for (int i = 0; i < g->icdx; i++) {
        double d = fabs(g->cmttbl[i] - f->cmttbl[i]);
        if (memcmp(&g->cmttbl[i], &f->cmttbl[i], sizeof(double)))
            ndiff++;
        if (d > dmax)
            dmax = d, worst = i;
    }
    printf("integers: %d differences; CFP values: %d of %d not bit-identical, largest "
           "difference %.3e",
           bad, ndiff, g->icdx, dmax);
    if (worst >= 0)
        printf(" (CMTTBL(%d): %.17g vs %.17g)", worst + 1, g->cmttbl[worst], f->cmttbl[worst]);
    printf("\n");
    return bad + ndiff;
}

int main(int argc, char **argv)
{
    char err[256];
    ac_cfptab g;
    if (argc == 3 && !strcmp(argv[1], "-c")) {
        ac_cfptab f;
        if (ac_cfp_generate(&g, err, sizeof err) || ac_cfp_load(&f, argv[2], err, sizeof err)) {
            fprintf(stderr, "cfpgen: %s\n", err);
            return 1;
        }
        printf("generated: ibel %d, icnn %d, ilfn %d, icmtn %d, icdx %d\n", g.ibel, g.icnn, g.ilfn,
               g.icmtn, g.icdx);
        int r = compare(&g, &f);
        ac_cfp_free(&g);
        ac_cfp_free(&f);
        return r ? 2 : 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: cfpgen OUT | cfpgen -c CFPTBL\n");
        return 2;
    }
    if (ac_cfp_generate(&g, err, sizeof err) || ac_cfp_write(&g, argv[1], err, sizeof err)) {
        fprintf(stderr, "cfpgen: %s\n", err);
        return 1;
    }
    ac_cfp_free(&g);
    return 0;
}
