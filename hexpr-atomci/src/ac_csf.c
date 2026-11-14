/*
 * ac_csf.c -- one side of a CSF (see ac_csf.h).
 */
#include "ac_csf.h"

int ac_csf_from_shms(ac_csf *c, const ac_cfptab *t, int nht,
                     const int32_t *nhlf, const int32_t *codes,
                     const int32_t *nel, const int32_t *csl)
{
    if (nht < 0 || nht > AC_CSF_MAXSH)
        return -1;
    c->nsh = nht;
    c->cpl[0].ts = 0;
    c->cpl[0].tl = 0;
    for (int k = 1; k <= nht; k++) {
        ac_shell_qn q;
        if (ac_cfp_decode(t, codes[k - 1], &q))
            return -1;
        if (q.tl != nhlf[k - 1] - 1)
            return -1;
        c->l[k] = q.tl / 2;
        c->code[k] = codes[k - 1];
        c->nel[k] = nel[k - 1];
        c->f[k].ts = q.ts;
        c->f[k].tl = q.tL;
        c->cpl[k].ts = csl[k - 1] / 256 - 1;
        c->cpl[k].tl = csl[k - 1] % 256 - 1;
    }
    return 0;
}
