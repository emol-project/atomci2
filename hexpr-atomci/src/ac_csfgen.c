/*
 * ac_csfgen.c -- all CSFs of a configuration (see ac_csfgen.h).
 */
#include "ac_csfgen.h"

#include <stdlib.h>
#include <string.h>

void ac_csflist_init(ac_csflist *s)
{
    memset(s, 0, sizeof *s);
}

void ac_csflist_free(ac_csflist *s)
{
    free(s->v);
    memset(s, 0, sizeof *s);
}

static int push(ac_csflist *s, const ac_csf *c)
{
    if (s->len == s->cap) {
        int ncap = s->cap ? 2 * s->cap : 64;
        ac_csf *nv = realloc(s->v, (size_t)ncap * sizeof *nv);
        if (!nv)
            return -1;
        s->v = nv;
        s->cap = ncap;
    }
    s->v[s->len++] = *c;
    return 0;
}

/* couple slot k (1-based) to the chain built so far, then recurse */
static int gen(const ac_cfptab *t, int nsh, const int *l, const int *occ,
               int k, ac_csf *cur, ac_csflist *out)
{
    if (k > nsh)
        return push(out, cur);
    const int *codes;
    int ns = ac_cfp_states(t, l[k - 1], occ[k - 1], &codes);
    if (ns < 0)
        return -1;
    for (int i = 0; i < ns; i++) {
        ac_shell_qn q;
        if (ac_cfp_decode(t, codes[i], &q))
            return -1;
        ac_lab prev = cur->cpl[k - 1];
        cur->l[k] = l[k - 1];
        cur->code[k] = codes[i];
        cur->nel[k] = occ[k - 1];
        cur->f[k].ts = q.ts;
        cur->f[k].tl = q.tL;
        for (int ts = abs(q.ts - prev.ts); ts <= q.ts + prev.ts; ts += 2)
            for (int tl = abs(q.tL - prev.tl); tl <= q.tL + prev.tl; tl += 2) {
                cur->cpl[k].ts = ts;
                cur->cpl[k].tl = tl;
                if (gen(t, nsh, l, occ, k + 1, cur, out))
                    return -1;
            }
    }
    return 0;
}

int ac_csfgen(const ac_cfptab *t, int nsh, const int *l, const int *occ,
              ac_csflist *out)
{
    if (nsh < 1 || nsh > AC_CSF_MAXSH)
        return -1;
    for (int k = 0; k < nsh; k++)
        if (occ[k] < 0 || occ[k] > 4 * l[k] + 2)
            return -1;
    ac_csf cur;
    memset(&cur, 0, sizeof cur);
    cur.nsh = nsh;
    int before = out->len;
    if (gen(t, nsh, l, occ, 1, &cur, out))
        return -1;
    return out->len - before;
}
