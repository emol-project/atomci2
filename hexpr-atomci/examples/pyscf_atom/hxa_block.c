/*
 * hxa_block.c -- a small C layer for pyscf_atom.py: the CSFs of a list of
 * configurations, and one (S, L, parity) block of H with the integrals put in.
 *
 *   h    [nwt*nwt]                 h(a,b), with a, b = 0..nwt-1
 *   rk   [(kmax+1)*nwt*nwt*nwt*nwt]  R^k(ab;cd) in C order
 */
#include "hexpr_atomci.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    ac_cfptab t;
    ac_orb_csf *csf;
    int n, nwt;
} hxa;

void *hxa_open(int nwt, const int *lorb, int ncfg, const int *occ, char *err, int errlen)
{
    hxa *x = calloc(1, sizeof *x);
    if (!x) {
        strncpy(err, "out of memory", (size_t)errlen - 1);
        return NULL;
    }
    x->nwt = nwt;
    if (ac_cfp_generate(&x->t, err, errlen)) {
        free(x);
        return NULL;
    }
    if (ac_orb_csfs(&x->t, nwt, lorb, ncfg, occ, &x->csf, &x->n, err, errlen)) {
        ac_cfp_free(&x->t);
        free(x);
        return NULL;
    }
    return x;
}

void hxa_close(void *p)
{
    hxa *x = p;
    free(x->csf);
    ac_cfp_free(&x->t);
    free(x);
}

int hxa_ncsf(const void *p)
{
    return ((const hxa *)p)->n;
}

static int csf_parity(const ac_csf *c)
{
    int s = 0;
    for (int i = 1; i <= c->nsh; i++)
        s += c->nel[i] * c->l[i];
    return s % 2;
}

void hxa_label(const void *p, int i, int *ts, int *tl, int *parity)
{
    const ac_csf *c = &((const hxa *)p)->csf[i].c;
    *ts = c->cpl[c->nsh].ts;
    *tl = c->cpl[c->nsh].tl;
    *parity = csf_parity(c);
}

typedef struct {
    const double *h, *rk;
    int nwt, kmax, bad;
    double sum;
} accum;

static void add_term(void *ctx, int k, int a, int b, int c, int d, double coef)
{
    accum *s = ctx;
    size_t n = (size_t)s->nwt;
    if (k < 0) {
        s->sum += coef * s->h[(size_t)(a - 1) * n + (size_t)(b - 1)];
    } else if (k > s->kmax) {
        s->bad = 1;
    } else {
        size_t i = (size_t)k;
        i = i * n + (size_t)(a - 1);
        i = i * n + (size_t)(b - 1);
        i = i * n + (size_t)(c - 1);
        i = i * n + (size_t)(d - 1);
        s->sum += coef * s->rk[i];
    }
}

/* Fills the upper triangle of H, dim x dim, for the CSFs of (ts, tl, parity) in
   their list order. dim must be their number. Returns 0, or nonzero on an error. */
int hxa_block(void *p, int ts, int tl, int parity, const double *h, const double *rk, int kmax,
              double *H, int dim)
{
    hxa *x = p;
    int *idx = malloc(sizeof(int) * (size_t)(dim > 0 ? dim : 1));
    int m = 0, rc = 0;
    if (!idx)
        return -1;
    for (int i = 0; i < x->n; i++) {
        const ac_csf *c = &x->csf[i].c;
        if (c->cpl[c->nsh].ts == ts && c->cpl[c->nsh].tl == tl && csf_parity(c) == parity) {
            if (m == dim) {
                free(idx);
                return -2;
            }
            idx[m++] = i;
        }
    }
    if (m != dim) {
        free(idx);
        return -2;
    }
    for (int i = 0; i < dim && !rc; i++) {
        for (int j = i; j < dim && !rc; j++) {
            accum s = {h, rk, x->nwt, kmax, 0, 0.0};
            rc = ac_pair_terms(&x->t, &x->csf[idx[i]], &x->csf[idx[j]], NULL, add_term, &s, NULL);
            if (!rc && s.bad)
                rc = -3;
            H[(size_t)i * (size_t)dim + (size_t)j] = s.sum;
        }
    }
    free(idx);
    return rc;
}
