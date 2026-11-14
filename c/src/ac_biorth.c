/*
 * ac_biorth.c -- biorthonormal orbital sets and the counter-transformation
 * of CI vectors (S2-M9). See ac_biorth.h and the technical note, Sec. 5.4.
 *
 * The method is that of P.-A. Malmqvist, Int. J. Quantum Chem. 30, 479
 * (1986), and J. Olsen, M. R. Godefroid, P. Joensson, P.-A. Malmqvist,
 * C. Froese Fischer, Phys. Rev. E 52, 4499 (1995); the implementation (the
 * E_ab from this program's angular engine, the closure check) is this
 * program's own.
 */
#include "ac_biorth.h"
#include "ac_csfgen.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, int errlen, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    if (err && errlen > 0)
        vsnprintf(err, (size_t)errlen, fmt, ap);
    va_end(ap);
    return -1;
}

/* ---- the triangular factors -------------------------------------------- */

/* S = LU without pivoting (Doolittle) and the rectangular case n < m (Olsen
   et al. (1995), Eq. (27) and appendix B). n <= m */
static int lu_nm(int n, int m, const double *S, int mode, double tol, double *cx, double *cy,
                 double *tx, double *ty, char *err, int errlen)
{
    double *L = calloc((size_t)n * n + 1, sizeof *L), *U = calloc((size_t)n * n + 1, sizeof *U);
    double *Li = calloc((size_t)n * n + 1, sizeof *Li), *Ui = calloc((size_t)n * n + 1, sizeof *Ui);
    double *LZ = calloc((size_t)n * (m - n) + 1, sizeof *LZ);
    int rc = -1;
    if (!L || !U || !Li || !Ui || !LZ) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    /* Doolittle on the square part T = S[:, 0..n-1] */
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double s = S[(size_t)i * m + j];
            for (int k = 0; k < i; k++)
                s -= L[i * n + k] * U[k * n + j];
            U[i * n + j] = s;
        }
        if (fabs(U[i * n + i]) < tol) {
            fail(err, errlen, "pivot %d of the overlap is %.3e (below %.1e)", i + 1,
                 U[i * n + i], tol);
            goto out;
        }
        L[i * n + i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            double s = S[(size_t)j * m + i];
            for (int k = 0; k < i; k++)
                s -= L[j * n + k] * U[k * n + i];
            L[j * n + i] = s / U[i * n + i];
        }
    }
    /* L D, D^-1 U */
    for (int i = 0; i < n; i++) {
        double d = U[i * n + i];
        double sg = mode == AC_BIORTH_LUNIT ? 1.0 : mode == AC_BIORTH_UUNIT ? d : sqrt(fabs(d));
        for (int j = i; j < n; j++) {
            L[j * n + i] *= sg;
            U[i * n + j] /= sg;
        }
    }
    /* inverses by substitution */
    for (int c = 0; c < n; c++) {
        for (int i = c; i < n; i++) {
            double s = i == c ? 1.0 : 0.0;
            for (int k = c; k < i; k++)
                s -= L[i * n + k] * Li[k * n + c];
            Li[i * n + c] = s / L[i * n + i];
        }
        for (int i = c; i >= 0; i--) {
            double s = i == c ? 1.0 : 0.0;
            for (int k = i + 1; k <= c; k++)
                s -= U[i * n + k] * Ui[k * n + c];
            Ui[i * n + c] = s / U[i * n + i];
        }
    }
    /* L^-1 Z, Z = S[:, n..m-1] */
    for (int i = 0; i < n; i++)
        for (int q = 0; q < m - n; q++) {
            double s = 0.0;
            for (int k = 0; k <= i; k++)
                s += Li[i * n + k] * S[(size_t)k * m + n + q];
            LZ[i * (m - n) + q] = s;
        }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            cx[i * n + j] = Li[j * n + i];
            tx[i * n + j] = L[j * n + i];
        }
    memset(cy, 0, (size_t)m * m * sizeof *cy);
    memset(ty, 0, (size_t)m * m * sizeof *ty);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cy[i * m + j] = Ui[i * n + j];
            ty[i * m + j] = U[i * n + j];
        }
        for (int q = 0; q < m - n; q++) {
            double s = 0.0;
            for (int k = i; k < n; k++)
                s += Ui[i * n + k] * LZ[k * (m - n) + q];
            cy[i * m + n + q] = -s;
            ty[i * m + n + q] = LZ[i * (m - n) + q];
        }
    }
    for (int i = n; i < m; i++)
        cy[i * m + i] = ty[i * m + i] = 1.0;
    rc = 0;
out:
    free(L);
    free(U);
    free(Li);
    free(Ui);
    free(LZ);
    return rc;
}

int ac_biorth_lu(int n, int m, const double *S, int mode, double tol, double *cx, double *cy,
                 double *tx, double *ty, char *err, int errlen)
{
    if (n <= m)
        return lu_nm(n, m, S, mode, tol, cx, cy, tx, ty, err, errlen);
    double *St = malloc((size_t)n * m * sizeof *St);
    if (!St)
        return fail(err, errlen, "out of memory");
    for (int p = 0; p < n; p++)
        for (int q = 0; q < m; q++)
            St[(size_t)q * n + p] = S[(size_t)p * m + q];
    /* with the roles of X and Y exchanged; L and U exchange too */
    int md = mode == AC_BIORTH_LUNIT ? AC_BIORTH_UUNIT : mode == AC_BIORTH_UUNIT ? AC_BIORTH_LUNIT
                                                                                   : mode;
    int rc = lu_nm(m, n, St, md, tol, cy, cx, ty, tx, err, errlen);
    free(St);
    return rc;
}

/* ---- orbitals ------------------------------------------------------------ */

int ac_radial_mix(ac_radial *dst, int nsrc, const ac_radial *const *src, int nmo, const int *l,
                  const double *coef, char *err, int errlen)
{
    memset(dst, 0, sizeof *dst);
    if (nsrc < 1)
        return fail(err, errlen, "no source orbitals");
    const ac_rgrid *g = src[0]->g;
    int ntot = 0;
    for (int s = 0; s < nsrc; s++) {
        if (src[s]->g->n != g->n || src[s]->g->x0 != g->x0 || src[s]->g->h != g->h)
            return fail(err, errlen, "source %d is on another grid", s);
        ntot += src[s]->nmo;
    }
    for (int j = 1; j < nmo; j++)
        if (l[j] < l[j - 1])
            return fail(err, errlen, "MOs not grouped by l");
    if (nmo > 0 && l[nmo - 1] >= AC_BASIS_MAXL)
        return fail(err, errlen, "l = %d too large", l[nmo - 1]);
    int n = g->n;
    dst->g = g;
    dst->z = src[0]->z;
    dst->nmo = nmo;
    dst->l = malloc((size_t)(nmo ? nmo : 1) * sizeof(int));
    dst->idx = malloc((size_t)(nmo ? nmo : 1) * sizeof(int));
    dst->val = calloc((size_t)(nmo ? nmo : 1) * n, sizeof(double));
    dst->der = calloc((size_t)(nmo ? nmo : 1) * n, sizeof(double));
    dst->lap = calloc((size_t)(nmo ? nmo : 1) * n, sizeof(double));
    dst->r0 = calloc((size_t)(nmo ? nmo : 1), sizeof(double));
    if (!dst->l || !dst->idx || !dst->val || !dst->der || !dst->lap || !dst->r0) {
        ac_radial_free(dst);
        return fail(err, errlen, "out of memory");
    }
    for (int lv = 0, j = 0; lv < AC_BASIS_MAXL; lv++) {
        dst->first[lv] = j;
        while (j < nmo && l[j] == lv) {
            dst->l[j] = lv;
            dst->idx[j] = j - dst->first[lv];
            j++;
        }
    }
    for (int j = 0; j < nmo; j++) {
        double *v = dst->val + (size_t)j * n, *d = dst->der + (size_t)j * n;
        double *x = dst->lap + (size_t)j * n;
        int off = 0;
        for (int s = 0; s < nsrc; s++) {
            const ac_radial *r = src[s];
            for (int i = 0; i < r->nmo; i++) {
                double c = coef[(size_t)j * ntot + off + i];
                if (c == 0.0)
                    continue;
                if (r->l[i] != l[j]) {
                    ac_radial_free(dst);
                    return fail(err, errlen, "MO %d (l = %d) mixes an orbital of l = %d", j,
                                l[j], r->l[i]);
                }
                const double *sv = r->val + (size_t)i * n, *sd = r->der + (size_t)i * n;
                const double *sx = r->lap + (size_t)i * n;
                for (int p = 0; p < n; p++) {
                    v[p] += c * sv[p];
                    d[p] += c * sd[p];
                    x[p] += c * sx[p];
                }
                dst->r0[j] += c * r->r0[i];
            }
            off += r->nmo;
        }
    }
    return 0;
}

/* ---- E_ab on the CSFs ------------------------------------------------------ */

typedef struct {
    int a, b;
} pair_ab;

static double delta_ab(void *ctx, int a, int b, int la, int lb)
{
    const pair_ab *p = ctx;
    (void)la;
    (void)lb;
    return a == p->a && b == p->b ? 1.0 : 0.0;
}

double ac_biorth_eij(ac_ci *ci, int i, int j, int a, int b, int *err)
{
    pair_ab p = {a, b};
    double w = ac_ci_one_ll(ci, i, j, 0, 0, delta_ab, &p, err);
    if (*err || w == 0.0)
        return 0.0;
    const ac_csf *c = &ci->csf[i].c;
    ac_lab T = c->cpl[c->nsh];
    int l = ci->lorb[a];
    return -sqrt(2.0 * (2 * l + 1)) * w / sqrt((double)(T.ts + 1) * (T.tl + 1));
}

static void csf_occ(const ac_ci *ci, int i, unsigned char *occ)
{
    memset(occ, 0, (size_t)ci->nwt + 1);
    const ac_ci_csf *f = &ci->csf[i];
    for (int s = 1; s <= f->c.nsh; s++)
        occ[f->seq[s]] = (unsigned char)(occ[f->seq[s]] + f->c.nel[s]);
}

/* configurations: distinct occupations, and the CSFs of each */
typedef struct {
    int ncfg, w;
    unsigned char *occ; /* [ncfg * w] */
    int *first, *member; /* member[first[c]..first[c+1]-1] */
} cfgs;

static int cfgs_build(cfgs *g, const ac_ci *ci, const unsigned char *occ)
{
    int n = ci->ncsf, w = ci->nwt + 1;
    g->w = w;
    g->ncfg = 0;
    g->occ = malloc((size_t)(n ? n : 1) * w);
    int *of = malloc((size_t)(n ? n : 1) * sizeof *of), *cnt = calloc((size_t)n + 1, sizeof *cnt);
    g->first = calloc((size_t)n + 2, sizeof *g->first);
    g->member = malloc((size_t)(n ? n : 1) * sizeof *g->member);
    if (!g->occ || !of || !cnt || !g->first || !g->member) {
        free(of);
        free(cnt);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        int c;
        for (c = 0; c < g->ncfg; c++)
            if (!memcmp(g->occ + (size_t)c * w, occ + (size_t)i * w, (size_t)w))
                break;
        if (c == g->ncfg)
            memcpy(g->occ + (size_t)g->ncfg++ * w, occ + (size_t)i * w, (size_t)w);
        of[i] = c;
        cnt[c]++;
    }
    for (int c = 0; c < g->ncfg; c++)
        g->first[c + 1] = g->first[c] + cnt[c];
    memset(cnt, 0, (size_t)(n + 1) * sizeof *cnt);
    for (int i = 0; i < n; i++)
        g->member[g->first[of[i]] + cnt[of[i]]++] = i;
    free(of);
    free(cnt);
    return 0;
}

static void cfgs_free(cfgs *g)
{
    free(g->occ);
    free(g->first);
    free(g->member);
}

static int cfgs_find(const cfgs *g, const unsigned char *occ)
{
    for (int c = 0; c < g->ncfg; c++)
        if (!memcmp(g->occ + (size_t)c * g->w, occ, (size_t)g->w))
            return c;
    return -1;
}

static int grow(ac_biorth_ops *o, int *cap)
{
    if (o->n < *cap)
        return 0;
    int nc = *cap ? 2 * *cap : 1024;
    int *a = realloc(o->a, (size_t)nc * sizeof *a);
    if (a)
        o->a = a;
    int *i = realloc(o->i, (size_t)nc * sizeof *i);
    if (i)
        o->i = i;
    int *j = realloc(o->j, (size_t)nc * sizeof *j);
    if (j)
        o->j = j;
    double *v = realloc(o->v, (size_t)nc * sizeof *v);
    if (v)
        o->v = v;
    if (!a || !i || !j || !v)
        return -1;
    *cap = nc;
    return 0;
}

void ac_biorth_ops_free(ac_biorth_ops *o)
{
    free(o->occ);
    free(o->start);
    free(o->a);
    free(o->i);
    free(o->j);
    free(o->v);
    memset(o, 0, sizeof *o);
}

int ac_biorth_ops_build(ac_biorth_ops *o, ac_ci *ci, char *err, int errlen)
{
    memset(o, 0, sizeof *o);
    int n = ci->ncsf, nwt = ci->nwt, w = nwt + 1, cap = 0;
    o->ncsf = n;
    o->nwt = nwt;
    o->occ = malloc((size_t)(n ? n : 1) * w);
    o->start = calloc((size_t)nwt + 2, sizeof *o->start);
    unsigned char *t = malloc((size_t)w);
    cfgs g = {0};
    if (!o->occ || !o->start || !t) {
        free(t);
        ac_biorth_ops_free(o);
        return fail(err, errlen, "out of memory");
    }
    for (int i = 0; i < n; i++)
        csf_occ(ci, i, o->occ + (size_t)i * w);
    if (cfgs_build(&g, ci, o->occ)) {
        free(t);
        cfgs_free(&g);
        ac_biorth_ops_free(o);
        return fail(err, errlen, "out of memory");
    }
    int rc = -1;
    for (int k = 1; k <= nwt; k++) {
        o->start[k] = o->n;
        int l = ci->lorb[k];
        for (int a = 1; a < k; a++) {
            if (ci->lorb[a] != l)
                continue;
            for (int c = 0; c < g.ncfg; c++) {
                const unsigned char *oc = g.occ + (size_t)c * w;
                if (oc[k] == 0 || oc[a] >= 4 * l + 2)
                    continue;
                memcpy(t, oc, (size_t)w);
                t[k]--;
                t[a]++;
                int c2 = cfgs_find(&g, t);
                if (c2 < 0)
                    continue;
                for (int pj = g.first[c]; pj < g.first[c + 1]; pj++)
                    for (int pi = g.first[c2]; pi < g.first[c2 + 1]; pi++) {
                        int i = g.member[pi], j = g.member[pj], e = 0;
                        double v = ac_biorth_eij(ci, i, j, a, k, &e);
                        if (e) {
                            fail(err, errlen, "engine error %d for <%d|E_%d,%d|%d>", e, i, a, k, j);
                            goto out;
                        }
                        if (v == 0.0)
                            continue;
                        if (grow(o, &cap)) {
                            fail(err, errlen, "out of memory");
                            goto out;
                        }
                        o->a[o->n] = a;
                        o->i[o->n] = i;
                        o->j[o->n] = j;
                        o->v[o->n++] = v;
                    }
            }
        }
    }
    o->start[nwt + 1] = o->n;
    rc = 0;
out:
    free(t);
    cfgs_free(&g);
    if (rc)
        ac_biorth_ops_free(o);
    return rc;
}

/* c <- exp(sum_a (t_ak/t_kk) E_ak) t_kk^(N_k) c, orbital by orbital (Olsen
   et al. (1995), Eqs. (42)-(47); Malmqvist (1986), section VI) */
int ac_biorth_transform(const ac_biorth_ops *o, const ac_ci *ci, const double *t, int nvec,
                        double *c, char *err, int errlen)
{
    int n = o->ncsf, nwt = o->nwt, w = nwt + 1;
    double *term = malloc((size_t)(n ? n : 1) * sizeof *term);
    double *next = malloc((size_t)(n ? n : 1) * sizeof *next);
    double *s = calloc((size_t)w, sizeof *s);
    if (!term || !next || !s) {
        free(term);
        free(next);
        free(s);
        return fail(err, errlen, "out of memory");
    }
    int rc = -1;
    for (int k = 1; k <= nwt; k++) {
        int l = ci->lorb[k];
        double tkk = t[(size_t)k * w + k];
        if (tkk == 0.0) {
            fail(err, errlen, "t_kk = 0 for orbital %d", k);
            goto out;
        }
        int any = 0;
        for (int a = 1; a < k; a++) {
            s[a] = ci->lorb[a] == l ? t[(size_t)a * w + k] / tkk : 0.0;
            any |= s[a] != 0.0;
        }
        for (int v = 0; v < nvec; v++) {
            double *x = c + (size_t)v * n;
            /* t_kk^(N_k) first, then the exponential */
            if (tkk != 1.0)
                for (int i = 0; i < n; i++) {
                    int nk = o->occ[(size_t)i * w + k];
                    if (nk)
                        x[i] *= pow(tkk, nk);
                }
            if (!any)
                continue;
            memcpy(term, x, (size_t)n * sizeof *term);
            for (int p = 1; p <= 4 * l + 2; p++) {
                memset(next, 0, (size_t)n * sizeof *next);
                int nz = 0;
                for (int e = o->start[k]; e < o->start[k + 1]; e++) {
                    double f = s[o->a[e]] * o->v[e] * term[o->j[e]];
                    if (f != 0.0) {
                        next[o->i[e]] += f;
                        nz = 1;
                    }
                }
                if (!nz)
                    break;
                for (int i = 0; i < n; i++) {
                    next[i] /= p;
                    x[i] += next[i];
                }
                memcpy(term, next, (size_t)n * sizeof *term);
            }
        }
    }
    rc = 0;
out:
    free(term);
    free(next);
    free(s);
    return rc;
}

/* ---- closure ---------------------------------------------------------------- */

/* closure of the CSF space under E_ak, a < k (Malmqvist (1986), section V;
   Olsen et al. (1995), section II) */
int ac_biorth_closure(ac_ci *ci, double *dev, char *what, int whatlen, char *err, int errlen)
{
    int n = ci->ncsf, nwt = ci->nwt, w = nwt + 1;
    *dev = 0.0;
    if (what && whatlen > 0)
        what[0] = 0;
    unsigned char *occ = malloc((size_t)(n ? n : 1) * w), *t = malloc((size_t)w);
    int *iocc = malloc((size_t)w * sizeof *iocc);
    cfgs g = {0};
    ac_csflist gen;
    ac_csflist_init(&gen);
    ac_ci tmp = *ci;
    tmp.csf = NULL;
    tmp.thc = NULL;
    int rc = -1;
    if (!occ || !t || !iocc) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    for (int i = 0; i < n; i++)
        csf_occ(ci, i, occ + (size_t)i * w);
    if (cfgs_build(&g, ci, occ)) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    for (int c = 0; c < g.ncfg; c++)
        for (int k = 1; k <= nwt; k++)
            for (int a = 1; a < k; a++) {
                int l = ci->lorb[k];
                const unsigned char *oc = g.occ + (size_t)c * w;
                if (ci->lorb[a] != l || oc[k] == 0 || oc[a] >= 4 * l + 2)
                    continue;
                memcpy(t, oc, (size_t)w);
                t[k]--;
                t[a]++;
                int c2 = cfgs_find(&g, t);
                /* the target configuration in full, after the CSFs of c and c2 */
                gen.len = 0;
                for (int s = 1; s <= nwt; s++)
                    iocc[s - 1] = t[s];
                if (ac_csfgen(ci->t, nwt, ci->lorb + 1, iocc, &gen) < 0) {
                    /* a shell CFPTBL does not hold (e.g. g^3): no list can
                       contain it, so the list is not closed if E_ak reaches
                       it at all; count it as weight 1 */
                    if (*dev < 1.0) {
                        *dev = 1.0;
                        if (what && whatlen > 0)
                            snprintf(what, (size_t)whatlen,
                                     "E_%d,%d leads to a configuration that CFPTBL cannot "
                                     "couple (counted as weight 1)",
                                     a, k);
                    }
                    continue;
                }
                int n1 = g.first[c + 1] - g.first[c];
                int n2 = c2 < 0 ? 0 : g.first[c2 + 1] - g.first[c2];
                free(tmp.csf);
                tmp.csf = calloc((size_t)(n1 + n2 + gen.len + 1), sizeof *tmp.csf);
                if (!tmp.csf) {
                    fail(err, errlen, "out of memory");
                    goto out;
                }
                for (int p = 0; p < n1; p++)
                    tmp.csf[p] = ci->csf[g.member[g.first[c] + p]];
                for (int p = 0; p < n2; p++)
                    tmp.csf[n1 + p] = ci->csf[g.member[g.first[c2] + p]];
                for (int p = 0; p < gen.len; p++) {
                    tmp.csf[n1 + n2 + p].c = gen.v[p];
                    for (int s = 1; s <= nwt; s++)
                        tmp.csf[n1 + n2 + p].seq[s] = s;
                }
                tmp.ncsf = n1 + n2 + gen.len;
                for (int j = 0; j < n1; j++) {
                    double full = 0.0, in = 0.0;
                    int e = 0;
                    for (int p = 0; p < gen.len && !e; p++) {
                        double v = ac_biorth_eij(&tmp, n1 + n2 + p, j, a, k, &e);
                        full += v * v;
                    }
                    for (int p = 0; p < n2 && !e; p++) {
                        double v = ac_biorth_eij(&tmp, n1 + p, j, a, k, &e);
                        in += v * v;
                    }
                    if (e) {
                        fail(err, errlen, "engine error %d", e);
                        goto out;
                    }
                    if (full - in > *dev) {
                        *dev = full - in;
                        if (what && whatlen > 0)
                            snprintf(what, (size_t)whatlen,
                                     "CSF %d, E_%d,%d: |E|j>|^2 = %.6g, inside the list %.6g",
                                     g.member[g.first[c] + j], a, k, full, in);
                    }
                }
            }
    rc = 0;
out:
    free(occ);
    free(t);
    free(iocc);
    free(tmp.csf);
    ac_ci_free_cache(&tmp);
    cfgs_free(&g);
    ac_csflist_free(&gen);
    return rc;
}
