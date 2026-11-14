/*
 * ac_basis.c -- radial basis and MO coefficients (see ac_basis.h).
 */
#include "ac_basis.h"
#include "ac_ftnio.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, int errlen, const char *fmt, ...)
{
    if (err && errlen > 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, (size_t)errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* integral of r^q exp(-a r) dr = q!/a^(q+1), and of r^q exp(-a r^2) dr =
   Gamma((q+1)/2)/(2 a^((q+1)/2)), for q > -1 */
static double sto_int(int q, double a)
{
    return tgamma(q + 1.0) / pow(a, q + 1.0);
}

static double gto_int(int q, double a)
{
    return tgamma((q + 1.0) / 2.0) / (2.0 * pow(a, (q + 1.0) / 2.0));
}

double ac_bfn_moment(const ac_bfn *fi, const ac_bfn *fj, int p)
{
    /* R_i R_j r^(2+p) = sum c c r^(ni+nj-2+2+p) exp(...) */
    int q = fi->n + fj->n + p;
    double s = 0.0;
    for (int a = 0; a < fi->nprim; a++)
        for (int b = 0; b < fj->nprim; b++)
            s += fi->coef[a] * fj->coef[b] *
                 (fi->gto ? gto_int(q, fi->alpha[a] + fj->alpha[b])
                          : sto_int(q, fi->alpha[a] + fj->alpha[b]));
    return s;
}

double ac_bfn_h(const ac_bfn *fi, const ac_bfn *fj, double z)
{
    /* 1/2 int (R_i' R_j' + l(l+1) R_i R_j / r^2) r^2 dr - Z int R_i R_j r dr,
       R' = ((n-1) r^(n-2) - a r^(n-1)) e^(-a r)          (STO)
            ((n-1) r^(n-2) - 2 a r^n) e^(-a r^2)           (GTO) */
    int n = fi->n, m = fj->n, l = fi->l, q = n + m;
    double s = 0.0;
    for (int i = 0; i < fi->nprim; i++)
        for (int j = 0; j < fj->nprim; j++) {
            double a = fi->alpha[i], b = fj->alpha[j], e = a + b, t;
            double (*in)(int, double) = fi->gto ? gto_int : sto_int;
            int g = fi->gto ? 1 : 0; /* GTO derivative terms carry r^(+1) more */
            t = ((n - 1) * (m - 1) + l * (l + 1)) * in(q - 2, e);
            t -= (g ? 2.0 : 1.0) * ((n - 1) * b + (m - 1) * a) * in(q - 1 + g, e);
            t += (g ? 4.0 : 1.0) * a * b * in(q + 2 * g, e);
            s += fi->coef[i] * fj->coef[j] * (0.5 * t - z * in(q - 1, e));
        }
    return s;
}

/* (L R)(r) = R'' + 2R'/r - l(l+1)R/r^2 = sum_prim c (A/r^2 + B/r + C + D r^2) prim,
   A = n(n-1) - l(l+1);  STO: B = -2an, C = a^2, D = 0;
                         GTO: B = 0, C = -2a(2n+1), D = 4a^2 */
static void lap_terms(const ac_bfn *f, int i, double t[4])
{
    double a = f->alpha[i];
    int n = f->n;
    t[0] = n * (n - 1) - f->l * (f->l + 1);
    t[1] = f->gto ? 0.0 : -2.0 * a * n;
    t[2] = f->gto ? -2.0 * a * (2 * n + 1) : a * a;
    t[3] = f->gto ? 4.0 * a * a : 0.0;
}

double ac_bfn_lap2(const ac_bfn *fi, const ac_bfn *fj)
{
    static const int pw[4] = {-2, -1, 0, 2};
    int q0 = fi->n + fj->n; /* r^(ni-1) r^(nj-1) r^2 */
    double s = 0.0;
    for (int i = 0; i < fi->nprim; i++)
        for (int j = 0; j < fj->nprim; j++) {
            double ti[4], tj[4], e = fi->alpha[i] + fj->alpha[j];
            lap_terms(fi, i, ti);
            lap_terms(fj, j, tj);
            for (int x = 0; x < 4; x++)
                for (int y = 0; y < 4; y++) {
                    if (ti[x] == 0.0 || tj[y] == 0.0)
                        continue;
                    int q = q0 + pw[x] + pw[y];
                    s += fi->coef[i] * fj->coef[j] * ti[x] * tj[y] *
                         (fi->gto ? gto_int(q, e) : sto_int(q, e));
                }
        }
    return s;
}

double ac_bfn_lap(const ac_bfn *f, double r)
{
    double s = 0.0;
    for (int i = 0; i < f->nprim; i++) {
        double t[4];
        lap_terms(f, i, t);
        double e = f->gto ? exp(-f->alpha[i] * r * r) : exp(-f->alpha[i] * r);
        s += f->coef[i] * e * (t[0] / (r * r) + t[1] / r + t[2] + t[3] * r * r);
    }
    return s * pow(r, f->n - 1);
}

double ac_bfn_value(const ac_bfn *f, double r)
{
    double s = 0.0;
    for (int a = 0; a < f->nprim; a++)
        s += f->coef[a] * (f->gto ? exp(-f->alpha[a] * r * r) : exp(-f->alpha[a] * r));
    return s * pow(r, f->n - 1);
}

/* scale the contraction coefficients so that the function is normalized */
static void normalize(ac_bfn *f)
{
    double s = ac_bfn_moment(f, f, 0);
    double c = 1.0 / sqrt(s);
    for (int a = 0; a < f->nprim; a++)
        f->coef[a] *= c;
}

static int load_ao(ac_basis *b, const char *path, char *err, int errlen)
{
    ac_ftn_file f;
    if (ac_ftn_open(&f, path, "rb"))
        return fail(err, errlen, "cannot open %s", path);
    if (ac_ftn_read(&f) != AC_FTN_OK) {
        ac_ftn_close(&f);
        return fail(err, errlen, "%s: no header record", path);
    }
    ac_ftn_cursor c;
    ac_ftn_cursor_init(&c, &f);
    for (int i = 0; i < 33; i++)
        (void)ac_ftn_i32(&c);
    int lmax = ac_ftn_i32(&c), nwt = ac_ftn_i32(&c);
    (void)ac_ftn_i32(&c);
    (void)ac_ftn_i32(&c);
    (void)ac_ftn_i32(&c); /* LOEI */
    if (lmax < 1 || lmax > AC_BASIS_MAXL || nwt < 1 || nwt > 150) {
        ac_ftn_close(&f);
        return fail(err, errlen, "%s: implausible LMAX %d, NWT %d", path, lmax, nwt);
    }
    size_t sto_len = 4 * (size_t)(38 + lmax + 2 * nwt + nwt) + 8 * (size_t)(2 + nwt);
    size_t gto_len = sto_len + 4 * (size_t)(lmax + 2 * nwt) + 8 * (size_t)(2 * 40 * 150);
    if (f.len != sto_len && f.len != gto_len) {
        ac_ftn_close(&f);
        return fail(err, errlen, "%s: header length %zu is neither STO (%zu) nor GTO (%zu)",
                    path, f.len, sto_len, gto_len);
    }
    b->gto = f.len == gto_len;
    b->nl = lmax;
    b->nbf = nwt;
    int issh[AC_BASIS_MAXL], npqn[150], ifst[AC_BASIS_MAXL], inxt[150], nterm[150];
    double del[150];
    for (int l = 0; l < lmax; l++) issh[l] = ac_ftn_i32(&c);
    for (int i = 0; i < 2 * nwt; i++) (void)ac_ftn_i32(&c); /* MPOB, MOB */
    b->z = ac_ftn_f64(&c);
    (void)ac_ftn_f64(&c); /* Z0 */
    for (int i = 0; i < nwt; i++) npqn[i] = ac_ftn_i32(&c);
    for (int i = 0; i < nwt; i++) del[i] = ac_ftn_f64(&c);
    b->bf = calloc((size_t)nwt, sizeof *b->bf);
    if (!b->bf) {
        ac_ftn_close(&f);
        return fail(err, errlen, "out of memory");
    }
    int order[150], k = 0;
    if (!b->gto) {
        for (int i = 0; i < nwt; i++)
            order[i] = i; /* contiguous by l */
    } else {
        for (int l = 0; l < lmax; l++) ifst[l] = ac_ftn_i32(&c);
        for (int i = 0; i < nwt; i++) inxt[i] = ac_ftn_i32(&c);
        for (int i = 0; i < nwt; i++) nterm[i] = ac_ftn_i32(&c);
        /* CE(40,150) then OE(40,150), column-major */
        static double ce[150][40], oe[150][40];
        for (int i = 0; i < 150; i++)
            for (int j = 0; j < 40; j++)
                ce[i][j] = ac_ftn_f64(&c);
        for (int i = 0; i < 150; i++)
            for (int j = 0; j < 40; j++)
                oe[i][j] = ac_ftn_f64(&c);
        for (int l = 0; l < lmax; l++) /* follow the IFST/INXT chain */
            for (int i = ifst[l], m = 0; m < issh[l]; m++, i = inxt[i - 1])
                order[k++] = i - 1;
        for (int i = 0; i < nwt; i++) {
            ac_bfn *g = &b->bf[i];
            g->nprim = nterm[i];
            if (g->nprim < 1 || g->nprim > AC_BASIS_MAXPRIM) {
                ac_ftn_close(&f);
                return fail(err, errlen, "%s: NTERM(%d) = %d", path, i + 1, nterm[i]);
            }
            /* CE are coefficients of normalized primitives; the scaling by
               OE^((2n+1)/4) happens after the header is written */
            for (int j = 0; j < g->nprim; j++) {
                g->alpha[j] = del[i] * oe[i][j]; /* DEL is already squared */
                g->coef[j] = ce[i][j] * pow(g->alpha[j], (2 * npqn[i] + 1) / 4.0);
            }
        }
    }
    ac_ftn_close(&f);
    if (c.bad)
        return fail(err, errlen, "%s: header shorter than expected", path);

    /* reorder by l (STO: already) and fill l, n, primitives */
    ac_bfn *tmp = calloc((size_t)nwt, sizeof *tmp);
    if (!tmp)
        return fail(err, errlen, "out of memory");
    int pos = 0;
    for (int l = 0; l < lmax; l++) {
        b->first[l] = pos;
        b->count[l] = issh[l];
        for (int m = 0; m < issh[l]; m++, pos++) {
            int i = order[pos];
            ac_bfn *g = &tmp[pos];
            if (b->gto)
                *g = b->bf[i];
            else {
                g->nprim = 1;
                g->alpha[0] = del[i];
                g->coef[0] = 1.0;
            }
            g->gto = b->gto;
            g->l = l;
            g->n = npqn[i];
            normalize(g);
        }
    }
    free(b->bf);
    b->bf = tmp;
    return 0;
}

static int load_vec(ac_basis *b, const char *path, char *err, int errlen)
{
    ac_ftn_file f;
    if (ac_ftn_open(&f, path, "rb"))
        return fail(err, errlen, "cannot open %s", path);
    int rc = -1;
    if (ac_ftn_read(&f) != AC_FTN_OK || f.len != 4 * (3 + 20 + 20 + 21)) {
        fail(err, errlen, "%s: header is not WRTCP's", path);
        goto out;
    }
    ac_ftn_cursor c;
    ac_ftn_cursor_init(&c, &f);
    (void)ac_ftn_i32(&c);
    (void)ac_ftn_i32(&c);
    int nsym = ac_ftn_i32(&c), nbf[20], nob[20];
    for (int i = 0; i < 20; i++) nbf[i] = ac_ftn_i32(&c);
    for (int i = 0; i < 20; i++) nob[i] = ac_ftn_i32(&c);
    if (nsym != b->nl) {
        fail(err, errlen, "%s: NSYM %d, basis LMAX %d", path, nsym, b->nl);
        goto out;
    }
    for (int l = 0; l < nsym; l++) {
        if (nbf[l] != b->count[l]) {
            fail(err, errlen, "%s: NBF(%d) = %d, basis has %d", path, l + 1, nbf[l], b->count[l]);
            goto out;
        }
        b->nob[l] = nob[l];
        long need = (long)nbf[l] * nob[l], got = 0;
        if (need == 0)
            continue;
        b->c[l] = malloc((size_t)need * sizeof(double));
        while (got < need) {
            if (ac_ftn_read(&f) != AC_FTN_OK) {
                fail(err, errlen, "%s: coefficients of l = %d truncated", path, l);
                goto out;
            }
            ac_ftn_cursor_init(&c, &f);
            int nt = ac_ftn_i32(&c);
            if (nt < 1 || got + nt > need || ac_ftn_left(&c) != (size_t)nt * 8) {
                fail(err, errlen, "%s: bad coefficient record", path);
                goto out;
            }
            for (int i = 0; i < nt; i++)
                b->c[l][got + i] = ac_ftn_f64(&c);
            got += nt;
        }
    }
    rc = 0;
out:
    ac_ftn_close(&f);
    return rc;
}

int ac_basis_load(ac_basis *b, const char *ao_path, const char *vec_path,
                  char *err, int errlen)
{
    memset(b, 0, sizeof *b);
    if (load_ao(b, ao_path, err, errlen) || load_vec(b, vec_path, err, errlen)) {
        ac_basis_free(b);
        return -1;
    }
    return 0;
}

void ac_basis_free(ac_basis *b)
{
    free(b->bf);
    for (int l = 0; l < AC_BASIS_MAXL; l++)
        free(b->c[l]);
    memset(b, 0, sizeof *b);
}
