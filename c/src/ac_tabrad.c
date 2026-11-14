/*
 * ac_tabrad.c -- ATSP numerical orbitals as an ac_radial (see ac_tabrad.h).
 */
#include "ac_tabrad.h"
#include "ac_ftnio.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ATSP_RHO0 (-4.0)
#define ATSP_H (1.0 / 16.0)

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

static int l_of(char c)
{
    static const char *sym = "spdfghiklmn";
    const char *p = strchr(sym, tolower((unsigned char)c));
    return p ? (int)(p - sym) : -1;
}

int ac_atspw_read(ac_atspw *w, const char *path, char *err, int errlen)
{
    memset(w, 0, sizeof *w);
    ac_ftn_file f;
    if (ac_ftn_open(&f, path, "rb"))
        return fail(err, errlen, "cannot open %s", path);
    int st;
    while ((st = ac_ftn_read(&f)) == AC_FTN_OK) {
        if (w->n == AC_ATSP_MAXORB) {
            ac_ftn_close(&f);
            return fail(err, errlen, "%s: more than %d orbitals", path, AC_ATSP_MAXORB);
        }
        if (f.len < 15 + 4 + 32) {
            ac_ftn_close(&f);
            return fail(err, errlen, "%s: short record", path);
        }
        int k = w->n;
        /* EL*3 at offset 12, e.g. " 2p" or "2p " */
        char el[4] = {0};
        memcpy(el, f.buf + 12, 3);
        int a = 0;
        for (int i = 0; i < 3; i++)
            if (el[i] != ' ')
                w->el[k][a++] = el[i];
        ac_ftn_cursor c;
        ac_ftn_cursor_init(&c, &f);
        c.pos = 15;
        int m = ac_ftn_i32(&c);
        double z = ac_ftn_f64(&c);
        w->e[k] = ac_ftn_f64(&c);
        (void)ac_ftn_f64(&c); /* EK */
        w->az[k] = ac_ftn_f64(&c);
        if (m < 8 || ac_ftn_left(&c) < (size_t)m * 8) {
            ac_ftn_close(&f);
            return fail(err, errlen, "%s: orbital %s has %d points", path, w->el[k], m);
        }
        w->m[k] = m;
        w->z = z;
        w->l[k] = a ? l_of(w->el[k][a - 1]) : -1;
        if (w->l[k] < 0) {
            ac_ftn_close(&f);
            return fail(err, errlen, "%s: bad orbital label '%s'", path, el);
        }
        /* the file holds ATSP's working variable P(r)/sqrt(r) */
        w->p[k] = malloc((size_t)m * sizeof(double));
        for (int j = 0; j < m; j++)
            w->p[k][j] = ac_ftn_f64(&c) * sqrt(exp(ATSP_RHO0 + j * ATSP_H) / z);
        w->n++;
    }
    ac_ftn_close(&f);
    if (st != AC_FTN_EOF || w->n == 0)
        return fail(err, errlen, "%s: not an ATSP .w file", path);
    return 0;
}

void ac_atspw_free(ac_atspw *w)
{
    for (int i = 0; i < w->n; i++)
        free(w->p[i]);
    memset(w, 0, sizeof *w);
}

double ac_atspw_r(const ac_atspw *w, int j)
{
    return exp(ATSP_RHO0 + j * ATSP_H) / w->z;
}

/* Lagrange interpolation of y[0..np-1] at nodes t[0..np-1]: value and the
   first two derivatives at s */
static void lagrange(const double *t, const double *y, int np, double s, double out[3])
{
    out[0] = out[1] = out[2] = 0.0;
    for (int i = 0; i < np; i++) {
        double den = 1.0;
        for (int j = 0; j < np; j++)
            if (j != i)
                den *= t[i] - t[j];
        /* L_i(s) = prod_{j != i} (s - t_j) / den, with derivatives */
        double v = 1.0, d1 = 0.0, d2 = 0.0;
        for (int j = 0; j < np; j++) {
            if (j == i)
                continue;
            double f = s - t[j];
            d2 = d2 * f + 2.0 * d1;
            d1 = d1 * f + v;
            v *= f;
        }
        out[0] += y[i] * v / den;
        out[1] += y[i] * d1 / den;
        out[2] += y[i] * d2 / den;
    }
}

int ac_radial_from_atsp(ac_radial *rad, const ac_atspw *w, const ac_rgrid *g, int *mo,
                        char *err, int errlen)
{
    memset(rad, 0, sizeof *rad);
    rad->g = g;
    rad->z = w->z;
    rad->nmo = w->n;
    int n = g->n, nmo = w->n;
    rad->l = malloc((size_t)nmo * sizeof(int));
    rad->idx = malloc((size_t)nmo * sizeof(int));
    rad->val = calloc((size_t)nmo * n, sizeof(double));
    rad->der = calloc((size_t)nmo * n, sizeof(double));
    rad->lap = calloc((size_t)nmo * n, sizeof(double));
    rad->r0 = calloc((size_t)nmo, sizeof(double));
    if (!rad->l || !rad->idx || !rad->val || !rad->der || !rad->lap || !rad->r0) {
        ac_radial_free(rad);
        return fail(err, errlen, "out of memory");
    }
    /* MO order: grouped by l, within l as in w */
    int k = 0, lmax = 0;
    for (int i = 0; i < nmo; i++)
        if (w->l[i] > lmax)
            lmax = w->l[i];
    if (lmax >= AC_BASIS_MAXL) {
        ac_radial_free(rad);
        return fail(err, errlen, "l = %d too large", lmax);
    }
    for (int l = 0; l <= lmax; l++) {
        rad->first[l] = k;
        int c = 0;
        for (int i = 0; i < nmo; i++)
            if (w->l[i] == l) {
                mo[i] = k;
                rad->l[k] = l;
                rad->idx[k] = c++;
                k++;
            }
    }
    for (int l = lmax + 1; l < AC_BASIS_MAXL; l++)
        rad->first[l] = k;

    for (int i = 0; i < nmo; i++) {
        int l = w->l[i], m = w->m[i], o = mo[i];
        double *q = malloc((size_t)m * sizeof(double)), *x = malloc((size_t)m * sizeof(double));
        double *rr = malloc((size_t)m * sizeof(double));
        for (int j = 0; j < m; j++) {
            rr[j] = ac_atspw_r(w, j);
            x[j] = log(rr[j]);
            q[j] = w->p[i][j] / pow(rr[j], l + 1);
        }
        /* inner polynomial in r: nodes 0, r_1..r_7 */
        double tin[8] = {0}, yin[8];
        yin[0] = w->az[i];
        for (int j = 0; j < 7; j++) {
            tin[j + 1] = rr[j];
            yin[j + 1] = q[j];
        }
        double *v = rad->val + (size_t)o * n, *d = rad->der + (size_t)o * n,
               *lp = rad->lap + (size_t)o * n;
        for (int p = 0; p < n; p++) {
            double r = g->r[p], xx = log(r), Q[3], Qx, Qxx;
            if (r > rr[m - 1])
                continue; /* beyond the table: 0 */
            if (r < rr[0]) {
                lagrange(tin, yin, 8, r, Q); /* Q, Q_r, Q_rr */
                Qx = r * Q[1];
                Qxx = r * Q[1] + r * r * Q[2];
            } else {
                int s = (int)floor((xx - x[0]) / ATSP_H) - 3;
                if (s < 0)
                    s = 0;
                if (s > m - 8)
                    s = m - 8;
                lagrange(x + s, q + s, 8, xx, Q); /* Q, Q_x, Q_xx */
                Qx = Q[1];
                Qxx = Q[2];
            }
            double rl = pow(r, l);
            v[p] = Q[0] * rl;
            d[p] = rl / r * (Qx + l * Q[0]);
            lp[p] = rl / (r * r) * (Qxx + (2 * l + 1) * Qx);
        }
        if (l == 0)
            rad->r0[o] = w->az[i];
        free(q);
        free(x);
        free(rr);
    }
    return 0;
}
