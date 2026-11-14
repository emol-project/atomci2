/*
 * ac_radial.c -- radial integrals by numerical quadrature (see ac_radial.h).
 *
 * The radial forms of the Breit-Pauli and property integrals are those of
 * the technical note (Secs. 4 to 6), derived there; the quadrature is this
 * program's own. Sources cited below by author and year:
 *   METECC (1993)    F. Sasaki et al., "Non-relativistic configuration
 *                    interaction calculations for many-electron atoms:
 *                    ATOMCI", in Methods and Techniques in Computational
 *                    Chemistry: METECC-94, E. Clementi (ed.), Cagliari (1993),
 *                    pp. 163-219, Eqs. (44), (45)
 *   Bethe-Salpeter (1957)  H. A. Bethe and E. E. Salpeter, Quantum Mechanics
 *                    of One- and Two-Electron Atoms, Springer (1957), sections
 *                    38-39
 *   Edmonds (1957)   A. R. Edmonds, Angular Momentum in Quantum Mechanics,
 *                    Princeton University Press (1957), Eq. (7.1.1)
 *   Racah (1942)     G. Racah, Phys. Rev. 62, 438 (1942)
 *   Pachucki (2004)  K. Pachucki, Phys. Rev. A 69, 052502 (2004), Eqs. (32),
 *                    (60), (62)
 *   Blume-Watson (1962), Glass-Hibbert (1978)  M. Blume and R. E. Watson,
 *                    Proc. R. Soc. A 270, 127 (1962); R. Glass and A. Hibbert,
 *                    Comput. Phys. Commun. 16, 19 (1978) (N^k, V^k; used here
 *                    only for checks, see ac_radial.h)
 */
#include "ac_radial.h"
#include "ac_wigner.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double ac_alpha = AC_ALPHA;

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

int ac_rgrid_init(ac_rgrid *g, double rmin, double rmax, double h)
{
    memset(g, 0, sizeof *g);
    g->x0 = log(rmin);
    g->h = h;
    g->n = (int)ceil((log(rmax) - g->x0) / h) + 1;
    if (g->n < 6)
        return -1;
    g->r = malloc((size_t)g->n * sizeof(double));
    g->w = malloc((size_t)g->n * sizeof(double));
    if (!g->r || !g->w) {
        ac_rgrid_free(g);
        return -1;
    }
    for (int i = 0; i < g->n; i++) {
        g->r[i] = exp(g->x0 + i * h);
        g->w[i] = h * g->r[i];
    }
    /* the integrands vanish at both ends, so the end weights need no halving */
    return 0;
}

void ac_rgrid_free(ac_rgrid *g)
{
    free(g->r);
    free(g->w);
    memset(g, 0, sizeof *g);
}

double ac_bfn_deriv(const ac_bfn *f, double r)
{
    double s = 0.0, t = 0.0;
    for (int a = 0; a < f->nprim; a++) {
        double e = f->gto ? exp(-f->alpha[a] * r * r) : exp(-f->alpha[a] * r);
        s += f->coef[a] * e;
        t += f->coef[a] * e * (f->gto ? -2.0 * f->alpha[a] * r : -f->alpha[a]);
    }
    /* d/dr [r^(n-1) s] = (n-1) r^(n-2) s + r^(n-1) s' */
    double v = pow(r, f->n - 1) * t;
    if (f->n > 1)
        v += (f->n - 1) * pow(r, f->n - 2) * s;
    return v;
}

int ac_radial_build(ac_radial *rad, const ac_basis *b, const ac_rgrid *g,
                    char *err, int errlen)
{
    memset(rad, 0, sizeof *rad);
    rad->g = g;
    rad->z = b->z;
    for (int l = 0; l < b->nl; l++) {
        rad->first[l] = rad->nmo;
        rad->nmo += b->nob[l];
    }
    int n = g->n, nmo = rad->nmo;
    rad->l = malloc((size_t)nmo * sizeof(int));
    rad->idx = malloc((size_t)nmo * sizeof(int));
    rad->val = calloc((size_t)nmo * n, sizeof(double));
    rad->der = calloc((size_t)nmo * n, sizeof(double));
    rad->lap = calloc((size_t)nmo * n, sizeof(double));
    rad->r0 = calloc((size_t)nmo, sizeof(double));
    double *bv = malloc((size_t)n * sizeof(double)), *bd = malloc((size_t)n * sizeof(double));
    double *bl = malloc((size_t)n * sizeof(double));
    if (!rad->l || !rad->idx || !rad->val || !rad->der || !rad->lap || !rad->r0 || !bv || !bd ||
        !bl) {
        free(bv);
        free(bd);
        free(bl);
        ac_radial_free(rad);
        return fail(err, errlen, "out of memory");
    }
    for (int l = 0; l < b->nl; l++) {
        for (int i = 0; i < b->nob[l]; i++) {
            rad->l[rad->first[l] + i] = l;
            rad->idx[rad->first[l] + i] = i;
        }
        for (int mu = 0; mu < b->count[l]; mu++) {
            const ac_bfn *f = &b->bf[b->first[l] + mu];
            for (int p = 0; p < n; p++) {
                bv[p] = ac_bfn_value(f, g->r[p]);
                bd[p] = ac_bfn_deriv(f, g->r[p]);
                bl[p] = ac_bfn_lap(f, g->r[p]);
            }
            double f0 = l == 0 && f->n == 1 ? ac_bfn_value(f, 0.0) : 0.0;
            for (int i = 0; i < b->nob[l]; i++) {
                double c = b->c[l][mu + (size_t)i * b->count[l]];
                double *v = rad->val + (size_t)(rad->first[l] + i) * n;
                double *d = rad->der + (size_t)(rad->first[l] + i) * n;
                double *x = rad->lap + (size_t)(rad->first[l] + i) * n;
                for (int p = 0; p < n; p++) {
                    v[p] += c * bv[p];
                    d[p] += c * bd[p];
                    x[p] += c * bl[p];
                }
                rad->r0[rad->first[l] + i] += c * f0;
            }
        }
    }
    free(bv);
    free(bd);
    free(bl);
    return 0;
}

void ac_radial_free(ac_radial *rad)
{
    free(rad->l);
    free(rad->idx);
    free(rad->val);
    free(rad->der);
    free(rad->lap);
    free(rad->r0);
    memset(rad, 0, sizeof *rad);
}

double ac_radial_moment(const ac_radial *rad, int a, int b, int p)
{
    const ac_rgrid *g = rad->g;
    const double *va = rad->val + (size_t)a * g->n, *vb = rad->val + (size_t)b * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * va[i] * vb[i] * pow(g->r[i], 2 + p);
    return s;
}

/* METECC (1993), Eq. (44), with the centrifugal term l(l+1)/(2 r^2) (the
   print has l(l+1)/r^2; technical note, Sec. 4.1) */
double ac_radial_h(const ac_radial *rad, int a, int b)
{
    const ac_rgrid *g = rad->g;
    int n = g->n, l = rad->l[a];
    const double *va = rad->val + (size_t)a * n, *vb = rad->val + (size_t)b * n;
    const double *da = rad->der + (size_t)a * n, *db = rad->der + (size_t)b * n;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double r = g->r[i];
        double t = 0.5 * (da[i] * db[i] * r * r + l * (l + 1) * va[i] * vb[i]);
        s += g->w[i] * (t - rad->z * va[i] * vb[i] * r);
    }
    return s;
}

/* Weights of the 6-point rule for one interval of the uniform grid: the
   integral over [p, p+1] of the degree-5 interpolant through the nodes
   0..5, in units of the step. wt[p][j], p = 0..4. */
static _Thread_local double wt[5][6];
static _Thread_local int wt_ready;

static void set_weights(void)
{
    for (int p = 0; p < 5; p++) {
        /* solve sum_j w_j j^m = ((p+1)^(m+1) - p^(m+1)) / (m+1), m = 0..5 */
        double a[6][7];
        for (int m = 0; m < 6; m++) {
            for (int j = 0; j < 6; j++)
                a[m][j] = pow(j, m);
            a[m][6] = (pow(p + 1, m + 1) - pow(p, m + 1)) / (m + 1);
        }
        for (int c = 0; c < 6; c++) {
            int piv = c;
            for (int r = c + 1; r < 6; r++)
                if (fabs(a[r][c]) > fabs(a[piv][c]))
                    piv = r;
            for (int k = 0; k < 7; k++) {
                double t = a[c][k];
                a[c][k] = a[piv][k];
                a[piv][k] = t;
            }
            for (int r = 0; r < 6; r++) {
                if (r == c)
                    continue;
                double f = a[r][c] / a[c][c];
                for (int k = c; k < 7; k++)
                    a[r][k] -= f * a[c][k];
            }
        }
        for (int j = 0; j < 6; j++)
            wt[p][j] = a[j][6] / a[j][j];
    }
    wt_ready = 1;
}

/* integral of f dx over [x_i, x_{i+1}] */
static double step(const double *f, int n, int i, double h)
{
    int j0 = i - 2;
    if (j0 < 0)
        j0 = 0;
    if (j0 > n - 6)
        j0 = n - 6;
    const double *w = wt[i - j0];
    double s = 0.0;
    for (int j = 0; j < 6; j++)
        s += w[j] * f[j0 + j];
    return s * h;
}

/* the Laplace expansion of 1/r12 (METECC (1993), Eq. (45); technical
   note, Sec. 4.1) */
void ac_radial_yk(const ac_radial *rad, int k, int a, int b, double *y)
{
    if (!wt_ready)
        set_weights();
    const ac_rgrid *g = rad->g;
    int n = g->n;
    const double *va = rad->val + (size_t)a * n, *vb = rad->val + (size_t)b * n;
    double *fa = malloc((size_t)n * sizeof(double)), *fb = malloc((size_t)n * sizeof(double));
    /* integrands in x: rho s^k ds = R_a R_b s^(k+3) dx, rho s^-(k+1) ds = R_a R_b s^(2-k) dx */
    for (int i = 0; i < n; i++) {
        double r = g->r[i], ab = va[i] * vb[i];
        fa[i] = ab * pow(r, k + 3);
        fb[i] = ab * pow(r, 2 - k);
    }
    /* outward A, stored scaled in y; inward B added */
    double acc = 0.0;
    y[0] = 0.0;
    for (int i = 1; i < n; i++) {
        acc += step(fa, n, i - 1, g->h);
        y[i] = acc * pow(g->r[i], -(k + 1));
    }
    acc = 0.0;
    for (int i = n - 2; i >= 0; i--) {
        acc += step(fb, n, i, g->h);
        y[i] += acc * pow(g->r[i], k);
    }
    free(fa);
    free(fb);
}

double ac_radial_dot(const ac_radial *rad, int c, int d, const double *y)
{
    const ac_rgrid *g = rad->g;
    const double *vc = rad->val + (size_t)c * g->n, *vd = rad->val + (size_t)d * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * vc[i] * vd[i] * g->r[i] * g->r[i] * y[i];
    return s;
}

double ac_radial_rk(const ac_radial *rad, int k, int a, int b, int c, int d)
{
    double *y = malloc((size_t)rad->g->n * sizeof(double));
    ac_radial_yk(rad, k, a, b, y);
    double s = ac_radial_dot(rad, c, d, y);
    free(y);
    return s;
}

/* outward and inward cumulative integrals of f dx: out[i] = int_0^{x_i},
   in[i] = int_{x_i}^inf */
static void cumulate(const ac_rgrid *g, const double *f, double *out, double *in)
{
    if (!wt_ready)
        set_weights();
    int n = g->n;
    if (out) {
        out[0] = 0.0;
        for (int i = 1; i < n; i++)
            out[i] = out[i - 1] + step(f, n, i - 1, g->h);
    }
    if (in) {
        in[n - 1] = 0.0;
        for (int i = n - 2; i >= 0; i--)
            in[i] = in[i + 1] + step(f, n, i, g->h);
    }
}

/* Blume-Watson (1962), Glass-Hibbert (1978); checks only */
double ac_radial_nk(const ac_radial *rad, int k, int i, int j, int ip, int jp)
{
    /* (alpha^2/4) int dr1 P_i P_i' r1^-(k+3) int_0^r1 dr2 P_j P_j' r2^k,
       P = r R; in x = ln r, dr = r dx */
    const ac_rgrid *g = rad->g;
    int n = g->n;
    const double *vi = rad->val + (size_t)i * n, *vj = rad->val + (size_t)j * n;
    const double *vip = rad->val + (size_t)ip * n, *vjp = rad->val + (size_t)jp * n;
    double *f = calloc((size_t)n, sizeof(double)), *a = malloc((size_t)n * sizeof(double));
    for (int p = 0; p < n; p++)
        f[p] = vj[p] * vjp[p] * pow(g->r[p], k + 3);
    cumulate(g, f, a, NULL);
    double s = 0.0;
    for (int p = 0; p < n; p++)
        s += g->w[p] * vi[p] * vip[p] * pow(g->r[p], -(k + 1)) * a[p];
    free(f);
    free(a);
    return 0.25 * ac_alpha * ac_alpha * s;
}

/* Blume-Watson (1962), Glass-Hibbert (1978); checks only */
double ac_radial_vk(const ac_radial *rad, int k, int i, int j, int ip, int jp)
{
    /* (alpha^2/4) int int P_i(1) P_j(2) r<^k/r>^(k+3) r2 (d/dr1 - 1/r1) P_i'(1) P_j'(2);
       (d/dr - 1/r) P = r dR/dr, so f1 = P_i r R_i'' = r^2 R_i dR_i'/dr and
       f2 = r2 P_j P_j' = r^3 R_j R_j' */
    const ac_rgrid *g = rad->g;
    int n = g->n;
    const double *vi = rad->val + (size_t)i * n, *vj = rad->val + (size_t)j * n;
    const double *dip = rad->der + (size_t)ip * n;
    const double *vjp = rad->val + (size_t)jp * n;
    double *f = calloc((size_t)n, sizeof(double)), *a = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    /* r2 < r1: r2^k f2 dr2 = r^(k+4) R R dx;  r2 > r1: r2^-(k+3) f2 dr2 = r^(1-k) R R dx */
    for (int p = 0; p < n; p++)
        f[p] = vj[p] * vjp[p] * pow(g->r[p], k + 4);
    cumulate(g, f, a, NULL);
    for (int p = 0; p < n; p++)
        f[p] = vj[p] * vjp[p] * pow(g->r[p], 1 - k);
    cumulate(g, f, NULL, b);
    double s = 0.0;
    for (int p = 0; p < n; p++) {
        double r = g->r[p];
        double f1 = r * r * vi[p] * dip[p];
        s += g->w[p] * f1 * (pow(r, -(k + 3)) * a[p] + pow(r, k) * b[p]);
    }
    free(f);
    free(a);
    free(b);
    return 0.25 * ac_alpha * ac_alpha * s;
}

/* the gradient formula (technical note, Sec. 4.2; Edmonds' reduced
   elements) */
static double grad_fn(const ac_radial *rad, int a, int s, int p)
{
    int l = rad->l[a], n = rad->g->n;
    double r = rad->g->r[p], R = rad->val[(size_t)a * n + p], d = rad->der[(size_t)a * n + p];
    if (s > 0)
        return -sqrt((l + 1.0) / (2 * l + 1)) * (d - l * R / r);
    return l ? sqrt(l / (2.0 * l + 1)) * (d + (l + 1) * R / r) : 0.0;
}

/* H_SO2 radial integral (technical note, Sec. 4.2; not the N^k/V^k
   form) */
double ac_radial_gdot(const ac_radial *rad, int a, int sa, int b, int sb, const double *y)
{
    const ac_rgrid *g = rad->g;
    double s = 0.0;
    for (int p = 0; p < g->n; p++)
        s += g->w[p] * g->r[p] * g->r[p] * grad_fn(rad, a, sa, p) * grad_fn(rad, b, sb, p) * y[p];
    return s;
}

/* H_SS: Edmonds (1957), Eq. (7.1.1) with the gradient elements (technical
   note, Sec. 4.2) */
void ac_radial_ssw(const ac_radial *rad, int k, int lam, int c, int d, double *w)
{
    const ac_rgrid *g = rad->g;
    int n = g->n;
    const double *vc = rad->val + (size_t)c * n, *vd = rad->val + (size_t)d * n;
    double *f = calloc((size_t)n, sizeof(double)), *a = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    /* A(r) = int_0^r rho s^k ds, B(r) = int_r^inf rho s^-(k+1) ds, rho = R R s^2 */
    for (int p = 0; p < n; p++)
        f[p] = vc[p] * vd[p] * pow(g->r[p], k + 3);
    cumulate(g, f, a, NULL);
    for (int p = 0; p < n; p++)
        f[p] = vc[p] * vd[p] * pow(g->r[p], 2 - k);
    cumulate(g, f, NULL, b);
    double ph = (lam + k) % 2 ? -sqrt(5.0) : sqrt(5.0);
    for (int p = 0; p < n; p++) {
        double r = g->r[p], rho_r2 = vc[p] * vd[p]; /* rho / r^2 */
        double v = 0.0;
        if (lam == k + 2) {
            /* j = k+1: D+_(k+1) D+_k Y */
            v = ac_sixj(2, 2, 4, 2 * k, 2 * lam, 2 * k + 2) * -sqrt((k + 1.0) * (k + 2)) *
                (2 * k + 1) * (rho_r2 - (2 * k + 3) * a[p] * pow(r, -(k + 3)));
        } else if (lam == k) {
            /* j = k+1: D-_(k+1) D+_k Y = (k+1)(2k+1) rho/r^2;
               j = k-1: D+_(k-1) D-_k Y = k (2k+1) rho/r^2 */
            v = (2 * k + 1) * rho_r2 *
                ((k + 1) * ac_sixj(2, 2, 4, 2 * k, 2 * lam, 2 * k + 2) +
                 (k >= 1 ? k * ac_sixj(2, 2, 4, 2 * k, 2 * lam, 2 * k - 2) : 0.0));
        } else if (lam == k - 2 && k >= 2) {
            /* j = k-1: D-_(k-1) D-_k Y */
            v = ac_sixj(2, 2, 4, 2 * k, 2 * lam, 2 * k - 2) * sqrt(k * (k - 1.0)) * (2 * k + 1) *
                ((2 * k - 1) * pow(r, k - 2) * b[p] - rho_r2);
        }
        w[p] = ph * v;
    }
    free(f);
    free(a);
    free(b);
}

/* D2 D1 f with D = c (d/dr + e/r), from f, f', f'' at r */
static double ladder2(double c1, double e1, double c2, double e2, double r, double f, double f1,
                      double f2)
{
    double h = c1 * (f1 + e1 * f / r), h1 = c1 * (f2 + e1 * f1 / r - e1 * f / (r * r));
    return c2 * (h1 + e2 * h / r);
}

/* the r12 kernel of the spin two-electron M1 terms (Pachucki (2004), Eqs.
   (60), (62)) */
void ac_radial_m1w(const ac_radial *rad, int k, int lam, int c, int d, double *u)
{
    const ac_rgrid *g = rad->g;
    int n = g->n;
    const double *vc = rad->val + (size_t)c * n, *vd = rad->val + (size_t)d * n;
    double *f = calloc((size_t)n, sizeof(double)), *a2 = malloc((size_t)n * sizeof(double));
    double *a0 = malloc((size_t)n * sizeof(double)), *b1 = malloc((size_t)n * sizeof(double));
    double *b3 = malloc((size_t)n * sizeof(double));
    /* in x = ln r (ds = s dx), rho = R_c R_d s^2:
       a2 = int_0^r rho s^(k+2) ds, a0 = int_0^r rho s^k ds,
       b1 = int_r^inf rho s^-(k+1) ds, b3 = int_r^inf rho s^-(k-1) ds */
    for (int p = 0; p < n; p++)
        f[p] = vc[p] * vd[p] * pow(g->r[p], k + 5);
    cumulate(g, f, a2, NULL);
    for (int p = 0; p < n; p++)
        f[p] = vc[p] * vd[p] * pow(g->r[p], k + 3);
    cumulate(g, f, a0, NULL);
    for (int p = 0; p < n; p++)
        f[p] = vc[p] * vd[p] * pow(g->r[p], 2 - k);
    cumulate(g, f, NULL, b1);
    for (int p = 0; p < n; p++)
        f[p] = vc[p] * vd[p] * pow(g->r[p], 4 - k);
    cumulate(g, f, NULL, b3);
    double p1 = 1.0 / (2 * k + 3), p2 = 1.0 / (2 * k - 1);
    double ph = (lam + k) % 2 ? -sqrt(5.0) : sqrt(5.0);
    double sp = ac_sixj(2, 2, 4, 2 * k, 2 * lam, 2 * k + 2);
    double sm = k >= 1 ? ac_sixj(2, 2, 4, 2 * k, 2 * lam, 2 * k - 2) : 0.0;
    double cp0 = -sqrt(k + 1.0), cp1 = -sqrt(k + 2.0), cm0 = sqrt((double)k),
           cm1 = sqrt(k + 1.0), cmm = sqrt(k - 1.0 > 0 ? k - 1.0 : 0.0),
           cpm = -sqrt((double)k);
    for (int p = 0; p < n; p++) {
        double r = g->r[p];
        /* Z = int rho r12-multipole, r12 = sum_k [r<^(k+2)/((2k+3) r>^(k+1))
           - r<^k/((2k-1) r>^(k-1))] C^k.C^k; Z' (the boundary terms cancel);
           Z'' from grad^2 (Z C^k) = 2 Y C^k, Y the 1/r12 multipole */
        double z = p1 * (pow(r, -(k + 1)) * a2[p] + pow(r, k + 2) * b1[p]) -
                   p2 * (pow(r, -(k - 1)) * a0[p] + pow(r, k) * b3[p]);
        double z1 = p1 * (-(k + 1) * pow(r, -(k + 2)) * a2[p] + (k + 2) * pow(r, k + 1) * b1[p]) -
                    p2 * (-(k - 1) * pow(r, -k) * a0[p] + k * pow(r, k - 1) * b3[p]);
        double y = pow(r, -(k + 1)) * a0[p] + pow(r, k) * b1[p];
        double z2 = 2.0 * y - 2.0 * z1 / r + k * (k + 1.0) * z / (r * r);
        /* <lam||[grad x grad]^(2)||k> Z with D+_j = -sqrt(j+1) (d/dr - j/r),
           D-_j = sqrt(j) (d/dr + (j+1)/r) (as ac_radial_ssw) */
        double v = 0.0;
        if (lam == k + 2)
            v = sp * ladder2(cp0, -k, cp1, -(k + 1.0), r, z, z1, z2);
        else if (lam == k) {
            v = sp * ladder2(cp0, -k, cm1, k + 2.0, r, z, z1, z2);
            if (k >= 1)
                v += sm * ladder2(cm0, k + 1.0, cpm, -(k - 1.0), r, z, z1, z2);
        } else if (lam == k - 2 && k >= 2)
            v = sm * ladder2(cm0, k + 1.0, cmm, (double)k, r, z, z1, z2);
        u[p] = ph * v;
    }
    free(f);
    free(a2);
    free(a0);
    free(b1);
    free(b3);
}

/* zeta, mass-velocity, Darwin: Bethe-Salpeter (1957), sections 38-39
   (technical note, Sec. 4.2) */
double ac_radial_zeta(const ac_radial *rad, int a, int b)
{
    return 0.5 * ac_alpha * ac_alpha * rad->z * ac_radial_moment(rad, a, b, -3);
}

double ac_radial_mv(const ac_radial *rad, int a, int b)
{
    const ac_rgrid *g = rad->g;
    const double *xa = rad->lap + (size_t)a * g->n, *xb = rad->lap + (size_t)b * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * xa[i] * xb[i] * g->r[i] * g->r[i];
    return -ac_alpha * ac_alpha / 8.0 * s;
}

double ac_radial_darwin(const ac_radial *rad, int a, int b)
{
    return ac_alpha * ac_alpha * rad->z / 8.0 * rad->r0[a] * rad->r0[b];
}

double ac_radial_rrc(const ac_radial *rad, int kind, int a, int b)
{
    if (kind == 0)
        return ac_radial_moment(rad, a, b, 1);
    const ac_rgrid *g = rad->g;
    const double *ra = rad->val + (size_t)a * g->n;
    const double *rb = rad->val + (size_t)b * g->n, *db = rad->der + (size_t)b * g->n;
    double f = kind == 1 ? -rad->l[b] : rad->l[b] + 1.0, s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * ra[i] * (db[i] + f * rb[i] / g->r[i]) * g->r[i] * g->r[i];
    return s;
}

/* the radial factor of [p x p]^(2) (Pachucki (2004), Eq. (62); Edmonds
   (1957), Eq. (7.1.1)) */
double ac_radial_dd(const ac_radial *rad, int a, int sa, int b, int sb)
{
    const ac_rgrid *g = rad->g;
    const double *ra = rad->val + (size_t)a * g->n, *da = rad->der + (size_t)a * g->n;
    const double *rb = rad->val + (size_t)b * g->n, *db = rad->der + (size_t)b * g->n;
    double fa = sa > 0 ? -rad->l[a] : rad->l[a] + 1.0;
    double fb = sb > 0 ? -rad->l[b] : rad->l[b] + 1.0, s = 0.0;
    for (int i = 0; i < g->n; i++) {
        double r = g->r[i];
        s += g->w[i] * (da[i] + fa * ra[i] / r) * (db[i] + fb * rb[i] / r) * r * r;
    }
    return s;
}

void ac_radial_kern(const ac_radial *rad, int p, int q, const double *u, double *z)
{
    const ac_rgrid *g = rad->g;
    int n = g->n;
    double *f = calloc((size_t)n, sizeof(double)), *a = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    /* in x = ln s, ds = s dx */
    for (int i = 0; i < n; i++)
        f[i] = u[i] * pow(g->r[i], 3 + p);
    cumulate(g, f, a, NULL);
    for (int i = 0; i < n; i++)
        f[i] = u[i] * pow(g->r[i], 3 - q);
    cumulate(g, f, NULL, b);
    for (int i = 0; i < n; i++)
        z[i] = pow(g->r[i], -q) * a[i] + pow(g->r[i], p) * b[i];
    free(f);
    free(a);
    free(b);
}

/* <l||C^k||l'> (Racah (1942)) */
static double cred_r(int l, int k, int lp)
{
    if ((l + k + lp) % 2 || k < abs(l - lp) || k > l + lp)
        return 0.0;
    int g = (l + k + lp) / 2;
    return ((l + g) % 2 ? -1.0 : 1.0) * sqrt((2.0 * l + 1) * (2.0 * lp + 1)) * ac_fabc(l, k, lp);
}

/* T_xy(r) = <l_x||[C^k x grad]^(g)||l_y> R_x(r) acting on R_y: (-1)^(lx+g+ly)
   sqrt[g] sum_lam {k 1 g; ly lx lam} <lx||C^k||lam> R_x <lam||grad||ly> R_y,
   with its angular factors computed once: c[0], c[1] multiply
   grad_fn(y, -1), grad_fn(y, +1); ph the phase sqrt[g] factor */
typedef struct {
    double c[2], ph;
} oo_coef;

static oo_coef oo_t_coef(const ac_radial *rad, int k, int g, int x, int y)
{
    int lx = rad->l[x], ly = rad->l[y];
    oo_coef o = {{0.0, 0.0}, ((lx + g + ly) % 2 ? -1.0 : 1.0) * sqrt(2.0 * g + 1)};
    for (int sy = -1; sy <= 1; sy += 2) {
        int lam = ly + sy;
        if (lam < 0)
            continue;
        double c = cred_r(lx, k, lam);
        if (c == 0.0)
            continue;
        o.c[sy > 0] = ac_sixj(2 * k, 2, 2 * g, 2 * ly, 2 * lx, 2 * lam) * c * sqrt(2.0 * ly + 1);
    }
    return o;
}

/* T_xy at grid point p, for the o of (k, g, x, y) */
static double oo_t_at(const ac_radial *rad, const oo_coef *o, int x, int y, int p)
{
    double s = 0.0;
    if (o->c[0] != 0.0)
        s += o->c[0] * grad_fn(rad, y, -1, p);
    if (o->c[1] != 0.0)
        s += o->c[1] * grad_fn(rad, y, 1, p);
    return o->ph * rad->val[(size_t)x * rad->g->n + p] * s;
}

void ac_radial_ooj(const ac_radial *rad, int k, int g, int x, int y, double *j)
{
    int lx = rad->l[x], ly = rad->l[y];
    double s = (k + 1 - g + ly - lx) % 2 ? -1.0 : 1.0;
    oo_coef oxy = oo_t_coef(rad, k, g, x, y), oyx = oo_t_coef(rad, k, g, y, x);
    for (int p = 0; p < rad->g->n; p++)
        j[p] = 0.5 * (oo_t_at(rad, &oxy, x, y, p) - s * oo_t_at(rad, &oyx, y, x, p));
}

void ac_radial_ooq(const ac_radial *rad, int x, int y, double *q)
{
    int n = rad->g->n;
    const double *vx = rad->val + (size_t)x * n, *vy = rad->val + (size_t)y * n;
    const double *lx = rad->lap + (size_t)x * n, *ly = rad->lap + (size_t)y * n;
    for (int p = 0; p < n; p++)
        q[p] = 0.5 * (vx[p] * ly[p] - vy[p] * lx[p]);
}

/* H_OO in the current/divergence form (technical note, Sec. 4.2) */
double ac_radial_oo(const ac_radial *rad, int k, int g, int a, int b, int c, int d)
{
    const ac_rgrid *gr = rad->g;
    int n = gr->n;
    double *u = malloc((size_t)n * sizeof(double)), *v = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double)), s = 0.0;
    if (g >= 0) {
        ac_radial_ooj(rad, k, g, a, b, u);
        ac_radial_ooj(rad, k, g, c, d, v);
        ac_radial_kern(rad, k, k + 1, u, z);
    } else {
        double *z2 = malloc((size_t)n * sizeof(double));
        ac_radial_ooq(rad, a, b, u);
        ac_radial_ooq(rad, c, d, v);
        ac_radial_kern(rad, k + 2, k + 1, u, z);
        ac_radial_kern(rad, k, k - 1, u, z2);
        for (int i = 0; i < n; i++)
            z[i] = z[i] / (2 * k + 3) - z2[i] / (2 * k - 1);
        free(z2);
    }
    for (int i = 0; i < n; i++)
        s += gr->w[i] * v[i] * z[i] * gr->r[i] * gr->r[i];
    free(u);
    free(v);
    free(z);
    return s;
}

/* the delta(r12) contact terms through R^k (technical note,
   Sec. 4.2) */
double ac_radial_contact(const ac_radial *rad, int k, int a, int b, int c, int d)
{
    const ac_rgrid *g = rad->g;
    int n = g->n;
    const double *va = rad->val + (size_t)a * n, *vb = rad->val + (size_t)b * n;
    const double *vc = rad->val + (size_t)c * n, *vd = rad->val + (size_t)d * n;
    double s = 0.0;
    for (int p = 0; p < n; p++)
        s += g->w[p] * va[p] * vb[p] * vc[p] * vd[p] * g->r[p] * g->r[p];
    return (2 * k + 1) * 0.25 * ac_alpha * ac_alpha * s;
}

/* <l_x||[C^k x [C^1 x grad]^(K')]^(g)||l_y> r R_x acting on R_y, at grid point p */
static double gc_q(const ac_radial *rad, int k, int kp, int g, int x, int y, int p)
{
    int lx = rad->l[x], ly = rad->l[y], n = rad->g->n;
    double s = 0.0;
    for (int lam = abs(lx - k); lam <= lx + k; lam++) {
        double c1 = cred_r(lx, k, lam);
        if (c1 == 0.0 || lam + kp < ly || abs(lam - kp) > ly)
            continue;
        double six = ac_sixj(2 * k, 2 * kp, 2 * g, 2 * ly, 2 * lx, 2 * lam);
        if (six == 0.0)
            continue;
        double inner = 0.0;
        for (int sy = -1; sy <= 1; sy += 2) {
            int mu = ly + sy;
            if (mu < 0)
                continue;
            double c2 = cred_r(lam, 1, mu);
            if (c2 == 0.0)
                continue;
            /* <mu||grad||l_y> R_y = -sqrt[l_y] G_mu: the sign opposite to D+- of
               ac_radial_ssw (which enters only squared there, see ac_radial.h) */
            inner -= ac_sixj(2, 2, 2 * kp, 2 * ly, 2 * lam, 2 * mu) * c2 * sqrt(2.0 * ly + 1) *
                     grad_fn(rad, y, sy, p);
        }
        s += six * c1 * ((lam + kp + ly) % 2 ? -1.0 : 1.0) * sqrt(2.0 * kp + 1) * inner;
    }
    return ((lx + g + ly) % 2 ? -1.0 : 1.0) * sqrt(2.0 * g + 1) * rad->g->r[p] *
           rad->val[(size_t)x * n + p] * s;
}

void ac_radial_gcj(const ac_radial *rad, int k, int kp, int g, int x, int y, double *e)
{
    int lx = rad->l[x], ly = rad->l[y];
    double s = (k - g + ly - lx) % 2 ? -1.0 : 1.0;
    for (int p = 0; p < rad->g->n; p++)
        e[p] = gc_q(rad, k, kp, g, x, y, p) + s * gc_q(rad, k, kp, g, y, x, p);
}

/* GC2e (cf. Pachucki (2004), Eq. (32); technical note, Sec. 5.1) */
double ac_radial_gc(const ac_radial *rad, int k, int kp, int g, int a, int b, int c, int d)
{
    const ac_rgrid *gr = rad->g;
    int n = gr->n;
    double *u = calloc((size_t)n, sizeof(double)), *z = malloc((size_t)n * sizeof(double));
    const double *vc = rad->val + (size_t)c * n, *vd = rad->val + (size_t)d * n;
    if (kp >= 0)
        ac_radial_gcj(rad, k, kp, g, a, b, u);
    else
        for (int i = 0; i < n; i++)
            u[i] = rad->val[(size_t)a * n + i] * rad->val[(size_t)b * n + i];
    ac_radial_kern(rad, k, k + 1, u, z);
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += gr->w[i] * vc[i] * vd[i] * z[i] * gr->r[i] * gr->r[i];
    free(u);
    free(z);
    return s;
}

/* OZ2e (cf. Pachucki (2004), Eq. (32); technical note, Sec. 5.1) */
double ac_radial_oz(const ac_radial *rad, int k, int g, int type, int a, int b, int c, int d)
{
    const ac_rgrid *gr = rad->g;
    int n = gr->n;
    double *u = calloc((size_t)n, sizeof(double)), *v = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double)), s = 0.0;
    const double *va = rad->val + (size_t)a * n, *vb = rad->val + (size_t)b * n;
    if (type == 0) {
        /* r R_a R_b (electron 1) against the current of cd (electron 2); the
           current function of ac_radial_ooj carries the overall sign of D+-,
           which is wrong (ac_radial.h): -j is the true reduced element */
        for (int i = 0; i < n; i++)
            u[i] = gr->r[i] * va[i] * vb[i];
        ac_radial_ooj(rad, k, g, c, d, v);
        for (int i = 0; i < n; i++)
            v[i] = -v[i];
        ac_radial_kern(rad, k, k + 1, u, z);
    } else {
        double *z2 = malloc((size_t)n * sizeof(double));
        ac_radial_gcj(rad, k, 1, g, a, b, u);
        ac_radial_ooq(rad, c, d, v);
        ac_radial_kern(rad, k + 2, k + 1, u, z);
        ac_radial_kern(rad, k, k - 1, u, z2);
        for (int i = 0; i < n; i++)
            z[i] = z[i] / (2 * k + 3) - z2[i] / (2 * k - 1);
        free(z2);
    }
    for (int i = 0; i < n; i++)
        s += gr->w[i] * v[i] * z[i] * gr->r[i] * gr->r[i];
    free(u);
    free(v);
    free(z);
    return s;
}
