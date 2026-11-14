/*
 * ac_hfsrel.c -- the regularized alpha^2 hfs correction A_AN/A^(4) of an S
 * state (see ac_hfsrel.h; technical note, Sec. 5.2).
 *
 * Every term is built from vectors O|psi> over the CSFs of the block:
 *   - one-body operators through ac_ci_one, spin rank 1 (w_a = s_a) or 0;
 *   - two-body operators with the spin on one electron, sum_{a != b}
 *     s_a f(r_a, r_b), through ac_ci_m1s2 (K = 0) with the multipole radial
 *     integrals of f in place of those of 1/r12: the separable terms
 *     (f = A(r_a) B(r_b), multipole 0 only) and N2e;
 *   - H^A_2e = pi delta(r_ab) (ac_ci_hij with the contact integrals only)
 *     + the orbit-orbit term (ac_ci_oo).
 * The spin-rank-1 quantities are reduced elements, all with the same
 * normalization, so their ratios to A4 are those of the technical
 * note (where w_a = sigma_az); the scalar ones are plain matrix elements.
 *
 * Large blocks: the CSFs are grouped by
 * configuration, and only configuration pairs within a single (one-body
 * operators) or double (two-body) excitation are evaluated. H is kept in
 * those blocks; the lowest root comes from Davidson's method and the
 * second order from preconditioned conjugate gradients on Q (H - E) Q.
 * Rows (configurations) are evaluated in parallel (OpenMP), each thread on
 * a shallow copy of the ac_ci with its own Theta cache; the radial caches
 * here are shared, NaN until computed, and read and written atomically.
 */
#include "ac_hfsrel.h"

#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define MAXK 13

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

/* AC_TIME: the wall time since the previous call, with a label, to stderr */
static void lap(const char *what)
{
    static double t0 = -1.0;
    double t = omp_get_wtime();
    if (getenv("AC_TIME") && t0 >= 0.0)
        fprintf(stderr, "hfsrel %s: %.2f s\n", what, t - t0);
    t0 = t;
}

/* ---- shared caches ---------------------------------------------------- */

static double ld(double *p)
{
    double v;
    __atomic_load(p, &v, __ATOMIC_RELAXED);
    return v;
}

static double st(double *p, double v)
{
    __atomic_store(p, &v, __ATOMIC_RELAXED);
    return v;
}

/* publish *slot = p unless another thread did; returns the winner */
static void *publish(void **slot, void *p)
{
    void *expect = NULL;
    if (__atomic_compare_exchange_n(slot, &expect, p, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        return p;
    free(p);
    return expect;
}

/* a NaN-filled page of n doubles at *slot, allocated when first asked */
static double *page(double **slot, size_t n)
{
    double *p = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (p)
        return p;
    p = malloc(n * sizeof *p);
    for (size_t i = 0; i < n; i++)
        p[i] = NAN;
    return publish((void **)slot, p);
}

/* slot matrices (nsl x nsl, slot pairs of equal l) */
enum { M_ID, M_V, M_VD, M_D, M_P4, M_KS, M_BM, M_HA, M_XP, M_M1, M_N };

typedef struct {
    const ac_radial *rad;
    const int *mo;
    int nsl;
    double z;
    double *m[M_N];
    const double *sa, *sb; /* the separable pair                           */
    double **ct;           /* [a*nsl+b] pages of nsl^2: int R R R R r^2 dr */
    double **n2;           /* [(k*nsl+a)*nsl+b] pages: N2e radial integrals */
    double **yk, **yd;     /* [(k*nsl+c)*nsl+d], c <= d: potentials        */
} hs;

/* R, R' (which = 1) or the radial Laplacian (2) of slot s (0-based) */
static const double *row(const hs *h, int s, int which)
{
    const double *base = which == 0 ? h->rad->val : which == 1 ? h->rad->der : h->rad->lap;
    return base + (size_t)h->mo[s] * h->rad->g->n;
}

static void build_matrices(hs *h)
{
    const ac_rgrid *g = h->rad->g;
    int n = g->n, ns = h->nsl;
    double z = h->z, pi = 4.0 * atan(1.0);
    for (int k = 0; k < M_N; k++)
        h->m[k] = calloc((size_t)ns * ns, sizeof(double));
    for (int a = 0; a < ns; a++)
        for (int b = 0; b < ns; b++) {
            int ia = h->mo[a], ib = h->mo[b];
            if (h->rad->l[ia] != h->rad->l[ib])
                continue;
            const double *va = row(h, a, 0), *vb = row(h, b, 0), *da = row(h, a, 1),
                         *db = row(h, b, 1), *la = row(h, a, 2), *lb = row(h, b, 2);
            double v = 0, vd = 0, p4 = 0, kk = 0;
            for (int p = 0; p < n; p++) {
                double r = g->r[p], w = g->w[p];
                v += w * va[p] * vb[p] * r;
                vd += w * va[p] * db[p];
                p4 += w * la[p] * lb[p] * r * r;
                /* the r^-1 parts of the two pieces cancel for cusp orbitals:
                   combined pointwise */
                kk += w * ((z / 6.0) * la[p] * lb[p] * r - (z / 3.0) * da[p] * lb[p]);
            }
            size_t ab = (size_t)a * ns + b;
            h->m[M_ID][ab] = a == b;
            h->m[M_V][ab] = z * v;
            h->m[M_VD][ab] = -z * vd; /* <a| V' d/dr |b> */
            h->m[M_D][ab] = h->rad->r0[ia] * h->rad->r0[ib] / (4.0 * pi);
            h->m[M_P4][ab] = p4;
            h->m[M_KS][ab] = kk; /* symmetrized below */
            h->m[M_M1][ab] = v;  /* int R_a R_b r dr */
        }
    for (int a = 0; a < ns; a++)
        for (int b = 0; b < a; b++) {
            double s = 0.5 * (h->m[M_KS][(size_t)a * ns + b] + h->m[M_KS][(size_t)b * ns + a]);
            h->m[M_KS][(size_t)a * ns + b] = h->m[M_KS][(size_t)b * ns + a] = s;
        }
    for (size_t ab = 0; ab < (size_t)ns * ns; ab++) {
        double p4 = h->m[M_P4][ab], d = h->m[M_D][ab], vd = h->m[M_VD][ab];
        h->m[M_BM][ab] = -p4 / 8.0 + z * pi * d - 0.25 * vd; /* H'^A, one-body */
        h->m[M_HA][ab] = -p4 / 8.0 + z * pi / 2.0 * d;     /* H^A, one-body  */
        h->m[M_XP][ab] = 2.0 / 3.0 * vd;                    /* X'             */
    }
}

/* one-body f of ac_ci_one: matrix element times the unit-operator factor */
typedef struct {
    const hs *h;
    const double *M;
    int spin;
} onef;

static double one_f(void *p, int a, int b, int l)
{
    const onef *o = p;
    double v = o->M[(size_t)(a - 1) * o->h->nsl + b - 1];
    if (v == 0.0)
        return 0.0;
    return v * (o->spin ? ac_ci_one_coef(sqrt(1.5), sqrt(2.0 * l + 1), 2, 0)
                        : ac_ci_one_coef(sqrt(2.0), sqrt(2.0 * l + 1), 0, 0));
}

/* two-body radial callbacks (ctx = hs) */
static double rm1_sep(void *p, int k, int lam, int a, int b, int c, int d)
{
    const hs *h = p;
    (void)lam;
    if (k)
        return 0.0;
    return h->sa[(size_t)(a - 1) * h->nsl + b - 1] * h->sb[(size_t)(c - 1) * h->nsl + d - 1];
}

/* Y^k of the pair density R_c R_d, or (deriv) the potential of (R_c R_d)'/r^2;
   both symmetric in c, d (cached at c <= d) */
static const double *pot(hs *h, int deriv, int k, int c, int d)
{
    if (c > d) {
        int t = c;
        c = d;
        d = t;
    }
    double **slot = (deriv ? h->yd : h->yk) + ((size_t)k * h->nsl + c) * h->nsl + d;
    double *y = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (y)
        return y;
    const ac_rgrid *g = h->rad->g;
    int n = g->n;
    y = malloc((size_t)n * sizeof *y);
    if (!deriv)
        ac_radial_yk(h->rad, k, h->mo[c], h->mo[d], y);
    else {
        double *u = malloc((size_t)n * sizeof *u);
        const double *vc = row(h, c, 0), *vd = row(h, d, 0), *dc = row(h, c, 1),
                     *dd = row(h, d, 1);
        for (int p = 0; p < n; p++)
            u[p] = (dc[p] * vd[p] + vc[p] * dd[p]) / (g->r[p] * g->r[p]);
        ac_radial_kern(h->rad, k, k + 1, u, y);
        free(u);
    }
    return publish((void **)slot, y);
}

/* N2e (the two-electron first-order term of A_AN, from Puchalski and
   Pachucki, arXiv:1310.2161v2)
   = (Z/6) sum_{a != b} s_a (r_ab/r_ab^3).(2 r_b/r_b^3 - r_a/r_a^3), by parts
   (the pair density rho and D rho = rho'/r^2 + 4 pi delta rho(0)):
   (Z/6) [-2 C(rho_ab, D rho_cd) - C(D rho_ab, rho_cd)] per multipole */
static double rm1_n2e(void *p, int k, int lam, int a, int b, int c, int d)
{
    hs *h = p;
    (void)lam;
    if (k >= MAXK)
        return 0.0;
    int ns = h->nsl;
    a--, b--, c--, d--;
    double *v = page(h->n2 + ((size_t)k * ns + a) * ns + b, (size_t)ns * ns) + (size_t)c * ns + d;
    double r = ld(v);
    if (!isnan(r))
        return r;
    const ac_rgrid *g = h->rad->g;
    const double *yd = pot(h, 1, k, c, d), *yk = pot(h, 0, k, c, d);
    const double *va = row(h, a, 0), *vb = row(h, b, 0), *da = row(h, a, 1), *db = row(h, b, 1);
    double i1 = 0.0, i2 = 0.0;
    for (int q = 0; q < g->n; q++) {
        double rr = g->r[q];
        i1 += g->w[q] * va[q] * vb[q] * rr * rr * yd[q];
        i2 += g->w[q] * (da[q] * vb[q] + va[q] * db[q]) * yk[q];
    }
    double s = -2.0 * i1 - i2;
    if (k == 0) { /* the point parts of D rho (s-s pair densities) */
        const double *m1 = h->m[M_M1];
        double r00ab = h->rad->r0[h->mo[a]] * h->rad->r0[h->mo[b]];
        double r00cd = h->rad->r0[h->mo[c]] * h->rad->r0[h->mo[d]];
        s += -2.0 * m1[(size_t)a * ns + b] * r00cd - r00ab * m1[(size_t)c * ns + d];
    }
    return st(v, h->z / 6.0 * s);
}

/* pi delta(r12) through ac_ci_hij: R^k replaced by the contact integrals */
static double h_zero(void *p, int a, int b)
{
    (void)p, (void)a, (void)b;
    return 0.0;
}

static double rk_del(void *p, int k, int a, int b, int c, int d)
{
    hs *h = p;
    int ns = h->nsl;
    a--, b--, c--, d--;
    double *v = page(h->ct + (size_t)a * ns + b, (size_t)ns * ns) + (size_t)c * ns + d;
    double r = ld(v);
    if (isnan(r)) {
        const ac_rgrid *g = h->rad->g;
        const double *va = row(h, a, 0), *vb = row(h, b, 0), *vc = row(h, c, 0),
                     *vd = row(h, d, 0);
        double s = 0.0;
        for (int q = 0; q < g->n; q++)
            s += g->w[q] * va[q] * vb[q] * vc[q] * vd[q] * g->r[q] * g->r[q];
        r = st(v, s);
    }
    return (2 * k + 1) * 0.25 * r;
}

/* ---- the block: configurations and their pairs ------------------------ */

#define MAXOCC 16

typedef struct {
    int no, orb[MAXOCC], occ[MAXOCC]; /* occupied orbitals (ascending), counts */
} occ_t;

typedef struct {
    int h, level;
    size_t off; /* the H block: n_g x n_h, row-major, at H + off */
} nbr_t;

typedef struct {
    int m, ng;
    int *csf;  /* [m] CSF of ci, in configuration order */
    int *gs;   /* [ng + 1] first row of each configuration */
    occ_t *oc; /* [ng] */
    int *nn;   /* [ng] neighbours (level <= 2, itself included) */
    nbr_t **nb;
    double *H; /* the blocks */
    size_t nh;
} blk;

static void occ_of(const ac_ci *ci, int i, occ_t *o)
{
    const ac_ci_csf *c = &ci->csf[i];
    int tmp[AC_ADDR_MAXORB + 1] = {0};
    for (int s = 1; s <= c->c.nsh; s++)
        tmp[c->seq[s]] += c->c.nel[s];
    o->no = 0;
    for (int s = 1; s <= ci->nwt; s++)
        if (tmp[s] && o->no < MAXOCC)
            o->orb[o->no] = s, o->occ[o->no++] = tmp[s];
}

/* sum |n_A - n_B| / 2 */
static int level(const occ_t *a, const occ_t *b)
{
    int i = 0, j = 0, d = 0;
    while (i < a->no || j < b->no) {
        if (j == b->no || (i < a->no && a->orb[i] < b->orb[j]))
            d += a->occ[i++];
        else if (i == a->no || b->orb[j] < a->orb[i])
            d += b->occ[j++];
        else
            d += abs(a->occ[i++] - b->occ[j++]);
    }
    return d / 2;
}

static int occ_cmp(const occ_t *a, const occ_t *b)
{
    if (a->no != b->no)
        return a->no - b->no;
    for (int i = 0; i < a->no; i++) {
        if (a->orb[i] != b->orb[i])
            return a->orb[i] - b->orb[i];
        if (a->occ[i] != b->occ[i])
            return a->occ[i] - b->occ[i];
    }
    return 0;
}

static const ac_ci *sort_ci;
static int csf_cmp(const void *x, const void *y)
{
    int i = *(const int *)x, j = *(const int *)y;
    occ_t a, b;
    occ_of(sort_ci, i, &a);
    occ_of(sort_ci, j, &b);
    int c = occ_cmp(&a, &b);
    return c ? c : i - j;
}

static void blk_free(blk *b)
{
    free(b->csf);
    free(b->gs);
    free(b->oc);
    if (b->nb)
        for (int g = 0; g < b->ng; g++)
            free(b->nb[g]);
    free(b->nb);
    free(b->nn);
    free(b->H);
    memset(b, 0, sizeof *b);
}

static int blk_build(const ac_ci *ci, const int *idx, int m, blk *b)
{
    memset(b, 0, sizeof *b);
    b->m = m;
    b->csf = malloc((size_t)m * sizeof(int));
    memcpy(b->csf, idx, (size_t)m * sizeof(int));
    sort_ci = ci;
    qsort(b->csf, (size_t)m, sizeof(int), csf_cmp);
    b->gs = malloc((size_t)(m + 1) * sizeof(int));
    b->oc = malloc((size_t)m * sizeof(occ_t));
    occ_t prev = {0};
    for (int i = 0; i < m; i++) {
        occ_t o;
        occ_of(ci, b->csf[i], &o);
        if (i == 0 || occ_cmp(&o, &prev)) {
            b->gs[b->ng] = i;
            b->oc[b->ng++] = o;
            prev = o;
        }
    }
    b->gs[b->ng] = m;
    int ng = b->ng;
    b->nn = calloc((size_t)ng, sizeof(int));
    b->nb = calloc((size_t)ng, sizeof(nbr_t *));
#pragma omp parallel for schedule(dynamic, 16)
    for (int g = 0; g < ng; g++) {
        int cap = 64, n = 0;
        nbr_t *v = malloc((size_t)cap * sizeof *v);
        for (int h = 0; h < ng; h++) {
            int lv = level(&b->oc[g], &b->oc[h]);
            if (lv > 2)
                continue;
            if (n == cap)
                v = realloc(v, (size_t)(cap *= 2) * sizeof *v);
            v[n++] = (nbr_t){h, lv, 0};
        }
        b->nn[g] = n;
        b->nb[g] = v;
    }
    size_t off = 0;
    for (int g = 0; g < ng; g++)
        for (int k = 0; k < b->nn[g]; k++) {
            b->nb[g][k].off = off;
            int h = b->nb[g][k].h;
            off += (size_t)(b->gs[g + 1] - b->gs[g]) * (b->gs[h + 1] - b->gs[h]);
        }
    b->nh = off;
    return 0;
}

/* thread-private copies of ci (own Theta cache) */
static ac_ci *copies(const ac_ci *ci, int *nt)
{
    *nt = omp_get_max_threads();
    ac_ci *c = malloc((size_t)*nt * sizeof *c);
    for (int t = 0; t < *nt; t++) {
        c[t] = *ci;
        c[t].thc = NULL;
    }
    return c;
}

static void copies_free(ac_ci *c, int nt)
{
    for (int t = 0; t < nt; t++)
        ac_ci_free_cache(&c[t]);
    free(c);
}

/* H in the blocks (level <= 2) with the caller's integrals */
static int blk_fill_h(ac_ci *ci, blk *b)
{
    b->H = malloc((b->nh ? b->nh : 1) * sizeof(double));
    if (!b->H)
        return -1;
    int nt, bad = 0;
    ac_ci *cc = copies(ci, &nt);
#pragma omp parallel for schedule(dynamic, 4)
    for (int g = 0; g < b->ng; g++) {
        ac_ci *c = &cc[omp_get_thread_num()];
        for (int k = 0; k < b->nn[g]; k++) {
            const nbr_t *q = &b->nb[g][k];
            int h = q->h, nh = b->gs[h + 1] - b->gs[h];
            for (int i = b->gs[g]; i < b->gs[g + 1]; i++)
                for (int j = b->gs[h]; j < b->gs[h + 1]; j++) {
                    int e = 0;
                    double v = ac_ci_hij(c, b->csf[i], b->csf[j], &e);
                    if (e)
                        bad = 1;
                    b->H[q->off + (size_t)(i - b->gs[g]) * nh + (j - b->gs[h])] = v;
                }
        }
    }
    copies_free(cc, nt);
    return bad ? -1 : 0;
}

static void blk_mul(const blk *b, const double *x, double *y)
{
#pragma omp parallel for schedule(dynamic, 8)
    for (int g = 0; g < b->ng; g++) {
        int n0 = b->gs[g], ngr = b->gs[g + 1] - n0;
        for (int i = 0; i < ngr; i++)
            y[n0 + i] = 0.0;
        for (int k = 0; k < b->nn[g]; k++) {
            const nbr_t *q = &b->nb[g][k];
            int h0 = b->gs[q->h], nh = b->gs[q->h + 1] - h0;
            const double *M = b->H + q->off;
            for (int i = 0; i < ngr; i++) {
                double s = 0.0;
                for (int j = 0; j < nh; j++)
                    s += M[(size_t)i * nh + j] * x[h0 + j];
                y[n0 + i] += s;
            }
        }
    }
}

static double dot(int n, const double *a, const double *b)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

static double blk_diag(const blk *b, int i)
{
    int g = 0;
    while (b->gs[g + 1] <= i)
        g++;
    for (int k = 0; k < b->nn[g]; k++)
        if (b->nb[g][k].h == g) {
            int n = b->gs[g + 1] - b->gs[g], r = i - b->gs[g];
            return b->H[b->nb[g][k].off + (size_t)r * n + r];
        }
    return 0.0;
}

/* the lowest root by Davidson's method (E. R. Davidson, J. Comput. Phys. 17,
   87 (1975); diagonal preconditioner), from the
   lowest eigenvector of the configuration with the lowest diagonal element */
static int davidson(const blk *b, const double *diag, double *e, double *c, int *niter)
{
    int m = b->m, maxsub = 30;
    double *V = malloc((size_t)maxsub * m * sizeof(double)), *AV = malloc((size_t)maxsub * m * sizeof(double));
    double *Hs = malloc((size_t)maxsub * maxsub * sizeof(double)), *w = malloc((size_t)maxsub * sizeof(double));
    double *r = malloc((size_t)m * sizeof(double));
    int lo = 0;
    for (int i = 1; i < m; i++)
        if (diag[i] < diag[lo])
            lo = i;
    memset(V, 0, (size_t)m * sizeof(double));
    V[lo] = 1.0;
    int nv = 1, rc = 1;
    double theta = 0.0;
    for (int it = 0; it < 500; it++) {
        blk_mul(b, V + (size_t)(nv - 1) * m, AV + (size_t)(nv - 1) * m);
        for (int i = 0; i < nv; i++)
            for (int j = 0; j <= i; j++)
                Hs[(size_t)j * nv + i] = Hs[(size_t)i * nv + j] =
                    0.5 * (dot(m, V + (size_t)i * m, AV + (size_t)j * m) +
                           dot(m, V + (size_t)j * m, AV + (size_t)i * m));
        if (ac_ci_eigvecs(nv, Hs, w))
            break;
        theta = w[0];
        for (int i = 0; i < m; i++) {
            double x = 0.0, ax = 0.0;
            for (int j = 0; j < nv; j++) {
                x += Hs[j] * V[(size_t)j * m + i];
                ax += Hs[j] * AV[(size_t)j * m + i];
            }
            c[i] = x;
            r[i] = ax - theta * x;
        }
        double rn = sqrt(dot(m, r, r));
        *niter = it + 1;
        if (rn < 1e-9) {
            rc = 0;
            break;
        }
        if (nv == maxsub) { /* restart from the Ritz vector */
            double *ac = AV;
            for (int i = 0; i < m; i++) {
                double ax = 0.0;
                for (int j = 0; j < nv; j++)
                    ax += Hs[j] * AV[(size_t)j * m + i];
                r[i] = ax;
            }
            memcpy(V, c, (size_t)m * sizeof(double));
            memcpy(ac, r, (size_t)m * sizeof(double));
            nv = 1;
            for (int i = 0; i < m; i++)
                r[i] = ac[i] - theta * c[i];
        }
        double *x = V + (size_t)nv * m;
        for (int i = 0; i < m; i++) {
            double d = diag[i] - theta;
            x[i] = -r[i] / (fabs(d) < 1e-2 ? (d < 0 ? -1e-2 : 1e-2) : d);
        }
        for (int pass = 0; pass < 2; pass++)
            for (int j = 0; j < nv; j++) {
                double s = dot(m, V + (size_t)j * m, x);
                for (int i = 0; i < m; i++)
                    x[i] -= s * V[(size_t)j * m + i];
            }
        double nx = sqrt(dot(m, x, x));
        if (nx < 1e-14)
            break;
        for (int i = 0; i < m; i++)
            x[i] /= nx;
        nv++;
    }
    *e = theta;
    free(V);
    free(AV);
    free(Hs);
    free(w);
    free(r);
    return rc;
}

/* z with Q (H - E) Q z = Q y, Q = 1 - |c><c| (preconditioned CG) */
static int solve(const blk *b, const double *diag, double e, const double *c, const double *y,
                 double *z, int *niter)
{
    int m = b->m;
    double *r = malloc((size_t)m * sizeof(double)), *p = malloc((size_t)m * sizeof(double));
    double *q = malloc((size_t)m * sizeof(double)), *u = malloc((size_t)m * sizeof(double));
#define PROJ(v)                                                                                    \
    do {                                                                                           \
        double s_ = dot(m, c, v);                                                                  \
        for (int i_ = 0; i_ < m; i_++)                                                             \
            (v)[i_] -= s_ * c[i_];                                                                 \
    } while (0)
    memcpy(r, y, (size_t)m * sizeof(double));
    PROJ(r);
    double bn = sqrt(dot(m, r, r));
    memset(z, 0, (size_t)m * sizeof(double));
    for (int i = 0; i < m; i++)
        u[i] = r[i] / fmax(diag[i] - e, 5e-2);
    PROJ(u);
    memcpy(p, u, (size_t)m * sizeof(double));
    double ru = dot(m, r, u);
    int rc = 1;
    for (int it = 0; it < 2000; it++) {
        blk_mul(b, p, q);
        for (int i = 0; i < m; i++)
            q[i] -= e * p[i];
        PROJ(q);
        double al = ru / dot(m, p, q);
        for (int i = 0; i < m; i++) {
            z[i] += al * p[i];
            r[i] -= al * q[i];
        }
        *niter = it + 1;
        if (sqrt(dot(m, r, r)) < 1e-11 * bn) {
            rc = 0;
            break;
        }
        for (int i = 0; i < m; i++)
            u[i] = r[i] / fmax(diag[i] - e, 5e-2);
        PROJ(u);
        double ru2 = dot(m, r, u);
        for (int i = 0; i < m; i++)
            p[i] = u[i] + ru2 / ru * p[i];
        ru = ru2;
    }
#undef PROJ
    free(r);
    free(p);
    free(q);
    free(u);
    return rc;
}

/* ---- operator vectors --------------------------------------------------- */

double ac_hfsrel_delta = NAN;

enum { OP_S1, OP_P1, OP_S2, OP_DEL, OP_OO };

/* out[K] = sum_i <K|O|i> c_i (block order) over configuration pairs within a
   single (one-body) or double excitation */
static int vec(const ac_ci *ci0, hs *h, const blk *b, const double *c, int op, int which,
               double *out)
{
    ac_ci base = *ci0;
    onef of = {h, which >= 0 ? h->m[which] : NULL, op == OP_S1};
    double a2 = ac_alpha * ac_alpha;
    if (op == OP_S2) {
        base.in.rm1 = which == -1 ? rm1_n2e : rm1_sep;
        base.in.zctx = h;
    }
    else if (op == OP_DEL) {
        base.in.h = h_zero;
        base.in.rk = rk_del;
        base.in.zeta = NULL;
        base.in.ctx = h;
    }
    int maxlv = op == OP_S1 || op == OP_P1 ? 1 : 2, nt, bad = 0;
    double t0 = omp_get_wtime();
    ac_ci *cc = copies(&base, &nt);
#pragma omp parallel for schedule(dynamic, 4)
    for (int g = 0; g < b->ng; g++) {
        ac_ci *cx = &cc[omp_get_thread_num()];
        for (int i = b->gs[g]; i < b->gs[g + 1]; i++) {
            double s = 0.0;
            for (int k = 0; k < b->nn[g]; k++) {
                const nbr_t *q = &b->nb[g][k];
                if (q->level > maxlv)
                    continue;
                for (int j = b->gs[q->h]; j < b->gs[q->h + 1]; j++) {
                    if (c[j] == 0.0)
                        continue;
                    int e = 0, K = b->csf[i], J = b->csf[j];
                    double x;
                    switch (op) {
                    case OP_S1:
                    case OP_P1:
                        x = ac_ci_one(cx, K, J, op == OP_S1 ? 2 : 0, 0, one_f, &of, &e);
                        break;
                    case OP_S2:
                        x = ac_ci_m1s2(cx, K, J, 0, &e) / (-2.0 / 3.0 * a2);
                        break;
                    case OP_DEL:
                        x = ac_ci_hij(cx, K, J, &e);
                        break;
                    default:
                        x = ac_ci_oo(cx, K, J, &e) / a2;
                    }
                    if (e)
                        bad = 1;
                    s += x * c[j];
                }
            }
            out[i] = s;
        }
    }
    copies_free(cc, nt);
    if (getenv("AC_TIME"))
        fprintf(stderr, "hfsrel   vector op %d (%d): %.2f s\n", op, which, omp_get_wtime() - t0);
    return bad ? -1 : 0;
}

/* A_AN of Puchalski and Pachucki (arXiv:1310.2161v2, Eq. (29)), regularized
   with their Eqs. (33), (34); the anomaly +Z^2 A4 is derived in
   the technical note, Sec. 5.2 */
static int block(FILE *out, ac_ci *ci, hs *h, const int *idx, int m, int ts, ac_hfsrel_res *r,
                 char *err, int errlen)
{
    blk b;
    blk_build(ci, idx, m, &b);
    lap("configuration pairs");
    double *v[16], *dg = malloc((size_t)m * sizeof *dg), *c = malloc((size_t)m * sizeof *c);
    for (int k = 0; k < 16; k++)
        v[k] = malloc((size_t)m * sizeof(double));
    int rc = 0, nd = 0, ncg = 0;
    double e0 = 0.0;
    if (blk_fill_h(ci, &b))
        rc = fail(err, errlen, "hfsrel: engine error (H) or out of memory (%zu elements)", b.nh);
    lap("H");
    if (!rc) {
        for (int i = 0; i < m; i++)
            dg[i] = blk_diag(&b, i);
        if (davidson(&b, dg, &e0, c, &nd))
            rc = fail(err, errlen, "hfsrel: Davidson not converged");
    }
    lap("Davidson");
    double z = h->z, pi = 4.0 * atan(1.0);
    enum { V_ID, V_X, V_F, V_D, V_K, V_SEP, V_N2, V_PID, V_BM, V_HA, V_G, V_DEL, V_OO, V_H2, V_Y, V_Z };
    static const double sepc[4] = {1.0 / 6.0, 0.0, 1.0 / 3.0, 1.0 / 3.0};
    static const int sepm[4][2] = {{M_V, M_P4}, {M_V, M_D}, {M_VD, M_V}, {M_V, M_VD}};
    double sep = 0.0;
    if (!rc)
        rc = vec(ci, h, &b, c, OP_S1, M_ID, v[V_ID]) || vec(ci, h, &b, c, OP_S1, M_XP, v[V_X]) ||
             vec(ci, h, &b, c, OP_S1, M_V, v[V_F]) || vec(ci, h, &b, c, OP_S1, M_D, v[V_D]) ||
             vec(ci, h, &b, c, OP_S1, M_KS, v[V_K]) || vec(ci, h, &b, c, OP_S2, -1, v[V_N2]) ||
             vec(ci, h, &b, c, OP_P1, M_ID, v[V_PID]) || vec(ci, h, &b, c, OP_P1, M_BM, v[V_BM]) ||
             vec(ci, h, &b, c, OP_P1, M_HA, v[V_HA]) || vec(ci, h, &b, c, OP_P1, M_V, v[V_G]) ||
             vec(ci, h, &b, c, OP_DEL, -1, v[V_DEL]) || vec(ci, h, &b, c, OP_OO, -1, v[V_OO]);
    for (int p = 0; p < 4 && !rc; p++) {
        h->sa = h->m[sepm[p][0]];
        h->sb = h->m[sepm[p][1]];
        rc = vec(ci, h, &b, c, OP_S2, 0, v[V_SEP]);
        sep += (p == 1 ? -4.0 / 3.0 * pi * z : sepc[p]) * dot(m, c, v[V_SEP]);
    }
    lap("operator vectors");
    if (rc) {
        if (err && !err[0])
            fail(err, errlen, "hfsrel: engine error");
    }
    else {
        /* plain scalar elements: ac_ci_one's scalar is sqrt([S][L]) times; the
           spin normalization from <sum s_az> = S at M = S */
        int nel = 0;
        for (int s = 1; s <= ci->csf[idx[0]].c.nsh; s++)
            nel += ci->csf[idx[0]].c.nel[s];
        double nf = dot(m, c, v[V_PID]) / nel, szf = 0.5 * ts / dot(m, c, v[V_ID]);
        for (int k = V_PID; k <= V_G; k++)
            for (int i = 0; i < m; i++)
                v[k][i] /= nf;
        double *h2 = v[V_H2], *y = v[V_Y];
        for (int i = 0; i < m; i++) {
            h2[i] = v[V_DEL][i] + v[V_OO][i];
            y[i] = v[V_BM][i] + h2[i]; /* H'^A psi */
        }
        double a4 = 4.0 * pi * z / 3.0 * dot(m, c, v[V_D]);
        double ha = dot(m, c, v[V_HA]) + dot(m, c, h2), hap = dot(m, c, y);
        double g = 0.25 * dot(m, c, v[V_G]), fm = 2.0 / 3.0 * dot(m, c, v[V_F]);
        /* second order: 2 <X'psi|G' H'^A psi>, G' = (E - H)^-1 on Q: z solves
           Q (H - E) Q z = Q H'^A psi, so the term is -2 <X'psi|z> */
        if (solve(&b, dg, e0, c, y, v[V_Z], &ncg))
            rc = fail(err, errlen, "hfsrel: CG not converged");
        lap("second order");
        double s2 = -2.0 * dot(m, v[V_X], v[V_Z]);
        memset(r, 0, sizeof *r);
        r->ts = ts;
        r->ncsf = m;
        r->e = e0;
        r->a4pp = 4.0 * szf * a4;
        r->ha = ha;
        r->t[AC_HR_K] = dot(m, c, v[V_K]);
        r->t[AC_HR_SEP] = sep;
        r->t[AC_HR_N2E] = dot(m, c, v[V_N2]);
        r->t[AC_HR_BREIT] = -4.0 / 3.0 * dot(m, v[V_F], h2);
        r->t[AC_HR_PROD] = -2.0 * a4 * g + 2.0 * fm * ha;
        r->t[AC_HR_ANOM] = z * z * a4;
        r->t[AC_HR_S2] = s2;
        r->total = 0.0;
        for (int k = 0; k < AC_HR_N; k++)
            r->total += r->t[k] /= a4;
        r->chk_x = dot(m, c, v[V_X]) / a4 - 1.0;
        r->chk_h = (hap - ha) / a4;
        fprintf(out,
                "hfsrel %dS: %d CSFs, %d configurations, %zu H elements (Davidson %d, CG %d)\n",
                ts + 1, m, b.ng, b.nh, nd, ncg);
        fprintf(out,
                "hfsrel %dS: E %.10f  A^(4) %.8f  <H^A> %+.8f  checks <X'>/A4-1 %+.2e"
                "  (<H'^A>-<H^A>)/A4 %+.2e\n",
                ts + 1, r->e, r->a4pp, ha, r->chk_x, r->chk_h);
        fprintf(out,
                "hfsrel %dS: K %+.8f  sep %+.8f  N2e %+.8f  Breit %+.8f  prod %+.8f  anom %+.8f"
                "  s2 %+.8f  A_AN/A4 %.8f\n",
                ts + 1, r->t[AC_HR_K], r->t[AC_HR_SEP], r->t[AC_HR_N2E], r->t[AC_HR_BREIT],
                r->t[AC_HR_PROD], r->t[AC_HR_ANOM], r->t[AC_HR_S2], r->total);
        /* <sum delta(r_ab)> of the CI converges slowly (no e-e cusp); its share in the product term 2 <F> <H^A> is linear */
        r->delta = dot(m, c, v[V_DEL]) / pi;
        r->ddelta = 2.0 * fm * pi / a4;
        fprintf(out, "hfsrel %dS: <sum delta(r_ab)> %.6f  d(A_AN/A4)/d<sum delta> (product term) %.6f",
                ts + 1, r->delta, r->ddelta);
        if (!isnan(ac_hfsrel_delta))
            fprintf(out, "  with <sum delta> = %.6f: A_AN/A4 %.8f", ac_hfsrel_delta,
                    r->total + r->ddelta * (ac_hfsrel_delta - r->delta));
        fprintf(out, "\n");
        fflush(out);
    }
    for (int k = 0; k < 16; k++)
        free(v[k]);
    free(dg);
    free(c);
    blk_free(&b);
    return rc;
}

int ac_hfsrel_print(FILE *out, ac_ci *ci, const ac_radial *rad, const int *mo,
                    ac_hfsrel_res *res, int maxres, int *nres, char *err, int errlen)
{
    if (err && errlen > 0)
        err[0] = 0;
    *nres = 0;
    if (!ci->in.roo)
        return fail(err, errlen, "hfsrel: no orbit-orbit integrals");
    lap("start");
    hs h;
    memset(&h, 0, sizeof h);
    h.rad = rad;
    h.mo = mo;
    h.nsl = ci->nwt;
    h.z = rad->z;
    build_matrices(&h);
    size_t ns = (size_t)h.nsl;
    h.ct = calloc(ns * ns, sizeof(double *));
    h.n2 = calloc(MAXK * ns * ns, sizeof(double *));
    h.yk = calloc(MAXK * ns * ns, sizeof(double *));
    h.yd = calloc(MAXK * ns * ns, sizeof(double *));
    int n = ci->ncsf, rc = 0;
    int *idx = malloc((size_t)n * sizeof *idx);
    for (int ts = 1; ts <= 2 * AC_CSF_MAXSH && !rc; ts++) {
        int m = 0;
        for (int a = 0; a < n; a++) {
            ac_lab T = ci->csf[a].c.cpl[ci->csf[a].c.nsh];
            if (T.ts == ts && T.tl == 0)
                idx[m++] = a;
        }
        if (!m)
            continue;
        ac_hfsrel_res r;
        rc = block(out, ci, &h, idx, m, ts, &r, err, errlen);
        if (!rc && res && *nres < maxres)
            res[(*nres)++] = r;
    }
    free(idx);
    for (int k = 0; k < M_N; k++)
        free(h.m[k]);
    for (size_t i = 0; i < ns * ns; i++)
        free(h.ct[i]);
    for (size_t i = 0; i < MAXK * ns * ns; i++) {
        free(h.n2[i]);
        free(h.yk[i]);
        free(h.yd[i]);
    }
    free(h.ct);
    free(h.n2);
    free(h.yk);
    free(h.yd);
    return rc;
}
