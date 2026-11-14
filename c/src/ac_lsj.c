/*
 * ac_lsj.c -- the CI in the LSJ basis (see ac_lsj.h).
 *
 * The formulas are derived in the technical note (Secs. 3.6 and 5). Sources
 * cited below by author and year:
 *   Edmonds (1957)   A. R. Edmonds, Angular Momentum in Quantum Mechanics,
 *                    Princeton University Press (1957), Eqs. (7.1.1),
 *                    (7.1.5), (7.1.6)
 *   Neese (2001)     F. Neese, J. Chem. Phys. 115, 11080 (2001)
 *   Pachucki (2004)  K. Pachucki, Phys. Rev. A 69, 052502 (2004): Eq. (32)
 *                    (alpha^2 corrections to g_J), Eqs. (59)-(62)
 *                    (relativistic M1)
 *   Jentschura (2010) U. D. Jentschura, Phys. Rev. A 81, 012512 (2010),
 *                    Eq. (10) (the H6 term, as a check)
 *   Lindgren-Rosen (1974) I. Lindgren and A. Rosen, Case Stud. At. Phys. 4,
 *                    93 (1974) (hyperfine operators)
 *   Sakurai (1967), Messiah (1962)  J. J. Sakurai, Advanced Quantum
 *                    Mechanics (1967), ch. 2; A. Messiah, Quantum Mechanics,
 *                    Vol. II (1962), ch. XXI (radiative rates)
 * Physical constants are those of CODATA 2018 (ac_lsj.h).
 */
#include "ac_lsj.h"
#include "ac_radial.h"
#include "ac_wigner.h"

#include <math.h>
#include <omp.h>
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

static ac_lab total(const ac_ci *ci, int i)
{
    return ci->csf[i].c.cpl[ci->csf[i].c.nsh];
}

/* the number of electrons CSF j must move to become CSF i (slot lists are
   in increasing orbital order) */
static int moved(const ac_ci *ci, int i, int j)
{
    const ac_ci_csf *F = &ci->csf[i], *G = &ci->csf[j];
    int p = 1, q = 1, nm = 0;
    while (p <= F->c.nsh || q <= G->c.nsh) {
        int sf = p <= F->c.nsh ? F->seq[p] : 1 << 30;
        int sg = q <= G->c.nsh ? G->seq[q] : 1 << 30;
        int d = 0;
        if (sf == sg)
            d = F->c.nel[p++] - G->c.nel[q++];
        else if (sf < sg)
            d = F->c.nel[p++];
        else
            q++;
        if (d > 0)
            nm += d;
    }
    return nm;
}

int ac_lsj_prepare(ac_lsj *m, ac_ci *ci, char *err, int errlen)
{
    int n = ci->ncsf;
    m->n = n;
    m->h = calloc((size_t)n * n, sizeof(double));
    /* W only with a spin-orbit term (without, ac_ci_so1 and ac_ci_so2 are 0:
       an n x n matrix of zeros, 670 MB at n = 9161, is not kept) */
    int so = ci->in.zeta || ci->in.rkg;
    m->w = so ? calloc((size_t)n * n, sizeof(double)) : NULL;
    m->w2 = ci->in.rss ? calloc((size_t)n * n, sizeof(double)) : NULL;
    if (!m->h || (so && !m->w) || (ci->in.rss && !m->w2)) {
        ac_lsj_free(m);
        return fail(err, errlen, "out of memory");
    }
    /* Upper triangle only: H is symmetric, and the reduced elements of the
       Hermitian double tensors W (SO) and W_SS obey
         <j||W||i> = (-1)^((S_i - S_j) + (L_i - L_j)) <i||W||j>
       (checked numerically to 1e-17 on the Si and Ti lists). Pairs that
       differ by more than two electrons have no element at all.
       Rows run in parallel (OpenMP): each thread works on a shallow copy of
       ci with its own Theta cache; the engine's other caches and those of
       the integral callbacks are thread-local. Each element is computed by
       one thread exactly as in a serial run. */
    int fe = 0, fi = 0, fj = 0, fkind = 0;
    long ops = 0;
#pragma omp parallel reduction(+ : ops)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (moved(&loc, i, j) > 2)
                    continue;
                ac_lab a = total(&loc, i), b = total(&loc, j);
                int e = 0, kind = 0;
                if (a.ts == b.ts && a.tl == b.tl) {
                    double v = ac_ci_hij(&loc, i, j, &e);
                    if (!e)
                        v += ac_ci_oo(&loc, i, j, &e);
                    m->h[(size_t)i * n + j] = m->h[(size_t)j * n + i] = v;
                }
                if (!e) {
                    kind = 1;
                    double s = ((a.ts - b.ts) + (a.tl - b.tl)) / 2 % 2 ? -1.0 : 1.0;
                    if (m->w) {
                        double w = ac_ci_so1(&loc, i, j, &e);
                        if (!e)
                            w += ac_ci_so2(&loc, i, j, &e);
                        m->w[(size_t)i * n + j] = w;
                        m->w[(size_t)j * n + i] = s * w;
                    }
                    if (!e && m->w2) {
                        double w2 = ac_ci_ss(&loc, i, j, &e);
                        m->w2[(size_t)i * n + j] = w2;
                        m->w2[(size_t)j * n + i] = s * w2;
                    }
                }
                if (e) {
#pragma omp critical(ac_lsj_fail)
                    if (!fe)
                        fe = e, fi = i, fj = j, fkind = kind;
                    break;
                }
            }
        }
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    ci->ops_evaluated += ops;
    if (fe) {
        ac_lsj_free(m);
        return fail(err, errlen, "%s(%d,%d): engine error %d", fkind ? "W" : "H", fi, fj, fe);
    }
    return 0;
}

void ac_lsj_free(ac_lsj *m)
{
    free(m->h);
    free(m->w);
    free(m->w2);
    memset(m, 0, sizeof *m);
}

static int tri2(int a, int b, int c)
{
    return (a + b + c) % 2 == 0 && c <= a + b && c >= abs(a - b);
}

/* the J block: scalar product of the double tensors W^(kappa kappa)
   (Edmonds (1957), Eq. (7.1.6); technical note, Sec. 3.6) */
int ac_lsj_matrix(const ac_lsj *m, const ac_ci *ci, int tJ, int *idx, double *H,
                  double *asym)
{
    int dim = 0;
    for (int i = 0; i < m->n; i++) {
        ac_lab a = total(ci, i);
        if (tri2(a.ts, a.tl, tJ))
            idx[dim++] = i;
    }
    for (int p = 0; p < dim; p++)
        for (int q = 0; q < dim; q++) {
            int i = idx[p], j = idx[q];
            ac_lab a = total(ci, i), b = total(ci, j);
            double v = m->h[(size_t)i * m->n + j];
            double w = m->w ? m->w[(size_t)i * m->n + j] : 0.0;
            if (w != 0.0) {
                int ph = (b.ts + a.tl + tJ) / 2; /* S' + L + J */
                v += -(ph % 2 ? -1.0 : 1.0) / sqrt(3.0) *
                     ac_sixj(tJ, a.tl, a.ts, 2, b.ts, b.tl) * w;
            }
            double w2 = m->w2 ? m->w2[(size_t)i * m->n + j] : 0.0;
            if (w2 != 0.0) { /* kappa = 2: ((-1)^2/sqrt5) (-1)^(S'+L+J) {J L S; 2 S' L'} */
                int ph = (b.ts + a.tl + tJ) / 2;
                v += (ph % 2 ? -1.0 : 1.0) / sqrt(5.0) * ac_sixj(tJ, a.tl, a.ts, 4, b.ts, b.tl) * w2;
            }
            H[(size_t)q * dim + p] = v;
        }
    double d = 0.0;
    for (int p = 0; p < dim; p++)
        for (int q = 0; q < p; q++) {
            double x = H[(size_t)q * dim + p], y = H[(size_t)p * dim + q];
            d = fmax(d, fabs(x - y));
            H[(size_t)q * dim + p] = H[(size_t)p * dim + q] = 0.5 * (x + y);
        }
    if (asym)
        *asym = d;
    return dim;
}

/* Edmonds (1957), Eq. (7.1.5) */
double ac_lsj_reduce(ac_lab a, int tJ, ac_lab b, int tJp, int ks, int kl, int tK, double w)
{
    if (w == 0.0)
        return 0.0;
    return sqrt((double)(tJ + 1) * (tJp + 1) * (tK + 1)) *
           ac_ninej(a.ts, b.ts, ks, a.tl, b.tl, kl, tJ, tJp, tK) * w;
}

/* ---- first-order Zeeman ------------------------------------------------- */

#define ALPHA2 (ac_alpha * ac_alpha)

typedef struct {
    int kind; /* 0 total S, 1 total L, 2 RMC, 3 GC rank (1,0), 4 GC rank (1,2),
                 5 orbital RMC; hyperfine: 6 orbital, 7 spin-dipole, 8 contact,
                 9 electric quadrupole; transitions: 10 E1 length r C^1,
                 11 E1 velocity grad, 12 E2 r^2 C^2; relativistic M1: 13-15
                 s r^2, s p^2, s Z/r, 16-17 l r^2, l p^2, 18-19 s C^2 r^2,
                 s C^2 Z/r, 20 s [p x p]^(2); isotope shift: 21 p^2/2,
                 22 delta(r); transitions: 23 E3 r^3 C^3, 24-25 M2 orbital
                 and spin parts, 26-27 M3 orbital and spin parts */
    double (*r)(void *, int, int);
    void *ctx;
    double (*rk)(void *, int, int, int); /* kinds 10-12: (ctx, kind, a, b) */
} gop;

/* <l||C^k||l'> = (-1)^(l+g) sqrt([l][l']) (l k l'; 0 0 0), 2g = l + k + l'
   (= -[l] F(l,2,l) for k = 2, l' = l) */
static double ckred(int l, int k, int lb)
{
    if ((l + k + lb) % 2 || k < abs(l - lb) || k > l + lb)
        return 0.0;
    return ((l + (l + k + lb) / 2) % 2 ? -1.0 : 1.0) * sqrt((2.0 * l + 1) * (2.0 * lb + 1)) *
           ac_fabc(l, k, lb);
}

static double c2red(int l, int lb)
{
    return ckred(l, 2, lb);
}

/* g_s of the magnetic transition operators (M1, M2, M3): AC_GS, or 2 with the
   environment variable AC_TRANS_GS2 set (ATSP's trans uses g_s = 2; for
   comparisons only) */
static double trans_gs(void)
{
    return getenv("AC_TRANS_GS2") ? 2.0 : AC_GS;
}

/* one-electron factors of the property operators. Sources: kinds 2-5
   (RMC, GC1e, ORMC) Neese (2001) and Pachucki (2004), Eq. (32) (H1, H2);
   6-9 (hyperfine) Lindgren-Rosen (1974); 10-12, 23-27 (multipoles)
   technical note, Sec. 5.3, Edmonds (1957), Eq. (7.1.1); 13-20 (relativistic M1)
   Pachucki (2004), Eqs. (59)-(62), kappa = 0; 21, 22 (isotope shift)
   technical note, Sec. 5.5 */
static double gop_f(void *p, int a, int b, int l, int lb)
{
    const gop *o = p;
    double sl = sqrt(1.5); /* <1/2||s||1/2> */
    if (lb != l && o->kind != 4 && o->kind != 7 && o->kind != 9 && o->kind < 10)
        return 0.0;
    switch (o->kind) {
    case 10: { /* r C^1: <a|r|b> <l||C^1||l'> */
        double c1 = ckred(l, 1, lb);
        return c1 == 0.0 ? 0.0 : o->rk(o->ctx, 0, a, b) * ac_ci_one_coef(sqrt(2.0), c1, 0, 2);
    }
    case 11: { /* grad: <l||C^1||l'> int R_a (R_b' - l' R_b/r) r^2 dr for l = l' + 1,
                  (R_b' + (l'+1) R_b/r) for l = l' - 1 (as ac_ci_rec2) */
        double c1 = ckred(l, 1, lb);
        return c1 == 0.0 ? 0.0
                         : o->rk(o->ctx, l == lb + 1 ? 1 : 2, a, b) *
                               ac_ci_one_coef(sqrt(2.0), c1, 0, 2);
    }
    case 12: { /* r^2 C^2 */
        double c2 = ckred(l, 2, lb);
        return c2 == 0.0 ? 0.0 : o->rk(o->ctx, 3, a, b) * ac_ci_one_coef(sqrt(2.0), c2, 0, 4);
    }
    case 23: { /* r^3 C^3 (rk kind 11) */
        double c3 = ckred(l, 3, lb);
        return c3 == 0.0 ? 0.0 : o->rk(o->ctx, 11, a, b) * ac_ci_one_coef(sqrt(2.0), c3, 0, 6);
    }
    case 24:   /* M2 = grad(r^2 C^2).((2/3) l + g_s s) (g_s of trans_gs) = sqrt6 r [C^1 x ((2/3) l + g_s s)]^(2)
                  (grad(r^k C^k) = sqrt(k(2k-1)) r^(k-1) [C^(k-1) x e]^(k)); orbital part
                  (0, 2): <l||[C^1 x l]^(2)||l'> = (-1)^(l+l') sqrt5 {1 1 2; l' l l'}
                  <l||C^1||l'> <l'||l||l'> (Edmonds (7.1.1)) */
    case 25:   /* spin part (1, 1), coupled to J rank 2 */
    case 26:   /* M3 = grad(r^3 C^3).((1/2) l + g_s s) = sqrt15 r^2 [C^2 x ((1/2) l + g_s s)]^(3);
                  orbital part (0, 3): (-1)^(l+l'+1) sqrt7 {2 1 3; l' l l'} <l||C^2||l'> <l'||l||l'> */
    case 27: { /* spin part (1, 2), coupled to J rank 3 */
        int k = o->kind < 26 ? 2 : 3; /* M k: C^(k-1), r^(k-1) (rk kinds 0, 3) */
        double ck = ckred(l, k - 1, lb);
        if (ck == 0.0)
            return 0.0;
        double r1 = o->rk(o->ctx, k == 2 ? 0 : 3, a, b), pre = sqrt(k * (2.0 * k - 1));
        if (o->kind == 25 || o->kind == 27)
            return pre * trans_gs() * r1 * ac_ci_one_coef(sl, ck, 2, 2 * k - 2);
        if (lb == 0)
            return 0.0;
        double v = ((l + lb + k) % 2 ? -1.0 : 1.0) * sqrt(2.0 * k + 1) *
                   ac_sixj(2 * k - 2, 2, 2 * k, 2 * lb, 2 * l, 2 * lb) * ck *
                   sqrt(lb * (lb + 1) * (2.0 * lb + 1));
        return pre * (2.0 / (k + 1)) * r1 * ac_ci_one_coef(sqrt(2.0), v, 0, 2 * k);
    }
    case 13: /* s f(r, p), f = r^2, p^2, Z/r (rk kinds 3, 4, 5): relativistic M1 */
    case 14:
    case 15:
        return l != lb ? 0.0
                       : o->rk(o->ctx, o->kind - 10, a, b) *
                             ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0);
    case 16: /* l f(r, p), f = r^2, p^2 (rk kinds 3, 4): relativistic M1, orbital */
    case 17:
        return l != lb ? 0.0
                       : o->rk(o->ctx, o->kind - 13, a, b) *
                             ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2);
    case 18: /* s C^2 f(r), f = r^2, Z/r (rk kinds 3, 5): relativistic M1, tensor */
    case 19: {
        double c2 = c2red(l, lb);
        return c2 == 0.0 ? 0.0
                         : o->rk(o->ctx, o->kind == 18 ? 3 : 5, a, b) * ac_ci_one_coef(sl, c2, 2, 4);
    }
    case 20: { /* s [p x p]^(2) = -s [grad x grad]^(2): Edmonds (7.1.1) through l'' = l +- 1,
                  <l''||grad||l> = <l''||C^1||l> D, the radial part by parts
                  int (D R_a)(D R_b) r^2 dr (rk kinds 6-9 = 6 + 2 (l'' > l) + (l'' > l')) */
        double v = 0.0;
        for (int lpp = l - 1; lpp <= l + 1; lpp += 2) {
            if (lpp < 0 || abs(lpp - lb) != 1)
                continue;
            double c = ckred(lpp, 1, l) * ckred(lpp, 1, lb);
            if (c != 0.0)
                v += ac_sixj(2, 2, 4, 2 * lb, 2 * l, 2 * lpp) * c *
                     o->rk(o->ctx, 6 + 2 * (lpp > l) + (lpp > lb), a, b);
        }
        v *= -sqrt(5.0) * ((l + lb) % 2 ? -1.0 : 1.0);
        return v == 0.0 ? 0.0 : ac_ci_one_coef(sl, v, 2, 4);
    }
    case 21: /* p^2/2 (rk kind 4 = <p^2>) */
        return l != lb ? 0.0
                       : 0.5 * o->rk(o->ctx, 4, a, b) *
                             ac_ci_one_coef(sqrt(2.0), sqrt(2.0 * l + 1), 0, 0);
    case 22: /* delta(r): R_a(0) R_b(0)/(4 pi) (rk kind 10) */
        return l != 0 || lb != 0 ? 0.0
                                 : o->rk(o->ctx, 10, a, b) / (16.0 * atan(1.0)) *
                                       ac_ci_one_coef(sqrt(2.0), 1.0, 0, 0);
    case 6: /* l r^-3 (0 for s, whose <r^-3> diverges) */
        return l == 0 ? 0.0
                      : o->r(o->ctx, a, b) *
               ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2);
    case 7: { /* -sqrt(10) (g_s/2) [s x C^2]^(1) r^-3 = (g_s/2)(3 (s.r^) r^ - s) r^-3 */
        double c2 = c2red(l, lb);
        return c2 == 0.0 ? 0.0
                         : -sqrt(10.0) * 0.5 * AC_GS * o->r(o->ctx, a, b) *
                               ac_ci_one_coef(sl, c2, 2, 4);
    }
    case 8: /* (g_s/2)(8 pi/3) delta(r) s: <a|delta|b> = R_a(0) R_b(0)/(4 pi), so the
               factor on r00 = R_a(0) R_b(0) is (g_s/2)(2/3) */
        return l == 0 ? AC_GS / 3.0 * o->r(o->ctx, a, b) * ac_ci_one_coef(sl, 1.0, 2, 0) : 0.0;
    case 9: { /* C^2 r^-3 */
        double c2 = c2red(l, lb);
        return c2 == 0.0 ? 0.0 : o->r(o->ctx, a, b) * ac_ci_one_coef(sqrt(2.0), c2, 0, 4);
    }
    case 0:
        return a == b ? ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0) : 0.0;
    case 1:
        return a == b ? ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2) : 0.0;
    case 2:
        return -2.0 * ALPHA2 * o->r(o->ctx, a, b) * ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0);
    case 3:
        return 0.5 * ALPHA2 * o->r(o->ctx, a, b) * (2.0 / 3.0) *
               ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0);
    case 5:
        return -ALPHA2 * o->r(o->ctx, a, b) *
               ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2);
    default: {
        /* s_z - (s.r^) z^ = (2/3) s_z - (s.Q)_z, Q_ij = r^_i r^_j - delta_ij/3;
           (s.Q)^(1) = -sqrt(5/3) [s x Q^(2)]^(1), Q^(2) = sqrt(2/3) C^(2);
           <l||C^2||l'> couples l' = l and l +- 2 (s-d, p-f in a correlated CI) */
        double c2 = c2red(l, lb);
        if (c2 == 0.0)
            return 0.0;
        return 0.5 * ALPHA2 * o->r(o->ctx, a, b) * (sqrt(10.0) / 3.0) *
               ac_ci_one_coef(sl, c2, 2, 4);
    }
    }
}

/* the pairs (p, q) of the bases of two levels that a one-body operator can
   connect: c_p, c_q nonzero and at most one electron moved; row p holds
   q[start[p] .. start[p+1]-1], ascending. Made once for all the operators
   between the two levels, before the engine aligns any shells. */
typedef struct {
    size_t *start;
    int *q;
} pairs1;

static void pairs1_free(pairs1 *P)
{
    free(P->start);
    free(P->q);
    P->start = NULL;
    P->q = NULL;
}

static int pairs1_build(const ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B, pairs1 *P)
{
    P->start = calloc((size_t)A->dim + 1, sizeof *P->start);
    P->q = NULL;
    if (!P->start)
        return -1;
#pragma omp parallel for schedule(dynamic, 16)
    for (int p = 0; p < A->dim; p++) {
        size_t m = 0;
        if (A->c[p] != 0.0)
            for (int q = 0; q < B->dim; q++)
                m += B->c[q] != 0.0 && moved(ci, A->idx[p], B->idx[q]) <= 1;
        P->start[p + 1] = m;
    }
    for (int p = 0; p < A->dim; p++)
        P->start[p + 1] += P->start[p];
    P->q = malloc((P->start[A->dim] + 1) * sizeof *P->q);
    if (!P->q) {
        pairs1_free(P);
        return -1;
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (int p = 0; p < A->dim; p++) {
        size_t m = P->start[p];
        if (A->c[p] != 0.0)
            for (int q = 0; q < B->dim; q++)
                if (B->c[q] != 0.0 && moved(ci, A->idx[p], B->idx[q]) <= 1)
                    P->q[m++] = q;
    }
    return 0;
}

/* <psi_A||[W]^(K)||psi_B> (2K = tK) of the operator kind, psi_A over the
   basis idxA of 2J = tJA and psi_B over idxB of tJB (the same level for g_J),
   on the pairs P of pairs1_build */
static double pair_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                           const pairs1 *P, const gop *o, int ks, int kl, int tK, int *err)
{
    /* rows in parallel on copies of ci, as in ac_lsj_prepare; the row sums
       are added in order, so the result does not depend on the threads */
    double *row = calloc((size_t)A->dim, sizeof *row);
    int fe = 0;
    if (!row) {
        *err = 1;
        return 0.0;
    }
    long ops = 0;
#pragma omp parallel reduction(+ : ops)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
#pragma omp for schedule(dynamic, 4)
        for (int p = 0; p < A->dim; p++) {
            if (A->c[p] == 0.0)
                continue;
            double s = 0.0;
            for (size_t m = P->start[p]; m < P->start[p + 1]; m++) {
                int q = P->q[m];
                int e = 0;
                double w =
                    ac_ci_one_ll(&loc, A->idx[p], B->idx[q], ks, kl, gop_f, (void *)o, &e);
                if (e) {
#pragma omp critical(ac_lsj_fail)
                    fe = e;
                    break;
                }
                s += B->c[q] * ac_lsj_reduce(total(&loc, A->idx[p]), A->tJ,
                                             total(&loc, B->idx[q]), B->tJ, ks, kl, tK, w);
            }
            row[p] = A->c[p] * s;
        }
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    ci->ops_evaluated += ops;
    double s = 0.0;
    for (int p = 0; p < A->dim; p++)
        s += row[p];
    free(row);
    *err = fe;
    return fe ? 0.0 : s;
}

/* ---- the CSF-pair elements of the two-electron property operators -------
   pair_reduced_2e needs <i||W||j> of the same CSF pairs again and again: for
   every level of a J block, for every J block that holds both CSFs, and for
   every level pair of the zoff table. Within one call of
   ac_lsj_print_levels_keep or ac_lsj_print_cross (ci and its integrals do
   not change there) each element is kept in a dense n x n table per
   operator, NaN until it is computed. pair_reduced_2e gives row i =
   A->idx[p] to one thread, which alone reads and writes row i, so the tables
   need no locking, and every value is the one an uncached run computes, bit
   for bit. A table is allocated on first use while the total stays below
   AC_PROP_CACHE_MB (default 2048); an operator without a table is computed
   as before. */
enum { PC_GC0, PC_GC1, PC_GC2, PC_OZ, PC_REC, PC_M10, PC_M12, PC_SMS, PC_N };

static struct {
    const ac_ci *ci;
    int n, depth;
    size_t bytes, limit;
    double *v[PC_N];
} pc;

static int pc_slot(int kp)
{
    switch (kp) {
    case 0:
    case 1:
    case 2:
        return PC_GC0 + kp;
    case -1:
        return PC_OZ;
    case -2:
        return PC_REC;
    case -3:
        return PC_SMS;
    case 10:
        return PC_M10;
    case 12:
        return PC_M12;
    }
    return -1;
}

static void pc_begin(const ac_ci *ci)
{
    if (pc.depth++)
        return;
    const char *s = getenv("AC_PROP_CACHE_MB");
    pc.ci = ci;
    pc.n = ci->ncsf;
    pc.bytes = 0;
    pc.limit = (size_t)((s ? atof(s) : 2048.0) * 1048576.0);
}

static void pc_end(void)
{
    if (--pc.depth)
        return;
    for (int k = 0; k < PC_N; k++) {
        free(pc.v[k]);
        pc.v[k] = NULL;
    }
    pc.ci = NULL;
}

/* the table of operator kp, or NULL (outside a window, another ci, over the
   limit); called outside the parallel region */
static double *pc_table(const ac_ci *ci, int kp)
{
    int k = pc_slot(kp);
    if (!pc.depth || ci != pc.ci || k < 0)
        return NULL;
    if (!pc.v[k]) {
        size_t nn = (size_t)pc.n * pc.n, sz = nn * sizeof(double);
        if (pc.bytes + sz > pc.limit)
            return NULL;
        double *t = malloc(sz);
        if (!t)
            return NULL;
        for (size_t i = 0; i < nn; i++)
            t[i] = NAN;
        pc.v[k] = t;
        pc.bytes += sz;
    }
    return pc.v[k];
}

/* <psi_A||[W^(1 kp)]^(1)||psi_B> of the two-electron gauge correction, or
   <psi_A||W^(0 1)||psi_B> of the orbit-orbit Zeeman term (kp = -1) or of the
   two-body recoil term (kp = -2), or <psi_A||[W^(1 K)]^(1)||psi_B> of the
   spin two-electron M1 term (kp = 10 + K, ac_ci_m1s2), or the scalar
   <psi_A||W^(0 0)||psi_B> of the specific mass shift (kp = -3, ac_ci_sms
   with the gradient factors of rk, ctx) */
static double pair_reduced_2e(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B, int kp,
                              double (*rk)(void *, int, int, int), void *ctx, int *err)
{
    /* in parallel as pair_reduced; each thread has its own Theta cache */
    double *row = calloc((size_t)A->dim, sizeof *row);
    int fe = 0;
    if (!row) {
        *err = 1;
        return 0.0;
    }
    long ops = 0;
    double *tab = pc_table(ci, kp);
    size_t n = (size_t)ci->ncsf;
#pragma omp parallel reduction(+ : ops)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
#pragma omp for schedule(dynamic, 4)
        for (int p = 0; p < A->dim; p++) {
            if (A->c[p] == 0.0)
                continue;
            double s = 0.0;
            for (int q = 0; q < B->dim; q++) {
                /* a two-body operator moves at most two electrons */
                if (B->c[q] == 0.0 || moved(&loc, A->idx[p], B->idx[q]) > 2)
                    continue;
                int e = 0, i = A->idx[p], j = B->idx[q];
                double *slot = tab ? tab + (size_t)i * n + j : NULL, w;
                if (slot && !isnan(*slot))
                    w = *slot;
                else {
                    w = kp == -3   ? ac_ci_sms(&loc, i, j, rk, ctx, &e)
                        : kp == -2 ? ac_ci_rec2(&loc, i, j, &e)
                        : kp < 0   ? ac_ci_oz2e(&loc, i, j, &e)
                        : kp >= 10 ? ac_ci_m1s2(&loc, i, j, kp - 10, &e)
                                   : ac_ci_gc2e(&loc, i, j, kp, &e);
                    if (slot && !e)
                        *slot = w;
                }
                if (e) {
#pragma omp critical(ac_lsj_fail)
                    fe = e;
                    break;
                }
                ac_lab a = total(&loc, i), b = total(&loc, j);
                int K = kp >= 10 ? kp - 10 : kp;
                s += B->c[q] * (kp == -3 ? ac_lsj_reduce(a, A->tJ, b, B->tJ, 0, 0, 0, w)
                                : kp < 0 ? ac_lsj_reduce(a, A->tJ, b, B->tJ, 0, 2, 2, w)
                                         : ac_lsj_reduce(a, A->tJ, b, B->tJ, 2, 2 * K, 2, w));
            }
            row[p] = A->c[p] * s;
        }
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    ci->ops_evaluated += ops;
    double s = 0.0;
    for (int p = 0; p < A->dim; p++)
        s += row[p];
    free(row);
    *err = fe;
    return fe ? 0.0 : s;
}

static double pair_reduced_gc2e(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                                int kp, int *err)
{
    return pair_reduced_2e(ci, A, B, kp, NULL, NULL, err);
}

double (*ac_lsj_hfs_rm3)(void *, int, int) = NULL;
double (*ac_lsj_hfs_r00)(void *, int, int) = NULL;

/* ---- level listing ---------------------------------------------------- */

typedef struct {
    int tJ;
    double e;
} level;

int ac_lsj_zoff = 0;
double (*ac_lsj_h6_rk)(void *, int, int, int) = NULL;
double (*ac_lsj_is_rk)(void *, int, int, int) = NULL;

/* the printed levels, kept for the table of <A||mu||B> */
typedef ac_lsj_kept kept;

void ac_lsj_kept_free(ac_lsj_kept *k, int n)
{
    for (int i = 0; i < n; i++) {
        free((void *)k[i].v.idx);
        free((void *)k[i].v.c);
    }
    free(k);
}

/* The pairs of the tables: within one list (cross = 0) each pair once, B
   after A (from A itself when diag); between two lists every pair. */
static int pair_first(int cross, int diag, int a)
{
    return cross ? 0 : diag ? a : a + 1;
}

static int keep_level(kept **k, int *nk, int tJ, int num, double e, int dim, const int *idx,
                      const double *c)
{
    kept *t = realloc(*k, (size_t)(*nk + 1) * sizeof *t);
    if (!t)
        return -1;
    *k = t;
    int *ix = malloc((size_t)dim * sizeof *ix);
    double *cc = malloc((size_t)dim * sizeof *cc);
    if (!ix || !cc) {
        free(ix);
        free(cc);
        return -1;
    }
    memcpy(ix, idx, (size_t)dim * sizeof *ix);
    memcpy(cc, c, (size_t)dim * sizeof *cc);
    t[*nk].v = (ac_lsj_level){tJ, dim, ix, cc};
    t[*nk].num = num;
    t[*nk].e = e;
    (*nk)++;
    return 0;
}

static int print_zoff(FILE *out, ac_ci *ci, const kept *k, int nk, const kept *kb, int nb,
                      int cross, double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                      void *ctx, char *err, int errlen)
{
    fprintf(out, "Zeeman reduced matrix elements <A||mu||B> between different levels "
                 "(Edmonds; mu = L + g_s S + alpha^2 terms, in mu_B):\n");
    fprintf(out, "%4s %3s %4s %3s %16s %16s %16s %14s %14s %14s%s%s%s%s\n", "J_A", "#A", "J_B",
            "#B", "<L>", "<S>", "<L + g_s S>", "RMC", "GC", "ORMC",
            ci->in.rgc ? "           GC2e" : "", ci->in.roz ? "           OZ2e" : "",
            ci->in.rrc ? "           rec2" : "", ac_lsj_h6_rk ? "             H6" : "");
    for (int a = 0; a < nk; a++)
        for (int b = pair_first(cross, 0, a); b < nb; b++) {
            const ac_lsj_level *A = &k[a].v, *B = &kb[b].v;
            if (abs(A->tJ - B->tJ) > 2 || A->tJ + B->tJ < 2)
                continue;
            ac_lsj_g r;
            int e;
            if (ac_lsj_zeeman_reduced(ci, A, B, tkin, zr, ctx, &r, &e))
                return fail(err, errlen, "Zeeman <A||mu||B>: engine error %d", e);
            fprintf(out, "%2d/2 %3d %2d/2 %3d %16.12f %16.12f %16.12f %14.6e %14.6e %14.6e",
                    A->tJ, k[a].num, B->tJ, kb[b].num, r.gL, r.gS, r.gL + AC_GS * r.gS, r.rmc,
                    r.gc, r.ormc);
            if (ci->in.rgc)
                fprintf(out, " %15.8e", r.gc2e);
            if (ci->in.roz)
                fprintf(out, " %15.8e", r.oz2e);
            if (ci->in.rrc)
                fprintf(out, " %15.8e", r.rec2);
            if (ac_lsj_h6_rk)
                fprintf(out, " %15.8e", r.h6);
            fprintf(out, "\n");
        }
    return 0;
}

/* A_J and B_J from the reduced elements (Lindgren-Rosen (1974); technical
   note, Sec. 5.2) */
static int print_hfs(FILE *out, ac_ci *ci, const kept *k, int nk, const kept *kb, int nb,
                     int cross, void *ctx, char *err, int errlen)
{
    fprintf(out, "Hyperfine reduced matrix elements <A||T^(K)||B> (Edmonds, a.u.; T^(1) = "
                 "orb + sd + con, T^(2) = quad); A_J per mu_I/I and B_J per Q:\n");
    fprintf(out, "%4s %3s %4s %3s %16s %16s %16s %16s %16s %14s %14s\n", "J_A", "#A", "J_B", "#B",
            "orb", "sd", "con", "T^(1)", "quad", "A_J(MHz)", "B_J(MHz/b)");
    for (int a = 0; a < nk; a++)
        for (int b = pair_first(cross, 1, a); b < nb; b++) {
            const ac_lsj_level *A = &k[a].v, *B = &kb[b].v;
            if (abs(A->tJ - B->tJ) > 4 || A->tJ + B->tJ < 2)
                continue;
            ac_lsj_hfs h;
            int e;
            if (ac_lsj_hfs_reduced(ci, A, B, ac_lsj_hfs_rm3, ac_lsj_hfs_r00, ctx, &h, &e))
                return fail(err, errlen, "hyperfine <A||T||B>: engine error %d", e);
            double t1 = h.orb + h.sd + h.con;
            fprintf(out, "%2d/2 %3d %2d/2 %3d %16.9e %16.9e %16.9e %16.9e %16.9e", A->tJ,
                    k[a].num, B->tJ, kb[b].num, h.orb, h.sd, h.con, t1, h.quad);
            if (!cross && a == b) {
                double J = A->tJ / 2.0;
                double aj = AC_HFS_A0 * t1 / sqrt(J * (J + 1) * (2 * J + 1));
                double bj = -2.0 * AC_HFS_B0 * h.quad *
                            sqrt(J * (2 * J - 1) / ((J + 1) * (2 * J + 1) * (2 * J + 3)));
                fprintf(out, " %14.6f %14.6f", aj, bj);
            }
            fprintf(out, "\n");
        }
    return 0;
}

/* mass and field shift (technical note, Sec. 5.5; the field shift is the
   non-relativistic limit of Shabaev, Phys. Rep. 356, 119 (2002), Eqs. (322),
   (324), for a hydrogen-like ns state) */
static int print_is(FILE *out, ac_ci *ci, const kept *k, int nk, void *ctx, char *err,
                    int errlen)
{
    fprintf(out, "Isotope shift (a.u.): K_NMS = <sum p_i^2/2>, K_SMS = <sum_{i<j} p_i.p_j>,"
                 " mass shift (m/M)(K_NMS + K_SMS); rho(0) = <sum delta(r_i)>,"
                 " field shift (2 pi/3) Z rho(0) delta<r^2>:\n");
    fprintf(out, "%4s %3s %22s %22s %22s %22s\n", "J", "#", "K_NMS", "K_SMS", "K_NMS + K_SMS",
            "rho(0)");
    for (int a = 0; a < nk; a++) {
        ac_lsj_is r;
        int e;
        if (ac_lsj_is_level(ci, &k[a].v, ac_lsj_is_rk, ctx, &r, &e))
            return fail(err, errlen, "isotope shift: engine error %d", e);
        fprintf(out, "%2d/2 %3d %22.15f %22.15f %22.15f %22.12f\n", k[a].v.tJ, k[a].num, r.nms,
                r.sms, r.nms + r.sms, r.rho0);
    }
    return 0;
}

double (*ac_lsj_trans_rk)(void *, int, int, int) = NULL;

#define TRLAB "%2d/2 %3d %2d/2 %3d %4s %12.3f"

int ac_lsj_digits = 0;
int ac_lsj_davidson = 500;
/* width and precision of the line strengths */
#define SW (ac_lsj_digits ? 24 : 16)
#define SP (ac_lsj_digits ? 17 : 9)

/* f and A from S (Sakurai (1967), ch. 2; Messiah (1962), ch. XXI;
   technical note, Sec. 5.3) */
static int print_trans(FILE *out, ac_ci *ci, const kept *k, int nk, const kept *kb, int nb,
                       int cross, void *ctx, char *err, int errlen)
{
    const double tau0 = 2.4188843265857e-17, a3 = ac_alpha * ac_alpha * ac_alpha;
    fprintf(out, "Transitions between printed levels (lower -> upper; S = |<||T||>|^2 in a.u.,"
                 " A in s^-1, E1 f of absorption%s):\n",
            cross ? "; last column: the state of the lower level" : "");
    fprintf(out, "%4s %3s %4s %3s %4s %12s %16s %16s %13s %13s %13s%s\n", "J_lo", "#", "J_up", "#",
            "type", "dE(cm-1)", "S_length", "S_velocity", "f_length", "f_velocity", "A",
            cross ? " lo" : "");
    for (int a = 0; a < nk; a++)
        for (int b = pair_first(cross, 0, a); b < nb; b++) {
            const kept *lo = &k[a], *up = &kb[b];
            int swap = up->e < lo->e;
            if (swap) {
                const kept *x = lo;
                lo = up;
                up = x;
            }
            if (abs(lo->v.tJ - up->v.tJ) > 6 || lo->v.tJ + up->v.tJ < 2)
                continue;
            /* within one list the lower level is the bra; between two lists
               the level of the first (S = |<||T||>|^2 either way) */
            const kept *bra = cross ? &k[a] : lo, *ket = cross ? &kb[b] : up;
            ac_lsj_trans t;
            int e;
            if (ac_lsj_trans_reduced(ci, &bra->v, &ket->v, ac_lsj_trans_rk, ctx, &t, &e))
                return fail(err, errlen, "transition <A||T||B>: engine error %d", e);
            double w = up->e - lo->e, gl = lo->v.tJ + 1.0, gu = up->v.tJ + 1.0;
            double cm = w * AC_HARTREE_CM;
            const char *tag = !cross ? "" : swap ? "  2" : "  1";
            if (fabs(t.e1l) > 1e-12 || fabs(t.e1v) > 1e-12) {
                double sl = t.e1l * t.e1l, sv = t.e1v * t.e1v;
                fprintf(out, TRLAB " %*.*e %*.*e %13.6e %13.6e %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "E1", cm, SW, SP, sl, SW, SP, sv,
                        2.0 / 3.0 * w * sl / gl,
                        2.0 / 3.0 * sv / (w * gl), 4.0 / 3.0 * a3 * w * w * w * sl / gu / tau0,
                        tag);
            }
            if (fabs(t.m1) > 1e-12) {
                double s = t.m1 * t.m1;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M1", cm, SW, SP, s, "", "", "",
                        a3 * ac_alpha * ac_alpha * w * w * w * s / 3.0 / gu / tau0, tag);
            }
            /* M1 with the relativistic part (Pachucki's Eqs. (60)-(62); the
               r_ab terms when ci->in.rm1 is set); the column after S is
               <||delta mu||> itself */
            double dm = ac_lsj_m1_rel(&t, w);
            if (fabs(dm) > 1e-14) {
                double s = (t.m1 + dm) * (t.m1 + dm);
                fprintf(out, TRLAB " %*.*e %*.*e %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M1r", cm, SW, SP, s, SW, SP, dm, "", "",
                        a3 * ac_alpha * ac_alpha * w * w * w * s / 3.0 / gu / tau0, tag);
            }
            if (fabs(t.e2) > 1e-12) {
                double s = t.e2 * t.e2;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "E2", cm, SW, SP, s, "", "", "",
                        a3 * ac_alpha * ac_alpha * pow(w, 5) * s / 15.0 / gu / tau0, tag);
            }
            if (fabs(t.m2) > 1e-12) {
                double s = t.m2 * t.m2;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M2", cm, SW, SP, s, "", "", "",
                        pow(ac_alpha, 7) * pow(w, 5) * s / 60.0 / gu / tau0, tag);
            }
            if (fabs(t.e3) > 1e-12) {
                double s = t.e3 * t.e3;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "E3", cm, SW, SP, s, "", "", "",
                        8.0 / 4725.0 * pow(ac_alpha, 7) * pow(w, 7) * s / gu / tau0, tag);
            }
            if (fabs(t.m3) > 1e-12) {
                double s = t.m3 * t.m3;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M3", cm, SW, SP, s, "", "", "",
                        2.0 / 4725.0 * pow(ac_alpha, 9) * pow(w, 7) * s / gu / tau0, tag);
            }
        }
    return 0;
}

int ac_lsj_print_cross(FILE *out, ac_ci *ci, const ac_lsj_kept *ka, int na,
                       const ac_lsj_kept *kb, int nb, double (*tkin)(void *, int, int),
                       double (*zr)(void *, int, int), void *ctx, char *err, int errlen)
{
    int rc = -1;
    pc_begin(ci);
    if (ac_lsj_zoff && print_zoff(out, ci, ka, na, kb, nb, 1, tkin, zr, ctx, err, errlen))
        goto out;
    if (ac_lsj_hfs_rm3 && print_hfs(out, ci, ka, na, kb, nb, 1, ctx, err, errlen))
        goto out;
    if (ac_lsj_trans_rk && print_trans(out, ci, ka, na, kb, nb, 1, ctx, err, errlen))
        goto out;
    rc = 0;
out:
    pc_end();
    return rc;
}

static int by_energy(const void *a, const void *b)
{
    double d = ((const level *)a)->e - ((const level *)b)->e;
    return (d > 0) - (d < 0);
}

int ac_lsj_print_levels(FILE *out, ac_ci *ci, int nlev, double (*tkin)(void *, int, int),
                        double (*zr)(void *, int, int), void *ctx, char *err, int errlen)
{
    return ac_lsj_print_levels_keep(out, ci, nlev, tkin, zr, ctx, NULL, NULL, err, errlen);
}

int ac_lsj_print_levels_keep(FILE *out, ac_ci *ci, int nlev, double (*tkin)(void *, int, int),
                             double (*zr)(void *, int, int), void *ctx, ac_lsj_kept **keep,
                             int *nkeep, char *err, int errlen)
{
    ac_lsj m;
    double t0 = omp_get_wtime();
    if (ac_lsj_prepare(&m, ci, err, errlen))
        return -1;
    if (getenv("AC_TIME"))
        fprintf(stderr, "ac_lsj_prepare: %.2f s\n", omp_get_wtime() - t0);
    pc_begin(ci);
    int n = ci->ncsf, nel = 0, tjmax = 0, rc = -1;
    for (int s = 1; s <= ci->csf[0].c.nsh; s++)
        nel += ci->csf[0].c.nel[s];
    for (int i = 0; i < n; i++) {
        ac_lab a = total(ci, i);
        if (a.ts + a.tl > tjmax)
            tjmax = a.ts + a.tl;
    }
    int *idx = malloc((size_t)n * sizeof *idx);
    double *H = malloc((size_t)n * n * sizeof *H), *w = malloc((size_t)n * sizeof *w);
    level *all = malloc((size_t)(tjmax / 2 + 1) * (nlev < n ? nlev : n) * sizeof *all);
    int nall = 0, nkept = 0;
    kept *kp = NULL;
    if (!idx || !H || !w || !all) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    fprintf(out, "%4s %3s %22s %-6s %6s %16s %14s %14s %14s%s%s%s%s\n", "J", "#", "E (hartree)",
            "term", "weight", "g_J", "dg_RMC", "dg_GC", "dg_ORMC",
            ci->in.rgc ? "          dg_GC2e" : "", ci->in.roz ? "          dg_OZ2e" : "",
            ci->in.rrc ? "              rec2" : "", ac_lsj_h6_rk ? "       dg_H6/kappa" : "");
    for (int tJ = nel % 2; tJ <= tjmax; tJ += 2) {
        double asym, t1 = omp_get_wtime();
        int dim = ac_lsj_matrix(&m, ci, tJ, idx, H, &asym);
        if (dim == 0)
            continue;
        if (asym > 1e-10)
            fprintf(stderr, "warning: J block %d/2 asymmetric by %.1e\n", tJ, asym);
        double t2 = omp_get_wtime();
        int nd = nlev < dim ? nlev : dim, it = 0;
        if (ac_lsj_davidson > 0 && dim >= ac_lsj_davidson && 4 * nd < dim) {
            /* the nlev lowest only; their vectors into the first columns of H */
            double *x = malloc((size_t)dim * nd * sizeof *x);
            int e = x ? ac_ci_davidson(dim, H, nd, 1e-11, w, x, &it) : -1;
            if (e) {
                free(x);
                fail(err, errlen, "Davidson failed (%d) in the J = %d/2 block", e, tJ);
                goto out;
            }
            memcpy(H, x, (size_t)dim * nd * sizeof *x);
            free(x);
        } else if (ac_ci_eigvecs(dim, H, w)) {
            fail(err, errlen, "dsyev failed");
            goto out;
        }
        if (getenv("AC_TIME")) {
            fprintf(stderr, "J = %d/2, dim %d: matrix %.2f s, eigenvectors %.2f s", tJ, dim,
                    t2 - t1, omp_get_wtime() - t2);
            fprintf(stderr, it ? ", Davidson %d iterations\n" : "\n", it);
        }
        for (int p = 0; p < dim && p < nlev; p++) {
            const double *c = H + (size_t)p * dim;
            /* leading LS term */
            double best = 0.0, wsum[64] = {0};
            int key[64], nk = 0, lead = 0;
            for (int q = 0; q < dim; q++) {
                ac_lab a = total(ci, idx[q]);
                int k = a.ts * 1000 + a.tl, r;
                for (r = 0; r < nk && key[r] != k; r++)
                    ;
                if (r == nk && nk < 64)
                    key[nk++] = k;
                if (r < 64)
                    wsum[r] += c[q] * c[q];
            }
            for (int r = 0; r < nk; r++)
                if (wsum[r] > best)
                    best = wsum[r], lead = key[r];
            char term[16];
            snprintf(term, sizeof term, "%d%c", lead / 1000 + 1, "SPDFGHIKLMN"[(lead % 1000) / 2]);
            fprintf(out, "%2d/2 %3d %22.15f %-6s %6.4f", tJ, p + 1, w[p], term, best);
            if (tJ > 0) {
                ac_lsj_g gg;
                int e;
                if (ac_lsj_gfactor(ci, dim, idx, tJ, c, tkin, zr, ctx, &gg, &e)) {
                    fail(err, errlen, "g factor: engine error %d", e);
                    goto out;
                }
                fprintf(out, " %16.13f %14.6e %14.6e %14.6e", gg.gL + AC_GS * gg.gS, gg.rmc, gg.gc,
                        gg.ormc);
                if (ci->in.rgc)
                    fprintf(out, " %17.10e", gg.gc2e);
                if (ci->in.roz)
                    fprintf(out, " %17.10e", gg.oz2e);
                if (ci->in.rrc)
                    fprintf(out, " %17.10e", gg.rec2);
                if (ac_lsj_h6_rk)
                    fprintf(out, " %17.10e", gg.h6);
            }
            fprintf(out, "\n");
            all[nall].tJ = tJ;
            all[nall++].e = w[p];
            if ((ac_lsj_zoff || ac_lsj_hfs_rm3 || ac_lsj_trans_rk || ac_lsj_is_rk || keep) &&
                keep_level(&kp, &nkept, tJ, p + 1, w[p], dim, idx, c)) {
                fail(err, errlen, "out of memory");
                goto out;
            }
        }
    }
    if (ac_lsj_zoff && print_zoff(out, ci, kp, nkept, kp, nkept, 0, tkin, zr, ctx, err, errlen))
        goto out;
    if (ac_lsj_hfs_rm3 && print_hfs(out, ci, kp, nkept, kp, nkept, 0, ctx, err, errlen))
        goto out;
    if (ac_lsj_trans_rk && print_trans(out, ci, kp, nkept, kp, nkept, 0, ctx, err, errlen))
        goto out;
    if (ac_lsj_is_rk && print_is(out, ci, kp, nkept, ctx, err, errlen))
        goto out;
    qsort(all, (size_t)nall, sizeof *all, by_energy);
    fprintf(out, "levels relative to the lowest (cm-1):\n");
    for (int i = 0; i < nall; i++)
        fprintf(out, "  J = %2d/2  %16.6f\n", all[i].tJ, (all[i].e - all[0].e) * AC_HARTREE_CM);
    rc = 0;
    if (keep) {
        *keep = kp;
        *nkeep = nkept;
        kp = NULL;
        nkept = 0;
    }
out:
    pc_end();
    ac_lsj_kept_free(kp, nkept);
    free(all);
    free(idx);
    free(H);
    free(w);
    ac_lsj_free(&m);
    return rc;
}

/* the alpha^2 terms of g_J: Pachucki (2004), Eq. (32) (H1-H6); RMC and GC1e
   as in Neese (2001); the H6 term checked against Jentschura (2010), Eq. (10)
   (technical note, Sec. 5.1) */
int ac_lsj_zeeman_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                          double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                          void *ctx, ac_lsj_g *r, int *err)
{
    memset(r, 0, sizeof *r);
    *err = 0;
    if (!tri2(A->tJ, 2, B->tJ))
        return 0;
    pairs1 P;
    if (pairs1_build(ci, A, B, &P)) {
        *err = 1;
        return -1;
    }
    gop o = {0, NULL, NULL, NULL};
    r->gS = pair_reduced(ci, A, B, &P, &o, 2, 0, 2, err);
    o.kind = 1;
    if (!*err)
        r->gL = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
    if (!*err && tkin) {
        gop t = {2, tkin, ctx, NULL};
        r->rmc = pair_reduced(ci, A, B, &P, &t, 2, 0, 2, err);
        t.kind = 5;
        if (!*err)
            r->ormc = pair_reduced(ci, A, B, &P, &t, 0, 2, 2, err);
    }
    if (!*err && zr) {
        gop u = {3, zr, ctx, NULL};
        r->gc = pair_reduced(ci, A, B, &P, &u, 2, 0, 2, err);
        u.kind = 4;
        if (!*err)
            r->gc += pair_reduced(ci, A, B, &P, &u, 2, 4, 2, err);
    }
    if (!*err && ac_lsj_h6_rk) {
        /* -s^j T_jz = sqrt(5/3) [s x T^(2)]^(1)_z, T^(2) = [p x p]^(2) (kind 20) */
        gop h = {20, NULL, ctx, ac_lsj_h6_rk};
        r->h6 = ALPHA2 * sqrt(15.0) / 3.0 * pair_reduced(ci, A, B, &P, &h, 2, 4, 2, err);
    }
    pairs1_free(&P);
    for (int kp = 0; kp <= 2 && !*err && ci->in.rgc; kp++)
        r->gc2e += pair_reduced_gc2e(ci, A, B, kp, err);
    if (!*err && ci->in.roz)
        r->oz2e = pair_reduced_gc2e(ci, A, B, -1, err);
    if (!*err && ci->in.rrc)
        r->rec2 = pair_reduced_gc2e(ci, A, B, -2, err);
    return *err ? -1 : 0;
}

/* the hyperfine operators of Lindgren-Rosen (1974) */
int ac_lsj_hfs_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                       double (*rm3)(void *, int, int), double (*r00)(void *, int, int),
                       void *ctx, ac_lsj_hfs *h, int *err)
{
    memset(h, 0, sizeof *h);
    *err = 0;
    if (!tri2(A->tJ, 2, B->tJ) && !tri2(A->tJ, 4, B->tJ))
        return 0;
    pairs1 P;
    if (pairs1_build(ci, A, B, &P)) {
        *err = 1;
        return -1;
    }
    if (tri2(A->tJ, 2, B->tJ)) {
        gop o = {6, rm3, ctx, NULL};
        h->orb = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        o.kind = 7;
        if (!*err)
            h->sd = pair_reduced(ci, A, B, &P, &o, 2, 4, 2, err);
        gop c = {8, r00, ctx, NULL};
        if (!*err)
            h->con = pair_reduced(ci, A, B, &P, &c, 2, 0, 2, err);
    }
    if (!*err && tri2(A->tJ, 4, B->tJ)) {
        gop q = {9, rm3, ctx, NULL};
        h->quad = pair_reduced(ci, A, B, &P, &q, 0, 4, 4, err);
    }
    pairs1_free(&P);
    return *err ? -1 : 0;
}

int ac_lsj_is_level(ac_ci *ci, const ac_lsj_level *A, double (*rk)(void *, int, int, int),
                    void *ctx, ac_lsj_is *r, int *err)
{
    memset(r, 0, sizeof *r);
    *err = 0;
    pairs1 P;
    if (pairs1_build(ci, A, A, &P)) {
        *err = 1;
        return -1;
    }
    /* J-scalars: <J||O||J> = sqrt(2J+1) <O> */
    double norm = 1.0 / sqrt(A->tJ + 1.0);
    gop o = {21, NULL, ctx, rk};
    r->nms = norm * pair_reduced(ci, A, A, &P, &o, 0, 0, 0, err);
    o.kind = 22;
    if (!*err)
        r->rho0 = norm * pair_reduced(ci, A, A, &P, &o, 0, 0, 0, err);
    pairs1_free(&P);
    if (!*err)
        r->sms = norm * pair_reduced_2e(ci, A, A, -3, rk, ctx, err);
    return *err ? -1 : 0;
}

int ac_lsj_trans_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                         double (*rk)(void *, int, int, int), void *ctx, ac_lsj_trans *t,
                         int *err)
{
    memset(t, 0, sizeof *t);
    *err = 0;
    if (!tri2(A->tJ, 2, B->tJ) && !tri2(A->tJ, 4, B->tJ) && !tri2(A->tJ, 6, B->tJ))
        return 0;
    pairs1 P;
    if (pairs1_build(ci, A, B, &P)) {
        *err = 1;
        return -1;
    }
    gop o = {10, NULL, ctx, rk};
    if (tri2(A->tJ, 2, B->tJ)) {
        t->e1l = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        o.kind = 11;
        if (!*err)
            t->e1v = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        gop s = {0, NULL, NULL, NULL};
        double rs = 0.0, rl = 0.0;
        if (!*err)
            rs = pair_reduced(ci, A, B, &P, &s, 2, 0, 2, err);
        s.kind = 1;
        if (!*err)
            rl = pair_reduced(ci, A, B, &P, &s, 0, 2, 2, err);
        t->m1 = rl + trans_gs() * rs;
        double *m[3] = {&t->m1r2, &t->m1p2, &t->m1zr};
        for (int k = 0; k < 3 && !*err; k++) {
            o.kind = 13 + k;
            *m[k] = pair_reduced(ci, A, B, &P, &o, 2, 0, 2, err);
        }
        double *ml[2] = {&t->m1lr2, &t->m1lp2};
        for (int k = 0; k < 2 && !*err; k++) {
            o.kind = 16 + k;
            *ml[k] = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        }
        double *mt[3] = {&t->m1tr2, &t->m1tzr, &t->m1tpp};
        for (int k = 0; k < 3 && !*err; k++) {
            o.kind = 18 + k;
            *mt[k] = pair_reduced(ci, A, B, &P, &o, 2, 4, 2, err);
        }
        /* the two-electron terms, when their integrals are there */
        if (!*err && ci->in.rm1 && (ac_lsj_m1_2e_parts & 1)) {
            t->m1s0 = pair_reduced_gc2e(ci, A, B, 10, err);
            if (!*err)
                t->m1s2 = pair_reduced_gc2e(ci, A, B, 12, err);
        }
        if (!*err && ci->in.rm1 && ci->in.roz && (ac_lsj_m1_2e_parts & 2))
            t->m1oz = pair_reduced_gc2e(ci, A, B, -1, err);
    }
    if (!*err && tri2(A->tJ, 4, B->tJ)) {
        o.kind = 12;
        t->e2 = pair_reduced(ci, A, B, &P, &o, 0, 4, 4, err);
        double ml = 0.0, ms = 0.0;
        o.kind = 24;
        if (!*err)
            ml = pair_reduced(ci, A, B, &P, &o, 0, 4, 4, err);
        o.kind = 25;
        if (!*err)
            ms = pair_reduced(ci, A, B, &P, &o, 2, 2, 4, err);
        t->m2 = ml + ms;
    }
    if (!*err && tri2(A->tJ, 6, B->tJ)) {
        o.kind = 23;
        t->e3 = pair_reduced(ci, A, B, &P, &o, 0, 6, 6, err);
        double ml = 0.0, ms = 0.0;
        o.kind = 26;
        if (!*err)
            ml = pair_reduced(ci, A, B, &P, &o, 0, 6, 6, err);
        o.kind = 27;
        if (!*err)
            ms = pair_reduced(ci, A, B, &P, &o, 2, 4, 6, err);
        t->m3 = ml + ms;
    }
    pairs1_free(&P);
    return *err ? -1 : 0;
}

/* Pachucki (2004), Eqs. (59), (60), kappa = 0 */
double ac_lsj_m1_spin_scalar(const ac_lsj_trans *t, double w)
{
    return -4.0 * ALPHA2 * (w * w * t->m1r2 / 12.0 + t->m1p2 / 3.0 - t->m1zr / 6.0);
}

double ac_lsj_m1_rel1e(const ac_lsj_trans *t, double w)
{
    /* Pachucki (2004), Eqs. (61), (62).
       orbital: -2 T_vec -> -L (w^2 r^2/10 + p^2/2); tensor: -2 T^ij sigma_j =
       -4 (T.s) = 4 sqrt(5/3) [s x T^(2)]^(1), T^(2) = -(w^2/20) [r x r]^(2)
       - (1/8) [p x p]^(2) + (Z/4) [r x r]^(2)/r^3, [r x r]^(2) = sqrt(2/3) r^2 C^2
       (the delta_ij parts of Eq. (62) cancel) */
    return ac_lsj_m1_spin_scalar(t, w) +
           ALPHA2 * (-w * w * t->m1lr2 / 10.0 - t->m1lp2 / 2.0 -
                     sqrt(10.0) / 15.0 * w * w * t->m1tr2 + sqrt(10.0) / 3.0 * t->m1tzr -
                     sqrt(15.0) / 6.0 * t->m1tpp);
}

int ac_lsj_m1_2e_parts = 3;

double ac_lsj_m1_rel(const ac_lsj_trans *t, double w)
{
    /* Pachucki (2004), Eqs. (60)-(62). The two-electron terms carry alpha^2
       already (ac_ci_m1s2, ac_ci_oz2e:
       the orbital term of Eq. (61) is the orbit-orbit Zeeman operator) */
    return ac_lsj_m1_rel1e(t, w) + t->m1s0 + t->m1s2 + t->m1oz;
}

int ac_lsj_gfactor(ac_ci *ci, int dim, const int *idx, int tJ, const double *c,
                   double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                   void *ctx, ac_lsj_g *g, int *err)
{
    memset(g, 0, sizeof *g);
    *err = 0;
    if (tJ <= 0)
        return -1;
    /* <J J|T^1_0|J J> = J / sqrt(J(J+1)(2J+1)) <J||T||J>, then divided by J */
    double J = tJ / 2.0, norm = 1.0 / sqrt(J * (J + 1) * (2 * J + 1));
    ac_lsj_level a = {tJ, dim, idx, c};
    if (ac_lsj_zeeman_reduced(ci, &a, &a, tkin, zr, ctx, g, err))
        return -1;
    g->gL *= norm;
    g->gS *= norm;
    g->rmc *= norm;
    g->ormc *= norm;
    g->gc *= norm;
    g->gc2e *= norm;
    g->oz2e *= norm;
    g->rec2 *= norm;
    g->h6 *= norm;
    return 0;
}
