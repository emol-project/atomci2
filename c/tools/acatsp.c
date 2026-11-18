/*
 * acatsp -- the stage-2 LSJ CI on ATSP's numerical orbitals, for direct
 * comparison with ATSP (hf, nonh, bpci) in the same orbitals.
 *
 *   acatsp CFPTBL atom.w configs [nlev] [noso] [mvd] [alpha=INV]
 *
 * atom.w      ATSP orbitals (ac_tabrad); they are the slots of every CSF,
 *             coupled in the order of the file
 * configs     one configuration per line, e.g. "1s(2) 2s(2) 2p(5)";
 *             orbitals of the file not listed are empty. A name ending in
 *             ".c" is an ATSP CSF list instead: exactly those CSFs, coupled
 *             as ATSP couples them (read_atsp_c)
 * ls          print the eigenvalues of every (S, L) block (no J)
 * lag=l:N:b[,l:N:b...]
 *             add analytic orbitals to the .w set (add_laguerre): per l the
 *             N functions r^l e^(-br) L_k^(2l+2)(2br) (s: with the nuclear
 *             cusp), orthogonalized to the .w orbitals of that l and
 *             diagonalized on the bare hydrogen-like h; labels continue
 *             the n of the .w (the large s sets of hfsrel)
 * lagfock[=N] (with lag=, configuration lists) diagonalize the added
 *             orbitals on a screened operator in place of the bare h: h plus
 *             the direct and shell-averaged exchange potential of the first
 *             configuration of the list with N electrons (default 1) taken
 *             from its last shells. The added orbitals span the same space;
 *             the levels are the same to rounding when the list holds every
 *             distribution of its electrons over the added orbitals of each
 *             l (all of them used), and differ when only some are used. The
 *             Davidson iteration then needs several times fewer steps (Ti,
 *             12000 to 80000 CSFs: 5 to 8 times fewer products H x)
 * allslots    (configuration lists) every orbital of the file is a slot of
 *             every CSF, empty shells included (ac_ci_from_orbitals). The
 *             default takes only the occupied orbitals of a configuration as
 *             the slots of its CSFs (ac_ci_from_orbitals_occ): the same CSFs
 *             and the same results to rounding, and the engine's work no
 *             longer grows with the number of empty orbitals (about 6 times
 *             faster with 22 orbitals). occslots, the earlier name of the
 *             default, is still accepted
 * hfsrel      for the lowest root of every (S, L = 0) block, the regularized
 *             alpha^2 hfs correction A_AN/A^(4) of Puchalski and Pachucki
 *             (arXiv:1310.2161v2, Eq. (29); ac_hfsrel;
 *             technical note, Sec. 5.2). The CSF list must
 *             be the complete CI of its orbitals (configurations closed
 *             under single excitations) and the orbitals cusp-exact
 * delta12=X   with hfsrel: also A_AN/A4 with <sum delta(r_ab)> = X in the
 *             product term (the CI value lacks the e-e cusp)
 * writec=P    (configuration lists) also write the CSFs as ATSP .c files,
 *             one per (S, L): P_<2S+1><L>.c
 * noso, mvd   as for acfs (H_SO1 off; mass velocity + Darwin added to h)
 * so2         add the two-electron spin-orbit H_SO2 (ac_ci_so2)
 * ss          add the spin-spin H_SS (ac_ci_ss)
 * oo          add the orbit-orbit H_OO (ac_ci_oo)
 * gc2e        print Delta g of the two-electron gauge correction (ac_ci_gc2e);
 *             gc2epart=1 spin-same-orbit, =2 spin-other-orbit only
 * oz2e        print Delta g of the orbit-orbit Zeeman term (ac_ci_oz2e)
 * rec2        print <O_z>/M_J of the two-body recoil operator sum_{i!=j}
 *             r_i x p_j (ac_ci_rec2); the recoil correction to g is
 *             -(m/M)(g_L part + rec2)
 * h6          print Delta g of the rank-2 part of Pachucki's H6 (K. Pachucki,
 *             Phys. Rev. A 69, 052502 (2004), Eq. (32)) per unit
 *             kappa = (g_s - 2)/2 (ac_lsj_g h6; Delta g = kappa h6)
 * is          print the isotope-shift parameters of every printed level:
 *             K_NMS = <sum p_i^2/2>, K_SMS = <sum_{i<j} p_i.p_j> (ac_ci_sms;
 *             mass shift (m/M)(K_NMS + K_SMS)) and rho(0) = <sum delta(r_i)>
 *             (field shift (2 pi/3) Z rho(0) delta<r^2>) (ac_lsj_is_level)
 * zoff        also print the Zeeman reduced matrix elements <A||mu||B> between
 *             every pair of printed levels with |J_A - J_B| <= 1
 *             (ac_lsj_zeeman_reduced; intermediate-field Zeeman couplings)
 * hfs         also print the hyperfine reduced matrix elements <A||T^(K)||B>
 *             (orbital, spin-dipole, contact, quadrupole) for every pair of
 *             printed levels, and A_J per mu_I/I and B_J per Q in MHz
 *             (ac_lsj_hfs_reduced)
 * m1r2e       with trans: add the two-electron terms of the relativistic M1 to
 *             the M1r line (ac_ci_m1s2, and the orbit-orbit Zeeman operator);
 *             m1r2epart=1 spin only, =2 orbital only
 * trans       also print the E1 (length and velocity), M1, E2, M2, E3 and M3 lines between
 *             every pair of printed levels: line strengths, E1 oscillator
 *             strengths and A in s^-1 (ac_lsj_trans_reduced). Configurations
 *             of both parities may be given in one list; one orbital set
 *             serves both states.
 * so2part=N   with so2: 1 spin-same-orbit only, 2 spin-other-orbit only
 * atspcompat  leave out the two-body H_SO2 and H_SS elements between CSFs
 *             differing by ab -> cd over four distinct orbitals, which
 *             ATSP's bpci omits on the Si e1.bp-5.c list only (explicit
 *             closed core); not for other comparisons (ac_atsp_compat)
 * rkrel       add ATSP's relativistic correction to every R^k,
 *             (2k+1)(alpha^2/4) int R R R R r^2 dr (librad/ykf.f); with mvd
 *             this reproduces ATSP's relativistic shift in bpci. It is the
 *             two-electron Darwin + spin-spin contact terms
 *             (ac_radial_contact)
 * bio=W2,C2   a second state with its own orbitals W2 (.w) and
 *             configurations or CSF list C2 (S2-M9; the method of Malmqvist,
 *             Int. J. Quantum Chem. 30, 479 (1986), and Olsen et al.,
 *             Phys. Rev. E 52, 4499 (1995)): both CIs are solved
 *             on their own orbitals, the two orbital sets are made
 *             biorthonormal (ac_biorth_lu), every printed level is
 *             counter-transformed (ac_biorth_transform; the CSF lists must
 *             be closed under deexcitation, ac_biorth_closure), and the
 *             zoff, hfs and trans tables are printed between the levels of
 *             state 1 (bra) and state 2 (ket). The two states must order
 *             their common orbitals alike; per l the orbitals of one state
 *             must begin with those of the other. Not with ls. The
 *             two-electron terms between the states (m1r2e, and gc2e, oz2e,
 *             rec2 in the zoff table) take their integrals with the bra
 *             orbitals of state 1 and the ket orbitals of state 2.
 * biomode=N   how the pivots are split (ac_biorth_lu; 0 default)
 * biorthforce go on when a CSF list is not closed under deexcitation
 * digits      print the line strengths of trans with 17 digits
 * dav=N       J blocks of dimension >= N (default 500) by Davidson for the
 *             nlev lowest levels only; dav=0 always the full dsyev
 * orth        orthonormalize the orbitals of each l on this grid (Gram-
 *             Schmidt in the order of the .w file) before anything else;
 *             ATSP's orbitals, interpolated, are orthonormal only to ~1e-9
 * rot1=a,b,deg, rot2=a,b,deg
 *             rotate two orbitals of one l of state 1 (2) before the CI:
 *             P_a <- cos P_a + sin P_b, P_b <- -sin P_a + cos P_b (a test of
 *             the invariance of bio)
 * alpha=INV   1/alpha (default CODATA 2018; ATSP uses 137.036)
 * q=...       radial integrals in ATSP's notation, e.g. q=r,2p,-3
 *             q=F,2,2p,2p  q=G,1,2s,2p  q=R,1,1s,2p,2s,2p;
 *             q=N,k,i,j,ip,jp and q=V,k,i,j,ip,jp: N^k, V^k (ac_radial.h);
 *             q=D,a,b: int R_a (R_b' + (l_b+1) R_b/r) r^2 dr (ac_radial_rrc),
 *             for an s orbital a and a p orbital b sqrt(3) <a|d/dx|b_x>
 *
 * h, R^k, zeta, T and Z/r come from ac_radial on the default grid.
 */
#include "ac_biorth.h"
#include "ac_cfp.h"
#include "ac_ci.h"
#include "ac_hfsrel.h"
#include "ac_lsj.h"
#include "ac_radial.h"
#include "ac_tabrad.h"
#include "ac_wigner.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXO 128 /* orbitals (slots of a configuration list with occslots) */

/* AC_TIME: the wall time since the previous call, with a label, to stderr */
static void lap(const char *what)
{
    static double t0 = -1.0;
    double t = omp_get_wtime();
    if (getenv("AC_TIME") && t0 >= 0.0)
        fprintf(stderr, "%s: %.2f s\n", what, t - t0);
    t0 = t;
}
/* AC_TIME: the wall time since t0, with a label, to stderr; returns the time now */
static double since(const char *what, double t0)
{
    double t = omp_get_wtime();
    if (getenv("AC_TIME"))
        fprintf(stderr, "%s: %.2f s\n", what, t - t0);
    return t;
}
#define MAXK 13

typedef struct {
    const ac_radial *rad;
    const int *mo; /* [slot-1] -> MO */
    int mvd;
    int rkrel;     /* ATSP's relativistic Slater integrals */
    const int *mor; /* one-electron integrals: [slot-1] -> MO of the right
                       (ket) orbital, NULL: mo (bio: left A, right B) */
} ctx_t;

/* The radial-integral caches below are shared by the threads of the LSJ
   build (ac_lsj_prepare). A value is a double that is NaN until computed:
   it is read and written atomically, and two threads that both miss
   compute the same number. Arrays are allocated once under a critical
   section; Y^k and w_lam functions are published with a compare-and-swap
   (the loser frees its copy). */
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

/* A two-electron integral cache: values in pages of nsl^2 (c, d), one page
   per outer index (kind, k, a, b), allocated NaN-filled when first asked.
   Its memory grows with the (k, a, b) actually used, not with the
   MAXK x (kinds) x nsl^4 of a flat array (4 GB for every Breit-Pauli kind at
   32 slots). Pages are published with a compare-and-swap. */
typedef struct {
    double **dir;
    size_t n; /* outer indices */
} paged_t;

static double *page_at(paged_t *c, size_t nouter, size_t outer, size_t npage)
{
    double **d = __atomic_load_n(&c->dir, __ATOMIC_ACQUIRE);
    if (!d) {
#pragma omp critical(acatsp_alloc)
        {
            d = __atomic_load_n(&c->dir, __ATOMIC_ACQUIRE);
            if (!d) {
                d = calloc(nouter, sizeof *d);
                c->n = nouter;
                __atomic_store_n(&c->dir, d, __ATOMIC_RELEASE);
            }
        }
    }
    double *p = __atomic_load_n(&d[outer], __ATOMIC_ACQUIRE);
    if (p)
        return p;
    p = malloc(npage * sizeof *p);
    for (size_t i = 0; i < npage; i++)
        p[i] = NAN;
    double *expect = NULL;
    if (!__atomic_compare_exchange_n(&d[outer], &expect, p, 0, __ATOMIC_ACQ_REL,
                                     __ATOMIC_ACQUIRE)) {
        free(p);
        p = expect;
    }
    return p;
}

static void paged_free(paged_t *c)
{
    if (c->dir)
        for (size_t i = 0; i < c->n; i++)
            free(c->dir[i]);
    free(c->dir);
    c->dir = NULL;
    c->n = 0;
}

static double *nan_array(double **slot, size_t n)
{
    double *p = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (p)
        return p;
#pragma omp critical(acatsp_alloc)
    {
        p = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (!p) {
            p = malloc(n * sizeof(double));
            for (size_t i = 0; i < n; i++)
                p[i] = NAN;
            __atomic_store_n(slot, p, __ATOMIC_RELEASE);
        }
    }
    return p;
}

/* the MO of the ket (right) orbital of a slot: bio's state 2, else mo */
static int ket(const ctx_t *x, int b)
{
    return (x->mor ? x->mor : x->mo)[b - 1];
}

/* a grid function of a slot pair (c bra, d ket), computed once: kind 0 Y^k,
   1 w_lam, 2 u_lam */
static const double *pair_fn(double **slot, const ctx_t *x, int kind, int k, int lam, int c, int d)
{
    double *y = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (y)
        return y;
    y = malloc((size_t)x->rad->g->n * sizeof(double));
    if (kind == 0)
        ac_radial_yk(x->rad, k, x->mo[c - 1], ket(x, d), y);
    else if (kind == 1)
        ac_radial_ssw(x->rad, k, lam, x->mo[c - 1], ket(x, d), y);
    else
        ac_radial_m1w(x->rad, k, lam, x->mo[c - 1], ket(x, d), y);
    double *expect = NULL;
    if (!__atomic_compare_exchange_n(slot, &expect, y, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        free(y);
        y = expect;
    }
    return y;
}

static int nsl; /* slots of this run; the flat caches are [..][nsl]^n */

/* the cell of (kind, a, b, c, d) in a paged cache of nkind kinds (kind runs
   over k and the sub-kinds of the integral) */
static double *cell(paged_t *p, size_t nkind, size_t kind, int a, int b, int c, int d)
{
    size_t m = (size_t)nsl * nsl;
    return page_at(p, nkind * m, (kind * nsl + a - 1) * nsl + b - 1, m) +
           (size_t)(c - 1) * nsl + d - 1;
}

/* one-electron integrals of slot pairs, computed once: h (+ MV + Darwin),
   zeta, and the T and Z/r of the g factors */
enum {
    OE_H, OE_ZETA, OE_T, OE_U, OE_R3, OE_R00, OE_R1, OE_G1, OE_G2, OE_R2,
    OE_D0, OE_D1, OE_D2, OE_D3, /* ac_radial_dd, (sa, sb) = (-,-), (-,+), (+,-), (+,+) */
    OE_RP3,                     /* <r^3> (E3) */
    OE_N
};
static double *oec; /* [OE_N][nsl][nsl] */

static double one_e(const ctx_t *x, int kind, int a, int b)
{
    double *v = nan_array(&oec, (size_t)OE_N * nsl * nsl) + ((size_t)kind * nsl + a - 1) * nsl + b - 1;
    double r = ld(v);
    if (!isnan(r))
        return r;
    int i = x->mo[a - 1], j = ket(x, b);
    switch (kind) {
    case OE_H:
        r = ac_radial_h(x->rad, i, j);
        if (x->mvd)
            r += ac_radial_mv(x->rad, i, j) + ac_radial_darwin(x->rad, i, j);
        break;
    case OE_ZETA:
        r = ac_radial_zeta(x->rad, i, j);
        break;
    case OE_T:
        r = ac_radial_h(x->rad, i, j) + x->rad->z * ac_radial_moment(x->rad, i, j, -1);
        break;
    case OE_R3: /* hyperfine <r^-3> (s pairs never asked) */
        r = ac_radial_moment(x->rad, i, j, -3);
        break;
    case OE_R00: /* hyperfine contact R_a(0) R_b(0) */
        r = x->rad->r0[i] * x->rad->r0[j];
        break;
    case OE_R1: /* transitions: <r>, the gradient factors, <r^2> */
        r = ac_radial_moment(x->rad, i, j, 1);
        break;
    case OE_G1:
    case OE_G2:
        r = ac_radial_rrc(x->rad, kind == OE_G1 ? 1 : 2, i, j);
        break;
    case OE_R2:
        r = ac_radial_moment(x->rad, i, j, 2);
        break;
    case OE_RP3:
        r = ac_radial_moment(x->rad, i, j, 3);
        break;
    case OE_D0:
    case OE_D1:
    case OE_D2:
    case OE_D3: /* the relativistic M1 tensor term */
        r = ac_radial_dd(x->rad, i, (kind - OE_D0) & 2 ? 1 : -1, j, (kind - OE_D0) & 1 ? 1 : -1);
        break;
    default:
        r = x->rad->z * ac_radial_moment(x->rad, i, j, -1);
    }
    return st(v, r);
}

static double get_h(void *p, int a, int b)
{
    return one_e(p, OE_H, a, b);
}

static paged_t rkc; /* [MAXK][nsl]^4 */
static double *ykc[MAXK][MAXO][MAXO]; /* Y^k of slot pairs, computed once */

static double get_rk(void *p, int k, int a, int b, int c, int d)
{
    const ctx_t *x = p;
    if (k >= MAXK)
        return 0.0;
    double *v = cell(&rkc, MAXK, (size_t)k, a, b, c, d);
    double r = ld(v);
    if (isnan(r)) {
        int i = x->mo[a - 1], j = x->mo[b - 1], p = x->mo[c - 1], q = x->mo[d - 1];
        /* = ac_radial_rk(rad, k, i, j, p, q), its Y^k(a, b) shared (symmetric
           in a, b) */
        int lo = a < b ? a : b, hi = a < b ? b : a;
        r = ac_radial_dot(x->rad, p, q, pair_fn(&ykc[k][lo - 1][hi - 1], x, 0, k, 0, lo, hi));
        if (x->rkrel)
            r += ac_radial_contact(x->rad, k, i, j, p, q);
        st(v, r);
    }
    return r;
}

static double get_zeta(void *p, int a, int b)
{
    return one_e(p, OE_ZETA, a, b);
}

/* H_SO2 radial integrals, on the Y^k of get_rk */
static paged_t rkgc; /* values, [MAXK][2][2][nsl]^4 */

static double get_rkg(void *p, int k, int a, int sa, int b, int sb, int c, int d)
{
    const ctx_t *x = p;
    if (k >= MAXK)
        return 0.0;
    double *v = cell(&rkgc, MAXK * 4, (size_t)k * 4 + (sa > 0) * 2 + (sb > 0), a, b, c, d);
    double r = ld(v);
    if (!isnan(r))
        return r;
    const double *y = pair_fn(&ykc[k][c - 1][d - 1], x, 0, k, 0, c, d);
    return st(v, ac_radial_gdot(x->rad, x->mo[a - 1], sa, x->mo[b - 1], sb, y));
}

/* H_SS radial integrals, w_lam functions of slot pairs computed once */
static double *wsc[MAXK][3][MAXO][MAXO];
static paged_t rssc; /* values, [MAXK][3][nsl]^4 */

static double get_rss(void *p, int k, int lam, int a, int b, int c, int d)
{
    const ctx_t *x = p;
    if (k >= MAXK)
        return 0.0;
    double *v = cell(&rssc, MAXK * 3, (size_t)k * 3 + (lam - k + 2) / 2, a, b, c, d);
    double r = ld(v);
    if (!isnan(r))
        return r;
    const double *w = pair_fn(&wsc[k][(lam - k + 2) / 2][c - 1][d - 1], x, 1, k, lam, c, d);
    const ac_rgrid *g = x->rad->g;
    const double *va = x->rad->val + (size_t)x->mo[a - 1] * g->n;
    const double *vb = x->rad->val + (size_t)x->mo[b - 1] * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * va[i] * vb[i] * g->r[i] * g->r[i] * w[i];
    return st(v, s);
}

/* the spin two-electron M1 integrals (ac_ci_m1s2): u_lam functions of slot
   pairs computed once; lam = -1 the plain R^k (on the Y^k of get_rk, without
   rkrel's contact term). a, c bra and b, d ket orbitals (bio: A and B, so
   Y^k(a, b) is not symmetric then) */
static double *m1wc[MAXK][3][MAXO][MAXO];
static paged_t rm1c; /* values, [MAXK][4][nsl]^4 */

static double get_rm1(void *p, int k, int lam, int a, int b, int c, int d)
{
    const ctx_t *x = p;
    if (k >= MAXK)
        return 0.0;
    int li = lam < 0 ? 0 : (lam - k + 2) / 2 + 1;
    double *v = cell(&rm1c, MAXK * 4, (size_t)k * 4 + li, a, b, c, d);
    double r = ld(v);
    if (!isnan(r))
        return r;
    if (lam < 0) {
        int lo = a < b || x->mor ? a : b, hi = a < b || x->mor ? b : a;
        return st(v, ac_radial_dot(x->rad, x->mo[c - 1], ket(x, d),
                                   pair_fn(&ykc[k][lo - 1][hi - 1], x, 0, k, 0, lo, hi)));
    }
    const double *u = pair_fn(&m1wc[k][li - 1][c - 1][d - 1], x, 2, k, lam, c, d);
    const ac_rgrid *g = x->rad->g;
    const double *va = x->rad->val + (size_t)x->mo[a - 1] * g->n;
    const double *vb = x->rad->val + (size_t)ket(x, b) * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * va[i] * vb[i] * g->r[i] * g->r[i] * u[i];
    return st(v, s);
}

/* H_OO radial integrals; per k four kinds: g = k-1, k, k+1 and the
   divergence term (g = -1) */
static paged_t rooc; /* values, [MAXK][4][nsl]^4 */
/* the electron-1 potentials of ac_radial_oo, [MAXK][4][nsl][nsl] (computed
   once per (k, g, a, b); the electron-2 function is recomputed per value) */
static double **oopot;

static double get_roo(void *p, int k, int g, int a, int b, int c, int d)
{
    const ctx_t *x = p;
    if (k >= MAXK)
        return 0.0;
    int gi = g < 0 ? 3 : g - k + 1;
    double *v = cell(&rooc, MAXK * 4, (size_t)k * 4 + gi, a, b, c, d);
    double r = ld(v);
    if (!isnan(r))
        return r;
    /* = ac_radial_oo(rad, k, g, a, b, c, d), in the same operations */
    const ac_rgrid *gr = x->rad->g;
    int n = gr->n;
    if (!__atomic_load_n(&oopot, __ATOMIC_ACQUIRE)) {
#pragma omp critical(acatsp_alloc)
        if (!oopot)
            __atomic_store_n(&oopot, calloc((size_t)MAXK * 4 * nsl * nsl, sizeof(double *)),
                             __ATOMIC_RELEASE);
    }
    double **slot = oopot + (((size_t)k * 4 + gi) * nsl + a - 1) * nsl + b - 1;
    double *z = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    double *u = malloc((size_t)n * sizeof(double));
    int ia = x->mo[a - 1], ib = x->mo[b - 1], ic = x->mo[c - 1], id = x->mo[d - 1];
    if (!z) {
        z = malloc((size_t)n * sizeof(double));
        if (g >= 0) {
            ac_radial_ooj(x->rad, k, g, ia, ib, u);
            ac_radial_kern(x->rad, k, k + 1, u, z);
        } else {
            double *z2 = malloc((size_t)n * sizeof(double));
            ac_radial_ooq(x->rad, ia, ib, u);
            ac_radial_kern(x->rad, k + 2, k + 1, u, z);
            ac_radial_kern(x->rad, k, k - 1, u, z2);
            for (int i = 0; i < n; i++)
                z[i] = z[i] / (2 * k + 3) - z2[i] / (2 * k - 1);
            free(z2);
        }
        double *expect = NULL;
        if (!__atomic_compare_exchange_n(slot, &expect, z, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            free(z);
            z = expect;
        }
    }
    if (g >= 0)
        ac_radial_ooj(x->rad, k, g, ic, id, u);
    else
        ac_radial_ooq(x->rad, ic, id, u);
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += gr->w[i] * u[i] * z[i] * gr->r[i] * gr->r[i];
    free(u);
    return st(v, s);
}

/* the electron-1 kernel of a slot pair (a, b) for sub-kind kind of nkind
   ([nkind][nsl][nsl] pointers, allocated once; the kernel itself is
   published with a compare-and-swap, as pair_fn) */
static double **pot_slot(double ***tab, size_t nkind, size_t kind, int a, int b)
{
    double **t = __atomic_load_n(tab, __ATOMIC_ACQUIRE);
    if (!t) {
#pragma omp critical(acatsp_alloc)
        {
            t = __atomic_load_n(tab, __ATOMIC_ACQUIRE);
            if (!t) {
                t = calloc(nkind * nsl * nsl, sizeof *t);
                __atomic_store_n(tab, t, __ATOMIC_RELEASE);
            }
        }
    }
    return t + (kind * nsl + a - 1) * nsl + b - 1;
}

static double *publish(double **slot, double *z)
{
    double *expect = NULL;
    if (!__atomic_compare_exchange_n(slot, &expect, z, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        free(z);
        z = expect;
    }
    return z;
}

/* gauge-correction radial integrals; per k: kp = -1..2 x g = k-2..k+2.
   a, c bra and b, d ket orbitals (bio: A and B), as get_rm1 */
static paged_t rgcc; /* values, [MAXK][4][5][nsl]^4 */
/* the electron-1 kernels of ac_radial_gc, [MAXK][4][5][nsl][nsl] (computed
   once per (k, kp, g, a, b) instead of once per value, as oopot) */
static double **gcpot;

static double get_rgc(void *p, int k, int kp, int g, int a, int b, int c, int d)
{
    const ctx_t *x = p;
    if (k >= MAXK || abs(g - k) > 2)
        return 0.0;
    size_t kind = ((size_t)k * 4 + kp + 1) * 5 + g - k + 2;
    double *v = cell(&rgcc, MAXK * 20, kind, a, b, c, d);
    double r = ld(v);
    if (!isnan(r))
        return r;
    /* = ac_radial_gc(rad, k, kp, g, a, b, c, d), in the same operations */
    const ac_rgrid *gr = x->rad->g;
    int n = gr->n;
    double **slot = pot_slot(&gcpot, (size_t)MAXK * 20, kind, a, b);
    double *z = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (!z) {
        int ia = x->mo[a - 1], ib = ket(x, b);
        double *u = calloc((size_t)n, sizeof(double));
        z = malloc((size_t)n * sizeof(double));
        if (kp >= 0)
            ac_radial_gcj(x->rad, k, kp, g, ia, ib, u);
        else
            for (int i = 0; i < n; i++)
                u[i] = x->rad->val[(size_t)ia * n + i] * x->rad->val[(size_t)ib * n + i];
        ac_radial_kern(x->rad, k, k + 1, u, z);
        free(u);
        z = publish(slot, z);
    }
    const double *vc = x->rad->val + (size_t)x->mo[c - 1] * n;
    const double *vd = x->rad->val + (size_t)ket(x, d) * n;
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += gr->w[i] * vc[i] * vd[i] * z[i] * gr->r[i] * gr->r[i];
    return st(v, s);
}

/* orbit-orbit Zeeman radial integrals; per k: type 0..1 x g = k-1..k+1 */
static paged_t rozc; /* values, [MAXK][2][3][nsl]^4 */
/* the electron-1 kernels of ac_radial_oz, [MAXK][2][3][nsl][nsl], as gcpot */
static double **ozpot;

/* radial factors of the two-body recoil term (ac_ci_rec2) */
static double get_rrc(void *p, int kind, int a, int b)
{
    const ctx_t *x = p;
    return ac_radial_rrc(x->rad, kind, x->mo[a - 1], ket(x, b));
}

static double get_roz(void *p, int k, int g, int type, int a, int b, int c, int d)
{
    const ctx_t *x = p;
    if (k >= MAXK || abs(g - k) > 1)
        return 0.0;
    size_t kind = ((size_t)k * 2 + type) * 3 + g - k + 1;
    double *v = cell(&rozc, MAXK * 6, kind, a, b, c, d);
    double r = ld(v);
    if (!isnan(r))
        return r;
    /* = ac_radial_oz(rad, k, g, type, a, b, c, d), in the same operations */
    const ac_rgrid *gr = x->rad->g;
    int n = gr->n;
    int ia = x->mo[a - 1], ib = ket(x, b), ic = x->mo[c - 1], id = ket(x, d);
    double **slot = pot_slot(&ozpot, (size_t)MAXK * 6, kind, a, b);
    double *z = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (!z) {
        double *u = calloc((size_t)n, sizeof(double));
        z = malloc((size_t)n * sizeof(double));
        if (type == 0) {
            for (int i = 0; i < n; i++)
                u[i] = gr->r[i] * x->rad->val[(size_t)ia * n + i] * x->rad->val[(size_t)ib * n + i];
            ac_radial_kern(x->rad, k, k + 1, u, z);
        } else {
            double *z2 = malloc((size_t)n * sizeof(double));
            ac_radial_gcj(x->rad, k, 1, g, ia, ib, u);
            ac_radial_kern(x->rad, k + 2, k + 1, u, z);
            ac_radial_kern(x->rad, k, k - 1, u, z2);
            for (int i = 0; i < n; i++)
                z[i] = z[i] / (2 * k + 3) - z2[i] / (2 * k - 1);
            free(z2);
        }
        free(u);
        z = publish(slot, z);
    }
    double *w2 = malloc((size_t)n * sizeof(double));
    if (type == 0) {
        /* the current of cd with the sign of ac_radial_oz (-j, ac_radial.h) */
        ac_radial_ooj(x->rad, k, g, ic, id, w2);
        for (int i = 0; i < n; i++)
            w2[i] = -w2[i];
    } else
        ac_radial_ooq(x->rad, ic, id, w2);
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += gr->w[i] * w2[i] * z[i] * gr->r[i] * gr->r[i];
    free(w2);
    return st(v, s);
}

static double get_t(void *p, int a, int b)
{
    return one_e(p, OE_T, a, b);
}

static double get_u(void *p, int a, int b)
{
    return one_e(p, OE_U, a, b);
}

static double get_r3(void *p, int a, int b)
{
    return one_e(p, OE_R3, a, b);
}

static double get_r00(void *p, int a, int b)
{
    return one_e(p, OE_R00, a, b);
}

/* ac_lsj_trans_rk: kind 0 <r>, 1 and 2 the gradient factors, 3 <r^2>,
   4 <p^2> (twice T), 5 Z <1/r>, 6-9 ac_radial_dd (the M1 tensor term),
   10 R_a(0) R_b(0), 11 <r^3> */
static double get_trk(void *p, int kind, int a, int b)
{
    static const int oe[12] = {OE_R1, OE_G1, OE_G2, OE_R2, OE_T, OE_U,
                               OE_D0, OE_D1, OE_D2, OE_D3, OE_R00, OE_RP3};
    return (kind == 4 ? 2.0 : 1.0) * one_e(p, oe[kind], a, b);
}

/* the (2S+1, L, seniority) shell state "2P1" of l^n as a CFPTBL code */
static int shell_code(const ac_cfptab *t, int l, int n, const char *term, ac_lab *f)
{
    static const char *sym = "SPDFGHIKLMNOQ";
    if (strlen(term) < 2)
        return -1;
    int mult = term[0] - '0', tL;
    const char *p = strchr(sym, term[1]);
    if (mult < 1 || mult > 9 || !p)
        return -1;
    tL = 2 * (int)(p - sym);
    int v = isdigit((unsigned char)term[2]) ? term[2] - '0' : -1;
    const int *codes;
    int nc = ac_cfp_states(t, l, n, &codes), found = -1;
    for (int i = 0; i < nc; i++) {
        ac_shell_qn q;
        if (ac_cfp_decode(t, codes[i], &q))
            continue;
        if (q.ts != mult - 1 || q.tL != tL)
            continue;
        if (v >= 0 && 2 * l + 1 - q.tq != v)
            continue;
        if (found >= 0)
            return -2; /* not unique */
        found = codes[i];
    }
    f->ts = mult - 1;
    f->tl = tL;
    return found;
}

static int lab_of(const char *s, ac_lab *x)
{
    static const char *sym = "SPDFGHIKLMNOQ";
    while (*s == ' ')
        s++;
    const char *p = s[0] ? strchr(sym, s[1]) : NULL;
    if (!s[0] || !p || !isdigit((unsigned char)s[0]))
        return -1;
    x->ts = s[0] - '0' - 1;
    x->tl = 2 * (int)(p - sym);
    return 0;
}

static int w_index(const ac_atspw *w, const char *lab)
{
    for (int s = 0; s < w->n; s++)
        if (!strcmp(w->el[s], lab))
            return s;
    return -1;
}

/* An ATSP CSF list (.c): line 2 the closed shells, then per CSF a
   configuration line ("  3s( 2)  3p( 1)", 8 characters per shell) and a
   coupling line (4 characters per shell term "2P1", then per further shell
   the intermediate coupling "2P  "), ended by "*". The slots are the
   closed shells, then the open shells in one order consistent with the
   order of the shells in every CSF. slot_w[s] receives the w orbital of
   slot s (0-based). */
static int read_atsp_c(const char *path, const ac_atspw *w, const ac_cfptab *t, ac_ci *ci,
                       int *slot_w, char *err, int errlen)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(err, (size_t)errlen, "cannot open %s", path);
        return -1;
    }
    memset(ci, 0, sizeof *ci);
    ci->t = t;
    char line[2048], cfgl[2048];
    int nslot = 0, cap = 0, ln = 0, nclosed = 0;
#define FAIL(...)                                                                              \
    do {                                                                                       \
        snprintf(err, (size_t)errlen, __VA_ARGS__);                                            \
        fclose(f);                                                                             \
        return -1;                                                                             \
    } while (0)
    /* orbital label -> w index */
#define WIDX(lab, out)                                                                         \
    do {                                                                                       \
        if (((out) = w_index(w, lab)) < 0)                                                     \
            FAIL("%s: orbital %s not in the .w file", path, lab);                              \
    } while (0)
    if (!fgets(line, sizeof line, f) || !fgets(line, sizeof line, f))
        FAIL("%s: no header", path);
    for (char *tok = strtok(line, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
        int s;
        WIDX(tok, s);
        slot_w[nslot++] = s;
        nclosed++;
    }

    /* pass 1: the order of the open shells, from the order within every
       CSF (a topological sort, ties by first appearance) */
    long body = ftell(f);
    static unsigned char prec[AC_ATSP_MAXORB][AC_ATSP_MAXORB];
    int first[AC_ATSP_MAXORB], nopen = 0, open[AC_ATSP_MAXORB];
    memset(prec, 0, sizeof prec);
    for (int i = 0; i < AC_ATSP_MAXORB; i++)
        first[i] = -1;
    while (fgets(cfgl, sizeof cfgl, f) && cfgl[0] != '*') {
        if (!fgets(line, sizeof line, f))
            break;
        int prev = -1;
        for (char *p = cfgl; *p && *p != '\n'; p += 8) {
            char lab[8];
            int n, s;
            if (sscanf(p, " %7[^(](%d)", lab, &n) != 2)
                break;
            WIDX(lab, s);
            if (first[s] < 0) {
                first[s] = nopen;
                open[nopen++] = s;
            }
            if (prev >= 0)
                prec[prev][s] = 1;
            prev = s;
        }
    }
    for (int done = 0; done < nopen; done++) {
        int pick = -1;
        for (int i = 0; i < nopen && pick < 0; i++) {
            int s = open[i], ok = s >= 0;
            for (int j = 0; ok && j < nopen; j++)
                ok = !(open[j] >= 0 && prec[open[j]][s]);
            if (ok)
                pick = i;
        }
        if (pick < 0)
            FAIL("%s: the CSFs order their shells inconsistently", path);
        if (nslot == AC_CSF_MAXSH)
            FAIL("%s: more than %d orbitals", path, AC_CSF_MAXSH);
        slot_w[nslot++] = open[pick];
        open[pick] = -1;
    }
    fseek(f, body, SEEK_SET);

    /* pass 2: the CSFs */
    while (fgets(cfgl, sizeof cfgl, f)) {
        ln++;
        if (cfgl[0] == '*')
            break;
        if (!fgets(line, sizeof line, f))
            FAIL("%s: CSF %d has no coupling line", path, ci->ncsf + 1);
        /* shells of the configuration */
        int ns = 0, sw[AC_CSF_MAXSH], nn[AC_CSF_MAXSH];
        for (char *p = cfgl; *p && *p != '\n'; p += 8) {
            char lab[8];
            int n;
            if (sscanf(p, " %7[^(](%d)", lab, &n) != 2)
                break;
            if (ns == AC_CSF_MAXSH)
                FAIL("%s: too many shells", path);
            WIDX(lab, sw[ns]);
            nn[ns++] = n;
        }
        /* slots, keeping the global order */
        int slot[AC_CSF_MAXSH], last = nclosed - 1;
        for (int k = 0; k < ns; k++) {
            int s;
            for (s = nclosed; s < nslot && slot_w[s] != sw[k]; s++)
                ;
            if (s == nslot || s <= last)
                FAIL("%s: CSF %d lists its shells out of the global order", path, ci->ncsf + 1);
            slot[k] = last = s;
        }
        if (ci->ncsf == cap) {
            cap = cap ? 2 * cap : 256;
            ci->csf = realloc(ci->csf, (size_t)cap * sizeof *ci->csf);
        }
        ac_ci_csf *x = &ci->csf[ci->ncsf];
        memset(x, 0, sizeof *x);
        int k = 0;
        for (int c = 0; c < nclosed; c++) {
            int l = w->l[slot_w[c]];
            k++;
            x->seq[k] = c + 1;
            x->c.l[k] = l;
            x->c.nel[k] = 4 * l + 2;
            const int *codes;
            if (ac_cfp_states(t, l, 4 * l + 2, &codes) < 1)
                FAIL("%s: no closed %c shell in CFPTBL", path, "spdfgh"[l]);
            x->c.code[k] = codes[0];
        }
        ac_lab run = {0, 0};
        int len = (int)strlen(line);
        for (int j = 0; j < ns; j++) {
            char term[5] = {0};
            if (4 * j + 4 > len)
                FAIL("%s: CSF %d: short coupling line", path, ci->ncsf + 1);
            memcpy(term, line + 4 * j, 4);
            char *tp = term;
            while (*tp == ' ')
                tp++;
            int l = w->l[sw[j]];
            k++;
            x->seq[k] = slot[j] + 1;
            x->c.l[k] = l;
            x->c.nel[k] = nn[j];
            int code = shell_code(t, l, nn[j], tp, &x->c.f[k]);
            if (code < 0)
                FAIL("%s: CSF %d: shell term %s of %s(%d) %s", path, ci->ncsf + 1, tp,
                     w->el[sw[j]], nn[j], code == -2 ? "not unique" : "not in CFPTBL");
            x->c.code[k] = code;
            if (j == 0)
                run = x->c.f[k];
            else {
                char ib[5] = {0};
                int at = 4 * ns + 4 * (j - 1);
                if (at + 2 > len)
                    FAIL("%s: CSF %d: missing intermediate coupling", path, ci->ncsf + 1);
                memcpy(ib, line + at, 4);
                if (lab_of(ib, &run))
                    FAIL("%s: CSF %d: bad coupling '%s'", path, ci->ncsf + 1, ib);
            }
            x->c.cpl[k] = run;
        }
        for (int c = 1; c <= nclosed; c++)
            x->c.cpl[c].ts = x->c.cpl[c].tl = 0;
        x->c.nsh = k;
        ci->ncsf++;
    }
    fclose(f);
    ci->nwt = nslot;
    for (int s = 0; s < nslot; s++)
        ci->lorb[s + 1] = w->l[slot_w[s]];
    (void)ln;
#undef WIDX
#undef FAIL
    return 0;
}

/* Write the CSFs of ci, one ATSP .c file per (S, L): PREFIX_<2S+1><L>.c.
   Slots full in every CSF go to the closed-shell line; empty slots are
   left out. Shell terms are "2S+1 L seniority", seniority = 2l+1 - 2Q. */
static int write_atsp_c(const ac_ci *ci, const ac_atspw *w, const int *slot_w,
                        const char *prefix)
{
    static const char *sym = "SPDFGHIKLMNOQ";
    int n = ci->ncsf, nw = ci->nwt;
    int closed[MAXO + 1];
    for (int s = 1; s <= nw; s++) {
        closed[s] = 1;
        for (int i = 0; i < n && closed[s]; i++) {
            const ac_csf *c = &ci->csf[i].c;
            int k;
            for (k = 1; k <= c->nsh && ci->csf[i].seq[k] != s; k++)
                ;
            closed[s] = k <= c->nsh && c->nel[k] == 4 * c->l[k] + 2;
        }
    }
    int *done = calloc((size_t)n, sizeof *done);
    for (int a = 0; a < n; a++) {
        if (done[a])
            continue;
        ac_lab T = ci->csf[a].c.cpl[ci->csf[a].c.nsh];
        char name[512];
        snprintf(name, sizeof name, "%s_%d%c.c", prefix, T.ts + 1, sym[T.tl / 2]);
        FILE *f = fopen(name, "w");
        if (!f)
            return -1;
        fprintf(f, "%64s\n", "");
        for (int s = 1; s <= nw; s++)
            if (closed[s])
                fprintf(f, "%4s", w->el[slot_w[s - 1]]);
        fprintf(f, "\n");
        int count = 0;
        for (int b = a; b < n; b++) {
            const ac_csf *c = &ci->csf[b].c;
            ac_lab U = c->cpl[c->nsh];
            if (done[b] || U.ts != T.ts || U.tl != T.tl)
                continue;
            done[b] = 1;
            count++;
            char cfg[512] = "", cpl[512] = "", mid[512] = "";
            int first = 1;
            for (int k = 1; k <= c->nsh; k++) {
                int s = ci->csf[b].seq[k];
                if (closed[s] || c->nel[k] == 0)
                    continue;
                char buf[32];
                snprintf(buf, sizeof buf, "%4s(%2d)", w->el[slot_w[s - 1]], c->nel[k]);
                strcat(cfg, buf);
                ac_shell_qn q;
                ac_cfp_decode(ci->t, c->code[k], &q);
                int v = 2 * c->l[k] + 1 - q.tq;
                snprintf(buf, sizeof buf, " %d%c%d", c->f[k].ts + 1, sym[c->f[k].tl / 2], v);
                strcat(cpl, buf);
                if (!first) {
                    snprintf(buf, sizeof buf, " %d%c ", c->cpl[k].ts + 1, sym[c->cpl[k].tl / 2]);
                    strcat(mid, buf);
                }
                first = 0;
            }
            fprintf(f, "%s\n%s%s\n", cfg, cpl, mid);
        }
        fprintf(f, "*\n");
        fclose(f);
        printf("wrote %s (%d CSFs)\n", name, count);
    }
    free(done);
    return 0;
}

/* CSFs i and j have the same orbitals with the same occupations */
static int ls_same_config(const ac_ci *ci, int i, int j)
{
    const ac_ci_csf *F = &ci->csf[i], *G = &ci->csf[j];
    if (F->c.nsh != G->c.nsh)
        return 0;
    for (int s = 1; s <= F->c.nsh; s++)
        if (F->seq[s] != G->seq[s] || F->c.nel[s] != G->c.nel[s])
            return 0;
    return 1;
}

/* the number of electrons CSF j must move to become CSF i */
static int ls_moved(const ac_ci *ci, int i, int j)
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

/* eigenvalues of every (S, L) block */
static int print_ls(ac_ci *ci, int nroots)
{
    int n = ci->ncsf;
    int *done = calloc((size_t)n, sizeof *done), *idx = malloc((size_t)n * sizeof *idx);
    for (int a = 0; a < n; a++) {
        if (done[a])
            continue;
        ac_lab T = ci->csf[a].c.cpl[ci->csf[a].c.nsh];
        int m = 0;
        for (int b = a; b < n; b++) {
            ac_lab U = ci->csf[b].c.cpl[ci->csf[b].c.nsh];
            if (!done[b] && U.ts == T.ts && U.tl == T.tl)
                idx[m++] = b, done[b] = 1;
        }
        double *H = calloc((size_t)m * m, sizeof *H), *ev = malloc((size_t)m * sizeof *ev);
        double t0 = omp_get_wtime();
        /* columns in parallel on copies of ci, as the rows of ac_lsj_prepare:
           each element is computed by one thread exactly as in a serial run */
        int fe = 0;
        long ops = 0;
        /* The CSFs of the term that belong to one configuration are
           neighbours in idx: seg[s] .. seg[s+1]-1 (at most AC_BP_BLOCK of
           them). The elements of a pair of such segments are computed
           together (ac_ci_hij_list: the same values); AC_BP_PAIRWISE in the
           environment computes them one by one, as before. */
        const char *sb = getenv("AC_BP_BLOCK");
        int maxrun = sb && atoi(sb) > 0 ? atoi(sb) : 512, nseg = 0;
        int *seg = getenv("AC_BP_PAIRWISE") ? NULL : malloc(((size_t)m + 1) * sizeof *seg);
        int *sp = NULL;
        long nsp = 0;
        if (seg) {
            for (int q = 0; q < m; q++)
                if (q == 0 || !ls_same_config(ci, idx[q - 1], idx[q]) || q - seg[nseg - 1] >= maxrun)
                    seg[nseg++] = q;
            seg[nseg] = m;
            sp = malloc(((size_t)nseg * ((size_t)nseg + 1) + 1) * sizeof *sp);
            for (int a2 = 0; a2 < nseg && sp; a2++)
                for (int b2 = 0; b2 <= a2; b2++)
                    if (ls_moved(ci, idx[seg[a2]], idx[seg[b2]]) <= 2)
                        sp[2 * nsp] = a2, sp[2 * nsp + 1] = b2, nsp++;
        }
#pragma omp parallel reduction(+ : ops)
        {
            ac_ci loc = *ci;
            loc.thc = NULL;
            loc.ops_evaluated = 0;
            if (sp) {
                double *buf = NULL;
                size_t bufn = 0;
#pragma omp for schedule(dynamic, 1)
                for (long g = 0; g < nsp; g++) {
                    int a2 = sp[2 * g], b2 = sp[2 * g + 1], e = 0;
                    int nf = seg[a2 + 1] - seg[a2], ng = seg[b2 + 1] - seg[b2];
                    if ((size_t)nf * (size_t)ng > bufn) {
                        free(buf);
                        bufn = (size_t)nf * (size_t)ng;
                        buf = malloc(bufn * sizeof *buf);
                    }
                    if (!buf)
                        e = 1;
                    else
                        ac_ci_hij_list(&loc, nf, idx + seg[a2], ng, idx + seg[b2], buf, &e);
                    if (e) {
#pragma omp critical(acatsp_ls_fail)
                        fe = e;
                        continue;
                    }
                    for (int i = seg[a2]; i < seg[a2 + 1]; i++)
                        for (int j = seg[b2]; j < seg[b2 + 1] && j <= i; j++)
                            H[(size_t)j * m + i] =
                                buf[(size_t)(i - seg[a2]) * (size_t)ng + (size_t)(j - seg[b2])];
                }
                free(buf);
            } else {
#pragma omp for schedule(dynamic, 1)
                for (int j = 0; j < m; j++)
                    for (int i = j; i < m; i++) {
                        int e;
                        H[(size_t)j * m + i] = ac_ci_hij(&loc, idx[i], idx[j], &e);
                        if (e) {
#pragma omp critical(acatsp_ls_fail)
                            fe = e;
                            break;
                        }
                    }
            }
            ops += loc.ops_evaluated;
            ac_ci_free_cache(&loc);
        }
        free(seg);
        free(sp);
        ci->ops_evaluated += ops;
        if (fe) {
            fprintf(stderr, "acatsp: engine error %d\n", fe);
            return -1;
        }
        double t1 = omp_get_wtime();
        if (ac_ci_eigvals(m, H, ev))
            return -1;
        if (getenv("AC_TIME"))
            fprintf(stderr, "term %d%c, dim %d: matrix %.2f s, eigenvalues %.2f s\n", T.ts + 1,
                    T.tl / 2 < 21 ? "SPDFGHIKLMNOQRTUVWXYZ"[T.tl / 2] : '?', m, t1 - t0,
                    omp_get_wtime() - t1);
        printf("term %d%c %d", T.ts + 1, T.tl / 2 < 21 ? "SPDFGHIKLMNOQRTUVWXYZ"[T.tl / 2] : '?', m);
        for (int r = 0; r < nroots && r < m; r++)
            printf(" %.12f", ev[r]);
        printf("\n");
        free(H);
        free(ev);
    }
    free(done);
    free(idx);
    return 0;
}

/* every radial-integral cache, before the integrals of another orbital set */
static void reset_caches(void)
{
    free(oec);
    oec = NULL;
    paged_t *pg[] = {&rkc, &rkgc, &rssc, &rooc, &rgcc, &rozc, &rm1c};
    const char *name[] = {"R^k", "SO2", "SS", "OO", "GC2e", "OZ2e", "M1s2e"};
    for (size_t i = 0; i < sizeof pg / sizeof *pg; i++) {
        if (getenv("AC_TIME") && pg[i]->dir) {
            size_t used = 0;
            for (size_t j = 0; j < pg[i]->n; j++)
                used += pg[i]->dir[j] != NULL;
            double page = (double)nsl * nsl * sizeof(double) / 1048576.0;
            fprintf(stderr, "cache %s: %zu of %zu pages, %.1f MB (flat %.1f MB)\n", name[i], used,
                    pg[i]->n, used * page, pg[i]->n * page);
        }
        paged_free(pg[i]);
    }
    double **pots[] = {oopot, gcpot, ozpot};
    size_t npot[] = {MAXK * 4, MAXK * 20, MAXK * 6};
    for (int j = 0; j < 3; j++)
        if (pots[j]) {
            for (size_t i = 0; i < npot[j] * nsl * nsl; i++)
                free(pots[j][i]);
            free(pots[j]);
        }
    oopot = gcpot = ozpot = NULL;
    for (int k = 0; k < MAXK; k++)
        for (int c = 0; c < MAXO; c++)
            for (int d = 0; d < MAXO; d++) {
                free(ykc[k][c][d]);
                ykc[k][c][d] = NULL;
                for (int j = 0; j < 3; j++) {
                    free(wsc[k][j][c][d]);
                    wsc[k][j][c][d] = NULL;
                    free(m1wc[k][j][c][d]);
                    m1wc[k][j][c][d] = NULL;
                }
            }
}

/* one state: its orbitals, CSFs and the context of its integrals */
typedef struct {
    const char *wpath, *cpath;
    int csfmode;
    ac_atspw w;
    ac_radial rad;
    int mo[AC_ATSP_MAXORB];   /* orbital of w -> MO of rad */
    ac_ci ci;
    int ncfg;
    int slot_w[MAXO]; /* slot (0-based) -> orbital of w */
    int slotmo[MAXO]; /* slot (0-based) -> MO */
    ctx_t x;
} state_t;

static int load_w(state_t *st, const ac_rgrid *g, char *err, int errlen)
{
    if (ac_atspw_read(&st->w, st->wpath, err, errlen))
        return -1;
    if (st->w.n > MAXO) {
        snprintf(err, (size_t)errlen, "%d orbitals, at most %d", st->w.n, MAXO);
        return -1;
    }
    size_t alen = strlen(st->cpath);
    st->csfmode = alen > 2 && !strcmp(st->cpath + alen - 2, ".c");
    if (ac_radial_from_atsp(&st->rad, &st->w, g, st->mo, err, errlen)) {
        char e2[256];
        snprintf(e2, sizeof e2, "radial: %s", err);
        snprintf(err, (size_t)errlen, "%s", e2);
        return -1;
    }
    return 0;
}

/* ---- lag=: analytic orbitals added to the .w set (S2-M11 C2) ------------- */

/* nb raw functions R_k = r^l e^(-br) L_k^(2l+2)(2br) on the grid, with R', the
   radial Laplacian (as r^l (Q'' + 2(l+1) Q'/r), Q = R/r^l, free of
   cancellation at small r) and R(0); f[k] = R'(0) + Z R(0) (s only) */
static int laguerre_raw(ac_radial *raw, const ac_rgrid *g, double z, int l, int nb, double b,
                        double *f)
{
    memset(raw, 0, sizeof *raw);
    int n = g->n, a = 2 * l + 2;
    raw->g = g;
    raw->z = z;
    raw->nmo = nb;
    raw->l = malloc((size_t)nb * sizeof(int));
    raw->idx = malloc((size_t)nb * sizeof(int));
    raw->val = malloc((size_t)nb * n * sizeof(double));
    raw->der = malloc((size_t)nb * n * sizeof(double));
    raw->lap = malloc((size_t)nb * n * sizeof(double));
    raw->r0 = calloc((size_t)nb, sizeof(double));
    double *L = malloc((size_t)3 * (nb + 1) * sizeof(double));
    if (!raw->l || !raw->idx || !raw->val || !raw->der || !raw->lap || !raw->r0 || !L) {
        free(L);
        return -1;
    }
    for (int lv = 0; lv < AC_BASIS_MAXL; lv++)
        raw->first[lv] = lv <= l ? 0 : nb;
    for (int k = 0; k < nb; k++)
        raw->l[k] = l, raw->idx[k] = k;
    for (int p = 0; p < n; p++) {
        double r = g->r[p], t = 2.0 * b * r, e = exp(-b * r), rl = pow(r, l);
        /* L[j][k] = L_k^(a+j)(t), j = 0, 1, 2 */
        for (int j = 0; j < 3; j++) {
            double *Lj = L + (size_t)j * (nb + 1), aj = a + j;
            Lj[0] = 1.0;
            if (nb > 1)
                Lj[1] = 1.0 + aj - t;
            for (int k = 1; k + 1 < nb; k++)
                Lj[k + 1] = ((2 * k + 1 + aj - t) * Lj[k] - (k + aj) * Lj[k - 1]) / (k + 1);
        }
        for (int k = 0; k < nb; k++) {
            double L0 = L[k], L1 = k >= 1 ? -2.0 * b * L[(nb + 1) + k - 1] : 0.0,
                   L2 = k >= 2 ? 4.0 * b * b * L[2 * (nb + 1) + k - 2] : 0.0;
            double Q = e * L0, Q1 = e * (L1 - b * L0), Q2 = e * (L2 - 2.0 * b * L1 + b * b * L0);
            raw->val[(size_t)k * n + p] = rl * Q;
            raw->der[(size_t)k * n + p] = (l ? l * rl / r * Q : 0.0) + rl * Q1;
            raw->lap[(size_t)k * n + p] = rl * (Q2 + 2.0 * (l + 1) * Q1 / r);
        }
    }
    free(L);
    for (int k = 0; k < nb; k++) {
        /* L_k^(a)(0) = binom(k+a, k) */
        double c0 = 1.0, c1 = 0.0;
        for (int i = 1; i <= k; i++)
            c0 *= (double)(i + a) / i;
        if (k >= 1) {
            c1 = 1.0;
            for (int i = 1; i <= k - 1; i++)
                c1 *= (double)(i + a + 1) / i;
            c1 *= -2.0 * b;
        }
        if (l == 0) {
            raw->r0[k] = c0;
            f[k] = (c1 - b * c0) + z * c0;
        }
    }
    return 0;
}

/* lagthr=: the canonical orthonormalization of add_laguerre drops the
   directions whose overlap eigenvalue is below lagthr x the largest */
static double lagthr = 1e-10;

static double gdot(const ac_rgrid *g, const double *u, const double *v)
{
    double s = 0.0;
    for (int p = 0; p < g->n; p++)
        s += g->w[p] * u[p] * v[p] * g->r[p] * g->r[p];
    return s;
}

/* lagfit: before add_laguerre orthogonalizes the added functions, the .w
   orbitals of that l are replaced by their least-squares projections onto
   the span of the raw functions (s: the cusp subspace), Gram-Schmidt
   orthonormalized in the order of the .w. Then every orbital of that l lies
   in one analytic, cusp-exact space and the added ones are its exact
   complement: a tabulated orbital that is almost in the span otherwise
   leaves near-null directions whose normalization amplifies its
   interpolation errors. */
static int lagfit;

/* lagfock[=N]: the added functions are diagonalized on a screened
   one-electron operator in place of the bare h: h plus the direct and the
   shell-averaged exchange potential (Slater (1960), average of configuration)
   of the first configuration of the list with N electrons taken from its
   last shells (default 1). The span is the same; a CSF space that holds
   every distribution of its electrons over the added orbitals of each l
   gives the same levels. The CI matrix is then much nearer to diagonal, and
   the Davidson iteration needs fewer steps. -1: off. */
static int lagfock = -1;

/* occ[s] = the occupation of w orbital s in the first configuration of the
   list, with nrem electrons taken from its last shells */
static int fock_occupation(const state_t *st, int nrem, int *occ, char *err, int errlen)
{
    const ac_atspw *w = &st->w;
    if (st->csfmode) {
        snprintf(err, (size_t)errlen, "lagfock needs a configuration list, not an ATSP .c list");
        return -1;
    }
    FILE *cf = fopen(st->cpath, "r");
    if (!cf) {
        snprintf(err, (size_t)errlen, "%s: %s", st->cpath, strerror(errno));
        return -1;
    }
    char line[1024];
    int any = 0, order[MAXO], no = 0;
    for (int s = 0; s < w->n; s++)
        occ[s] = 0;
    while (!any && fgets(line, sizeof line, cf)) {
        char *hash = strchr(line, '#');
        if (hash)
            *hash = 0;
        for (char *tok = strtok(line, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
            char lab[8];
            int n, s;
            if (sscanf(tok, "%7[^(](%d)", lab, &n) != 2 || (s = w_index(w, lab)) < 0) {
                snprintf(err, (size_t)errlen, "lagfock: %s of the first configuration is not in %s",
                         tok, st->wpath);
                fclose(cf);
                return -1;
            }
            occ[s] = n;
            order[no++] = s;
            any = 1;
        }
    }
    fclose(cf);
    if (!any) {
        snprintf(err, (size_t)errlen, "lagfock: no configuration in %s", st->cpath);
        return -1;
    }
    for (int i = no - 1; i >= 0 && nrem > 0; i--)
        while (occ[order[i]] > 0 && nrem > 0)
            occ[order[i]]--, nrem--;
    printf("lagfock: the added orbitals are diagonalized on h + the potential of");
    for (int i = 0; i < no; i++)
        if (occ[order[i]])
            printf(" %s(%d)", w->el[order[i]], occ[order[i]]);
    printf("\n");
    return 0;
}

/* hp[i * m0 + j] += <i| V |j> for the m0 functions of proj (all of one l):
   V = sum_c q_c (J_c - (1/2) sum_k (l k l_c; 0 0 0)^2 K_c^k) */
static int fock_add(const state_t *st, const ac_radial *proj, int l, int m0, const int *occ,
                    double *hp)
{
    const ac_rgrid *g = st->rad.g;
    const ac_atspw *w = &st->w;
    int n = g->n;
    double *vd = calloc((size_t)n, sizeof *vd), *u = malloc((size_t)n * sizeof *u);
    double *z = malloc((size_t)n * sizeof *z), *x = calloc((size_t)m0 * m0, sizeof *x);
    if (!vd || !u || !z || !x) {
        free(vd);
        free(u);
        free(z);
        free(x);
        return -1;
    }
    for (int s = 0; s < w->n; s++) {
        if (!occ[s])
            continue;
        const double *rc = st->rad.val + (size_t)st->mo[s] * n;
        int lc = w->l[s];
        for (int p = 0; p < n; p++)
            u[p] = rc[p] * rc[p];
        ac_radial_kern(&st->rad, 0, 1, u, z);
        for (int p = 0; p < n; p++)
            vd[p] += occ[s] * z[p];
        for (int k = abs(l - lc); k <= l + lc; k += 2) {
            double f = ac_fabc(l, k, lc), c = -0.5 * occ[s] * f * f;
            for (int j = 0; j < m0; j++) {
                const double *pj = proj->val + (size_t)j * n;
                for (int p = 0; p < n; p++)
                    u[p] = rc[p] * pj[p];
                ac_radial_kern(&st->rad, k, k + 1, u, z);
                for (int p = 0; p < n; p++)
                    z[p] *= rc[p];
                for (int i = 0; i < m0; i++)
                    x[(size_t)i * m0 + j] += c * gdot(g, proj->val + (size_t)i * n, z);
            }
        }
    }
    for (int j = 0; j < m0; j++) {
        const double *pj = proj->val + (size_t)j * n;
        for (int p = 0; p < n; p++)
            u[p] = vd[p] * pj[p];
        for (int i = 0; i < m0; i++)
            x[(size_t)i * m0 + j] += gdot(g, proj->val + (size_t)i * n, u);
    }
    for (int i = 0; i < m0; i++)
        for (int j = 0; j < m0; j++)
            hp[(size_t)i * m0 + j] += 0.5 * (x[(size_t)i * m0 + j] + x[(size_t)j * m0 + i]);
    free(vd);
    free(u);
    free(z);
    free(x);
    return 0;
}

/* the coefficient vectors of the cusp subspace in the raw functions:
   column j of C (nb x m0), e_k - (f_k/f_piv) e_piv for s, e_k otherwise */
static double *cusp_basis(int l, int nb, const double *f, int piv, int *m0)
{
    *m0 = l == 0 ? nb - 1 : nb;
    double *C = calloc((size_t)nb * *m0, sizeof *C);
    for (int j = 0, k = 0; k < nb; k++) {
        if (l == 0 && k == piv)
            continue;
        C[(size_t)j * nb + k] = 1.0;
        if (l == 0)
            C[(size_t)j * nb + piv] = -f[k] / f[piv];
        j++;
    }
    return C;
}

static int fit_orbitals(state_t *st, const ac_radial *raw, const double *f, int piv, int l,
                        const int *ia, int na, char *err, int errlen)
{
    const ac_rgrid *g = st->rad.g;
    int nb = raw->nmo, m0;
    double *C = cusp_basis(l, nb, f, piv, &m0);
    /* S0 = C^T S_BB C, s0 = C^T S_BA */
    double *sbb = malloc((size_t)nb * nb * sizeof *sbb), *sba = malloc((size_t)nb * na * sizeof *sba);
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < nb; j++)
            sbb[(size_t)i * nb + j] = gdot(g, raw->val + (size_t)i * g->n, raw->val + (size_t)j * g->n);
    for (int i = 0; i < nb; i++)
        for (int a = 0; a < na; a++)
            sba[(size_t)i * na + a] =
                gdot(g, raw->val + (size_t)i * g->n, st->rad.val + (size_t)ia[a] * g->n);
    double *S0 = calloc((size_t)m0 * m0, sizeof *S0), *s0 = calloc((size_t)m0 * na, sizeof *s0);
    for (int p = 0; p < m0; p++)
        for (int q = 0; q < m0; q++) {
            double s = 0.0;
            for (int i = 0; i < nb; i++)
                for (int j = 0; j < nb; j++)
                    s += C[(size_t)p * nb + i] * sbb[(size_t)i * nb + j] * C[(size_t)q * nb + j];
            S0[(size_t)q * m0 + p] = s;
        }
    for (int p = 0; p < m0; p++)
        for (int a = 0; a < na; a++) {
            double s = 0.0;
            for (int i = 0; i < nb; i++)
                s += C[(size_t)p * nb + i] * sba[(size_t)i * na + a];
            s0[(size_t)p * na + a] = s;
        }
    /* y = S0^-1 s0 by the eigenvectors of S0 */
    double *ev = malloc((size_t)m0 * sizeof *ev), *y = calloc((size_t)m0 * na, sizeof *y);
    if (ac_ci_eigvecs(m0, S0, ev)) {
        snprintf(err, (size_t)errlen, "lagfit: dsyev failed");
        return -1;
    }
    for (int c = 0; c < m0; c++) {
        if (ev[c] <= 1e-14 * ev[m0 - 1])
            continue;
        for (int a = 0; a < na; a++) {
            double t = 0.0;
            for (int p = 0; p < m0; p++)
                t += S0[(size_t)c * m0 + p] * s0[(size_t)p * na + a];
            for (int p = 0; p < m0; p++)
                y[(size_t)p * na + a] += S0[(size_t)c * m0 + p] * t / ev[c];
        }
    }
    /* the fits as coefficients on the raw functions; residual 1 - <A|P A> */
    double *cf = calloc((size_t)na * nb, sizeof *cf);
    int *ls = malloc((size_t)na * sizeof *ls);
    printf("lagfit: l = %d, 1 - |P phi|^2 of the .w orbitals:", l);
    for (int a = 0; a < na; a++) {
        double nr = 0.0;
        for (int p = 0; p < m0; p++) {
            nr += s0[(size_t)p * na + a] * y[(size_t)p * na + a];
            for (int i = 0; i < nb; i++)
                cf[(size_t)a * nb + i] += C[(size_t)p * nb + i] * y[(size_t)p * na + a];
        }
        printf(" %.1e", 1.0 - nr);
        ls[a] = l;
    }
    printf("\n");
    ac_radial fit, gs;
    const ac_radial *s1[1] = {raw};
    int rc = ac_radial_mix(&fit, 1, s1, na, ls, cf, err, errlen);
    free(C);
    free(sbb);
    free(sba);
    free(S0);
    free(s0);
    free(ev);
    free(y);
    free(cf);
    if (rc) {
        free(ls);
        return -1;
    }
    /* Gram-Schmidt in the order of the .w */
    double *T = calloc((size_t)na * na, sizeof *T);
    for (int a = 0; a < na; a++) {
        T[(size_t)a * na + a] = 1.0;
        for (int b = 0; b < a; b++) { /* subtract <b'|a> b', b' = sum T[b][.] fit */
            double s = 0.0;
            for (int q = 0; q <= b; q++)
                s += T[(size_t)b * na + q] * ac_radial_moment(&fit, q, a, 0);
            for (int q = 0; q <= b; q++)
                T[(size_t)a * na + q] -= s * T[(size_t)b * na + q];
        }
        double nn = 0.0;
        for (int p = 0; p <= a; p++)
            for (int q = 0; q <= a; q++)
                nn += T[(size_t)a * na + p] * T[(size_t)a * na + q] * ac_radial_moment(&fit, p, q, 0);
        for (int q = 0; q <= a; q++)
            T[(size_t)a * na + q] /= sqrt(nn);
    }
    rc = ac_radial_mix(&gs, 1, (const ac_radial *[]){&fit}, na, ls, T, err, errlen);
    ac_radial_free(&fit);
    free(T);
    free(ls);
    if (rc)
        return -1;
    /* put them in place of the .w orbitals of l */
    int nmo = st->rad.nmo, ntot = nmo + na;
    double *coef = calloc((size_t)nmo * ntot, sizeof *coef);
    for (int j = 0; j < nmo; j++) {
        int a = -1;
        for (int q = 0; q < na; q++)
            if (ia[q] == j)
                a = q;
        coef[(size_t)j * ntot + (a < 0 ? j : nmo + a)] = 1.0;
    }
    ac_radial dst;
    rc = ac_radial_mix(&dst, 2, (const ac_radial *[]){&st->rad, &gs}, nmo, st->rad.l, coef, err,
                       errlen);
    free(coef);
    ac_radial_free(&gs);
    if (rc)
        return -1;
    ac_radial_free(&st->rad);
    st->rad = dst;
    return 0;
}

/* lag=l:N:b[,l:N:b...]: for each l, the N raw Laguerre functions (s: the
   N - 1 combinations with the nuclear cusp R'(0) = -Z R(0)), orthogonalized
   to the .w orbitals of that l, orthonormalized (canonically) and
   diagonalized on the bare hydrogen-like h; appended to the orbitals of w
   with labels continuing its n. */
static int add_laguerre(state_t *st, const char *spec, char *err, int errlen)
{
    ac_atspw *w = &st->w;
    const ac_rgrid *g = st->rad.g;
    double z = w->z;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", spec);
    ac_radial fin[AC_BASIS_MAXL];
    int have[AC_BASIS_MAXL] = {0};
    /* lagfock: the occupations of the screening potential (the .w orbitals;
       read before strtok below is in use) */
    int focc[MAXO] = {0};
    if (lagfock >= 0 && fock_occupation(st, lagfock, focc, err, errlen))
        return -1;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int l, nb;
        double b;
        if (sscanf(tok, "%d:%d:%lf", &l, &nb, &b) != 3 || l < 0 || l >= AC_BASIS_MAXL || nb < 2 ||
            have[l]) {
            snprintf(err, (size_t)errlen, "bad lag= item %s", tok);
            return -1;
        }
        ac_radial raw, proj;
        double *f = calloc((size_t)nb, sizeof *f);
        if (laguerre_raw(&raw, g, z, l, nb, b, f)) {
            snprintf(err, (size_t)errlen, "out of memory");
            return -1;
        }
        /* the cusp subspace (s): e_k - (f_k/f_p) e_p, p the largest |f| */
        int m0 = l == 0 ? nb - 1 : nb, piv = 0;
        for (int k = 1; k < nb; k++)
            if (fabs(f[k]) > fabs(f[piv]))
                piv = k;
        int na = 0, ia[AC_ATSP_MAXORB];
        for (int i = 0; i < w->n; i++)
            if (w->l[i] == l)
                ia[na++] = st->mo[i];
        if (lagfit && na && fit_orbitals(st, &raw, f, piv, l, ia, na, err, errlen)) {
            free(f);
            ac_radial_free(&raw);
            return -1;
        }
        /* projected functions: B c - sum_A A <A|B c>, as coefficients on
           (rad, raw) for ac_radial_mix */
        int ntot = st->rad.nmo + nb;
        double *coef = calloc((size_t)m0 * ntot, sizeof *coef);
        double *sab = malloc((size_t)(na ? na : 1) * nb * sizeof *sab);
        for (int i = 0; i < na; i++)
            for (int k = 0; k < nb; k++)
                sab[(size_t)i * nb + k] = gdot(g, st->rad.val + (size_t)ia[i] * g->n,
                                               raw.val + (size_t)k * g->n);
        int *lsame = malloc((size_t)m0 * sizeof *lsame);
        for (int j = 0, k = 0; k < nb; k++) {
            if (l == 0 && k == piv)
                continue;
            double *c = coef + (size_t)j * ntot + st->rad.nmo;
            c[k] = 1.0;
            if (l == 0)
                c[piv] = -f[k] / f[piv];
            for (int i = 0; i < na; i++) {
                double s = 0.0;
                for (int q = 0; q < nb; q++)
                    s += sab[(size_t)i * nb + q] * c[q];
                coef[(size_t)j * ntot + ia[i]] = -s;
            }
            lsame[j++] = l;
        }
        const ac_radial *src2[2] = {&st->rad, &raw};
        int rc = ac_radial_mix(&proj, 2, src2, m0, lsame, coef, err, errlen);
        free(coef);
        free(sab);
        free(f);
        ac_radial_free(&raw);
        if (rc) {
            free(lsame);
            return -1;
        }
        /* canonical orthonormalization, then the bare h */
        double *S = malloc((size_t)m0 * m0 * sizeof *S), *ev = malloc((size_t)m0 * sizeof *ev);
        for (int i = 0; i < m0; i++)
            for (int j = 0; j < m0; j++)
                S[(size_t)j * m0 + i] = ac_radial_moment(&proj, i, j, 0);
        if (ac_ci_eigvecs(m0, S, ev)) {
            snprintf(err, (size_t)errlen, "lag: dsyev failed");
            return -1;
        }
        int m1 = 0;
        double *X = malloc((size_t)m0 * m0 * sizeof *X);
        printf("lag: l = %d, overlap eigenvalues of the projected functions:", l);
        for (int c = 0; c < m0 && c < 8; c++)
            printf(" %.2e", ev[c]);
        printf(" ... %.2e (kept above %.0e)\n", ev[m0 - 1], lagthr);
        for (int c = 0; c < m0; c++)
            if (ev[c] > lagthr * ev[m0 - 1]) {
                for (int i = 0; i < m0; i++)
                    X[(size_t)m1 * m0 + i] = S[(size_t)c * m0 + i] / sqrt(ev[c]);
                m1++;
            }
        double *H = malloc((size_t)m1 * m1 * sizeof *H), *hp = malloc((size_t)m0 * m0 * sizeof *hp);
        for (int i = 0; i < m0; i++)
            for (int j = 0; j < m0; j++)
                hp[(size_t)i * m0 + j] = ac_radial_h(&proj, i, j);
        if (lagfock >= 0 && fock_add(st, &proj, l, m0, focc, hp)) {
            snprintf(err, (size_t)errlen, "out of memory");
            return -1;
        }
        for (int a = 0; a < m1; a++)
            for (int c = 0; c < m1; c++) {
                double s = 0.0;
                for (int i = 0; i < m0; i++)
                    for (int j = 0; j < m0; j++)
                        s += X[(size_t)a * m0 + i] * hp[(size_t)i * m0 + j] * X[(size_t)c * m0 + j];
                H[(size_t)c * m1 + a] = s;
            }
        if (ac_ci_eigvecs(m1, H, ev)) {
            snprintf(err, (size_t)errlen, "lag: dsyev failed");
            return -1;
        }
        double *T = calloc((size_t)m1 * m0, sizeof *T);
        for (int q = 0; q < m1; q++)
            for (int a = 0; a < m1; a++)
                for (int i = 0; i < m0; i++)
                    T[(size_t)q * m0 + i] += H[(size_t)q * m1 + a] * X[(size_t)a * m0 + i];
        const ac_radial *src1[1] = {&proj};
        rc = ac_radial_mix(&fin[l], 1, src1, m1, lsame, T, err, errlen);
        ac_radial_free(&proj);
        free(S);
        free(ev);
        free(X);
        free(H);
        free(hp);
        free(T);
        free(lsame);
        if (rc)
            return -1;
        have[l] = 1;
        printf("lag: l = %d, %d raw functions (b = %g), %d added after the %d orbitals of the .w\n",
               l, nb, b, m1, na);
    }
    /* the new orbital set: per l, the .w orbitals then the added ones */
    int nsrc = 1, off[AC_BASIS_MAXL + 1], ntot = st->rad.nmo, nadd = 0;
    const ac_radial *src[AC_BASIS_MAXL + 1] = {&st->rad};
    for (int l = 0; l < AC_BASIS_MAXL; l++)
        if (have[l]) {
            off[l] = ntot;
            ntot += fin[l].nmo;
            nadd += fin[l].nmo;
            src[nsrc++] = &fin[l];
        }
    if (w->n + nadd > MAXO) {
        snprintf(err, (size_t)errlen, "lag: %d orbitals, at most %d", w->n + nadd, MAXO);
        return -1;
    }
    int nmo = st->rad.nmo + nadd, *lmo = malloc((size_t)nmo * sizeof *lmo);
    double *coef = calloc((size_t)nmo * ntot, sizeof *coef);
    int newmo[AC_ATSP_MAXORB], j = 0, nw = w->n;
    for (int l = 0; l < AC_BASIS_MAXL; l++) {
        int nmax = l;
        for (int i = 0; i < nw; i++)
            if (w->l[i] == l) {
                coef[(size_t)j * ntot + st->mo[i]] = 1.0;
                newmo[i] = j;
                lmo[j++] = l;
                if (atoi(w->el[i]) > nmax)
                    nmax = atoi(w->el[i]);
            }
        if (!have[l])
            continue;
        for (int q = 0; q < fin[l].nmo; q++) {
            coef[(size_t)j * ntot + off[l] + q] = 1.0;
            int i = w->n++, nn = nmax + 1 + q;
            if (nn > 99) {
                free(coef);
                free(lmo);
                snprintf(err, (size_t)errlen, "lag: label n = %d above 99", nn);
                return -1;
            }
            w->el[i][0] = nn > 9 ? (char)('0' + nn / 10) : (char)('0' + nn);
            w->el[i][1] = nn > 9 ? (char)('0' + nn % 10) : "spdfghik"[l];
            w->el[i][2] = nn > 9 ? "spdfghik"[l] : 0;
            w->el[i][3] = 0;
            w->l[i] = l;
            w->m[i] = 0;
            w->p[i] = NULL;
            newmo[i] = j;
            lmo[j++] = l;
        }
    }
    ac_radial dst;
    int rc = ac_radial_mix(&dst, nsrc, src, nmo, lmo, coef, err, errlen);
    free(coef);
    free(lmo);
    for (int l = 0; l < AC_BASIS_MAXL; l++)
        if (have[l])
            ac_radial_free(&fin[l]);
    if (rc)
        return -1;
    ac_radial_free(&st->rad);
    st->rad = dst;
    for (int i = 0; i < w->n; i++)
        st->mo[i] = newmo[i];
    return 0;
}

/* a CSF has only the occupied orbitals of its configuration as slots
   (ac_ci_from_orbitals_occ); allslots: every orbital (ac_ci_from_orbitals) */
static int occslots = 1;

/* the CSFs: a configuration list (every CSF of each configuration, slots
   in the order of w) or an ATSP .c list */
static int load_csfs(state_t *st, const ac_cfptab *t, char *err, int errlen)
{
    const ac_atspw *w = &st->w;
    if (st->csfmode) {
        if (read_atsp_c(st->cpath, w, t, &st->ci, st->slot_w, err, errlen))
            return -1;
    } else {
        int ncfg = 0, cap = 0, *occ = NULL;
        FILE *cf = fopen(st->cpath, "r");
        if (!cf) {
            snprintf(err, (size_t)errlen, "%s: %s", st->cpath, strerror(errno));
            return -1;
        }
        char line[1024];
        while (fgets(line, sizeof line, cf)) {
            char *hash = strchr(line, '#');
            if (hash)
                *hash = 0;
            int any = 0;
            if (ncfg == cap) {
                cap = cap ? 2 * cap : 256;
                int *no = realloc(occ, (size_t)cap * w->n * sizeof *occ);
                if (!no) {
                    free(occ);
                    fclose(cf);
                    snprintf(err, (size_t)errlen, "out of memory");
                    return -1;
                }
                occ = no;
            }
            for (int s = 0; s < w->n; s++)
                occ[ncfg * w->n + s] = 0;
            for (char *tok = strtok(line, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
                char lab[8];
                int n;
                if (sscanf(tok, "%7[^(](%d)", lab, &n) != 2) {
                    snprintf(err, (size_t)errlen, "bad token %s", tok);
                    free(occ);
                    fclose(cf);
                    return -1;
                }
                int s = w_index(w, lab);
                if (s < 0) {
                    snprintf(err, (size_t)errlen, "orbital %s not in %s", lab, st->wpath);
                    free(occ);
                    fclose(cf);
                    return -1;
                }
                occ[ncfg * w->n + s] = n;
                any = 1;
            }
            if (any)
                ncfg++;
        }
        fclose(cf);
        st->ncfg = ncfg;
        int lorb[MAXO];
        for (int s = 0; s < w->n; s++)
            lorb[s] = w->l[s];
        int rc = (occslots ? ac_ci_from_orbitals_occ : ac_ci_from_orbitals)(&st->ci, t, w->n, lorb,
                                                                           ncfg, occ, err, errlen);
        free(occ);
        if (rc)
            return -1;
        for (int s = 0; s < w->n; s++)
            st->slot_w[s] = s;
    }
    for (int s = 0; s < st->ci.nwt; s++)
        st->slotmo[s] = st->mo[st->slot_w[s]];
    return 0;
}

/* Gram-Schmidt of the MOs of each l, in MO order (= the order of w) */
static int orthonormalize(state_t *st, char *err, int errlen)
{
    int n = st->rad.nmo;
    double *coef = calloc((size_t)n * n, sizeof *coef), *S = malloc((size_t)n * n * sizeof *S);
    if (!coef || !S) {
        free(coef);
        free(S);
        snprintf(err, (size_t)errlen, "out of memory");
        return -1;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            S[i * n + j] = st->rad.l[i] == st->rad.l[j] ? ac_radial_moment(&st->rad, i, j, 0) : 0.0;
    for (int j = 0; j < n; j++) {
        double *v = coef + (size_t)j * n;
        v[j] = 1.0;
        for (int k = 0; k < j; k++) {
            if (st->rad.l[k] != st->rad.l[j])
                continue;
            const double *u = coef + (size_t)k * n;
            double p = 0.0;
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++)
                    p += u[a] * S[a * n + b] * v[b];
            for (int a = 0; a < n; a++)
                v[a] -= p * u[a];
        }
        double q = 0.0;
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++)
                q += v[a] * S[a * n + b] * v[b];
        for (int a = 0; a < n; a++)
            v[a] /= sqrt(q);
    }
    ac_radial r;
    const ac_radial *src[1] = {&st->rad};
    int rc = ac_radial_mix(&r, 1, src, n, st->rad.l, coef, err, errlen);
    free(coef);
    free(S);
    if (rc)
        return -1;
    ac_radial_free(&st->rad);
    st->rad = r;
    return 0;
}

/* rot=a,b,deg: P_a <- cos P_a + sin P_b, P_b <- -sin P_a + cos P_b */
static int rotate(state_t *st, const char *spec, char *err, int errlen)
{
    char buf[64], *f[3];
    int nf = 0;
    snprintf(buf, sizeof buf, "%s", spec);
    for (char *tok = strtok(buf, ","); tok && nf < 3; tok = strtok(NULL, ","))
        f[nf++] = tok;
    int ia = nf == 3 ? w_index(&st->w, f[0]) : -1, ib = nf == 3 ? w_index(&st->w, f[1]) : -1;
    if (ia < 0 || ib < 0 || st->w.l[ia] != st->w.l[ib]) {
        snprintf(err, (size_t)errlen, "bad rotation %s", spec);
        return -1;
    }
    double th = atof(f[2]) * 3.14159265358979323846 / 180.0, c = cos(th), s = sin(th);
    int n = st->rad.nmo, a = st->mo[ia], b = st->mo[ib];
    double *coef = calloc((size_t)n * n, sizeof *coef);
    if (!coef) {
        snprintf(err, (size_t)errlen, "out of memory");
        return -1;
    }
    for (int i = 0; i < n; i++)
        coef[(size_t)i * n + i] = 1.0;
    coef[(size_t)a * n + a] = c;
    coef[(size_t)a * n + b] = s;
    coef[(size_t)b * n + a] = -s;
    coef[(size_t)b * n + b] = c;
    ac_radial r;
    const ac_radial *src[1] = {&st->rad};
    int rc = ac_radial_mix(&r, 1, src, n, st->rad.l, coef, err, errlen);
    free(coef);
    if (rc)
        return -1;
    ac_radial_free(&st->rad);
    st->rad = r;
    return 0;
}

typedef struct {
    int nlev, noso, mvd, ls, so2, rkrel, ss, oo, gc2e, oz2e, rec2, m1r2e, hfsrel;
} opts_t;

/* the CI of one state: its levels, kept in *keep when keep is not NULL */
static int run_state(state_t *st, const opts_t *o, ac_lsj_kept **keep, int *nkeep, char *err,
                     int errlen)
{
    reset_caches();
    nsl = st->ci.nwt;
    st->x = (ctx_t){&st->rad, st->slotmo, o->mvd, o->rkrel, NULL};
    ac_ci *ci = &st->ci;
    ci->in.h = get_h;
    ci->in.rk = get_rk;
    ci->in.ctx = &st->x;
    ci->in.zctx = &st->x;
    if (!o->noso)
        ci->in.zeta = get_zeta;
    if (o->so2)
        ci->in.rkg = get_rkg;
    if (o->oz2e || o->m1r2e)
        ci->in.roz = get_roz;
    if (o->m1r2e)
        ci->in.rm1 = get_rm1;
    if (o->rec2)
        ci->in.rrc = get_rrc;
    if (o->gc2e)
        ci->in.rgc = get_rgc;
    if (o->oo)
        ci->in.roo = get_roo;
    if (o->ss)
        ci->in.rss = get_rss;
    const ac_atspw *w = &st->w;
    double ttab = omp_get_wtime();
    printf("acatsp: %s, Z = %g, %d orbitals, %d configurations, %d CSFs, 1/alpha = %.9f,"
           " H_SO1 %s, H_SO2 %s, H_SS %s, mass velocity + Darwin %s\n",
           st->wpath, w->z, w->n, st->ncfg, ci->ncsf, 1.0 / ac_alpha, o->noso ? "off" : "on",
           o->so2 ? "on" : "off", o->ss ? "on" : "off", o->mvd ? "on" : "off");
    for (int s = 0; s < w->n; s++) {
        int i = st->mo[s];
        const ac_radial *rad = &st->rad;
        printf("orbital %-3s l=%d  <h> %.12f  <r^-3> %.10f  zeta %.12e hartree = %.6f cm-1"
               "  MV %.10e  D %.10e\n",
               w->el[s], w->l[s], ac_radial_h(rad, i, i),
               w->l[s] ? ac_radial_moment(rad, i, i, -3) : 0.0,
               w->l[s] ? ac_radial_zeta(rad, i, i) : 0.0,
               w->l[s] ? ac_radial_zeta(rad, i, i) * AC_HARTREE_CM : 0.0,
               ac_radial_mv(rad, i, i), ac_radial_darwin(rad, i, i));
    }
    since("orbital table", ttab);
    if (o->ls)
        return print_ls(ci, o->nlev) ? (snprintf(err, (size_t)errlen, "ls failed"), -1) : 0;
    if (o->hfsrel) {
        int nres;
        ci->in.zeta = NULL;
        ci->in.roo = get_roo;
        return ac_hfsrel_print(stdout, ci, &st->rad, st->slotmo, NULL, 0, &nres, err, errlen);
    }
    return ac_lsj_print_levels_keep(stdout, ci, o->nlev, get_t, get_u, &st->x, keep, nkeep, err,
                                    errlen);
}

/* ---- two states on biorthonormal orbitals (S2-M9) ------------------------ */

/* The common slot order of two states: a topological order of the slots of
   both that keeps the order of the shells within every CSF (ties: state 1's
   order, then state 2's); the CSFs of both are renumbered to it. */
static int merge_slots(state_t *s1, state_t *s2, char lab[][4], int *lorb, int *ng, char *err,
                       int errlen)
{
    state_t *st[2] = {s1, s2};
    int n = 0, gi[2][AC_CSF_MAXSH];
    static unsigned char prec[2 * AC_CSF_MAXSH][2 * AC_CSF_MAXSH];
    char cand[2 * AC_CSF_MAXSH][4];
    int lc[2 * AC_CSF_MAXSH];
    memset(prec, 0, sizeof prec);
    for (int k = 0; k < 2; k++)
        for (int s = 0; s < st[k]->ci.nwt; s++) {
            const char *el = st[k]->w.el[st[k]->slot_w[s]];
            int c;
            for (c = 0; c < n && strcmp(cand[c], el); c++)
                ;
            if (c == n) {
                snprintf(cand[n], 4, "%s", el);
                lc[n++] = st[k]->w.l[st[k]->slot_w[s]];
            } else if (lc[c] != st[k]->w.l[st[k]->slot_w[s]]) {
                snprintf(err, (size_t)errlen, "orbital %s has different l in the two states", el);
                return -1;
            }
            gi[k][s] = c;
        }
    /* the constraints: consecutive shells within each CSF (the order of the
       slots of a state that never meet in one CSF is free; ATSP lists of
       the two terms may order them differently) */
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < st[k]->ci.ncsf; i++) {
            const ac_ci_csf *f = &st[k]->ci.csf[i];
            for (int j = 2; j <= f->c.nsh; j++)
                prec[gi[k][f->seq[j - 1] - 1]][gi[k][f->seq[j] - 1]] = 1;
        }
    int pos[2 * AC_CSF_MAXSH], done[2 * AC_CSF_MAXSH] = {0};
    for (int g = 0; g < n; g++) {
        int pick = -1;
        for (int c = 0; c < n && pick < 0; c++) {
            if (done[c])
                continue;
            int ok = 1;
            for (int d = 0; d < n && ok; d++)
                ok = done[d] || !prec[d][c];
            if (ok)
                pick = c;
        }
        if (pick < 0) {
            snprintf(err, (size_t)errlen, "the two states order their orbitals inconsistently");
            return -1;
        }
        done[pick] = 1;
        pos[pick] = g;
        snprintf(lab[g], 4, "%s", cand[pick]);
        lorb[g] = lc[pick];
    }
    if (n > AC_CSF_MAXSH) {
        snprintf(err, (size_t)errlen, "%d orbitals in the two states, at most %d", n,
                 AC_CSF_MAXSH);
        return -1;
    }
    for (int k = 0; k < 2; k++) {
        ac_ci *ci = &st[k]->ci;
        int map[AC_CSF_MAXSH + 1], smo[AC_CSF_MAXSH], sw[AC_CSF_MAXSH];
        for (int s = 0; s < ci->nwt; s++)
            map[s + 1] = pos[gi[k][s]] + 1;
        for (int i = 0; i < ci->ncsf; i++)
            for (int j = 1; j <= ci->csf[i].c.nsh; j++)
                ci->csf[i].seq[j] = map[ci->csf[i].seq[j]];
        for (int g = 0; g < n; g++)
            smo[g] = sw[g] = -1;
        for (int s = 0; s < ci->nwt; s++) {
            smo[map[s + 1] - 1] = st[k]->slotmo[s];
            sw[map[s + 1] - 1] = st[k]->slot_w[s];
        }
        ci->nwt = n;
        for (int g = 0; g < n; g++) {
            ci->lorb[g + 1] = lorb[g];
            st[k]->slotmo[g] = smo[g];
            st[k]->slot_w[g] = sw[g];
        }
    }
    *ng = n;
    return 0;
}

static double overlap(const ac_radial *ra, int a, const ac_radial *rb, int b)
{
    const ac_rgrid *g = ra->g;
    const double *va = ra->val + (size_t)a * g->n, *vb = rb->val + (size_t)b * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * va[i] * vb[i] * pow(g->r[i], 2);
    return s;
}

/* rewrite the kept levels of a state on its new orbitals; *leak receives
   the largest weight that left a level's (S, L) blocks (0 in exact
   arithmetic) */
static int transform_levels(const ac_biorth_ops *o, const ac_ci *ci, const double *t,
                            ac_lsj_kept *k, int nk, double *leak, char *err, int errlen)
{
    double *x = malloc((size_t)ci->ncsf * sizeof *x);
    char *in = malloc((size_t)ci->ncsf);
    if (!x || !in) {
        free(x);
        free(in);
        snprintf(err, (size_t)errlen, "out of memory");
        return -1;
    }
    for (int q = 0; q < nk; q++) {
        ac_lsj_level *v = &k[q].v;
        memset(x, 0, (size_t)ci->ncsf * sizeof *x);
        memset(in, 0, (size_t)ci->ncsf);
        for (int p = 0; p < v->dim; p++) {
            x[v->idx[p]] = v->c[p];
            in[v->idx[p]] = 1;
        }
        if (ac_biorth_transform(o, ci, t, 1, x, err, errlen)) {
            free(x);
            free(in);
            return -1;
        }
        double *c = (double *)v->c;
        for (int p = 0; p < v->dim; p++)
            c[p] = x[v->idx[p]];
        for (int i = 0; i < ci->ncsf; i++)
            if (!in[i])
                *leak = fmax(*leak, x[i] * x[i]);
    }
    free(x);
    free(in);
    return 0;
}

static int bio_cross(state_t *s1, state_t *s2, ac_lsj_kept *k1, int n1, ac_lsj_kept *k2,
                     int n2, int mode, int force, const opts_t *o, char *err, int errlen)
{
    char lab[AC_CSF_MAXSH][4];
    int lorb[AC_CSF_MAXSH], ng;
    if (merge_slots(s1, s2, lab, lorb, &ng, err, errlen))
        return -1;
    int w = ng + 1, rc = -1;
    double *t1 = calloc((size_t)w * w, sizeof *t1), *t2 = calloc((size_t)w * w, sizeof *t2);
    double *coef = NULL;
    int nmo = 0, lmo[2 * AC_CSF_MAXSH], mol[AC_CSF_MAXSH], mor[AC_CSF_MAXSH];
    ac_radial rab = {0};
    ac_biorth_ops o1 = {0}, o2 = {0};
    ac_ci cc = {0};
    int ntot = s1->rad.nmo + s2->rad.nmo;
    coef = calloc((size_t)2 * ng * ntot + 1, sizeof *coef);
    if (!t1 || !t2 || !coef) {
        snprintf(err, (size_t)errlen, "out of memory");
        goto out;
    }
    for (int g = 0; g < ng; g++) {
        t1[g + 1 + (g + 1) * w] = t2[(g + 1) * w + g + 1] = 1.0;
        mol[g] = mor[g] = 0;
    }
    printf("biorthonormal orbitals (S2-M9), %d orbitals in the common order:", ng);
    for (int g = 0; g < ng; g++)
        printf(" %s", lab[g]);
    printf("\n");
    for (int l = 0; l < AC_BASIS_MAXL; l++) {
        int gx[AC_CSF_MAXSH], gy[AC_CSF_MAXSH], nx = 0, ny = 0;
        for (int g = 0; g < ng; g++) {
            if (lorb[g] != l)
                continue;
            if (s1->slotmo[g] >= 0)
                gx[nx++] = g;
            if (s2->slotmo[g] >= 0)
                gy[ny++] = g;
        }
        if (!nx && !ny)
            continue;
        int nmin = nx < ny ? nx : ny;
        for (int i = 0; i < nmin; i++)
            if (gx[i] != gy[i]) {
                snprintf(err, (size_t)errlen,
                         "l = %d: the orbitals of the two states differ before the last ones", l);
                goto out;
            }
        double S[AC_CSF_MAXSH * AC_CSF_MAXSH], cx[AC_CSF_MAXSH * AC_CSF_MAXSH],
            cy[AC_CSF_MAXSH * AC_CSF_MAXSH], tx[AC_CSF_MAXSH * AC_CSF_MAXSH],
            ty[AC_CSF_MAXSH * AC_CSF_MAXSH];
        for (int p = 0; p < nx; p++)
            for (int q = 0; q < ny; q++)
                S[p * ny + q] = overlap(&s1->rad, s1->slotmo[gx[p]], &s2->rad, s2->slotmo[gy[q]]);
        if (ac_biorth_lu(nx, ny, S, mode, 1e-6, cx, cy, tx, ty, err, errlen))
            goto out;
        printf("  l = %d: %d and %d orbitals; <X_i|Y_i>:", l, nx, ny);
        for (int i = 0; i < nmin; i++)
            printf(" %s %.10f", lab[gx[i]], S[i * ny + i]);
        printf("\n");
        for (int p = 0; p < nx; p++)
            for (int q = 0; q < nx; q++)
                t1[(gx[p] + 1) * w + gx[q] + 1] = tx[p * nx + q];
        for (int p = 0; p < ny; p++)
            for (int q = 0; q < ny; q++)
                t2[(gy[p] + 1) * w + gy[q] + 1] = ty[p * ny + q];
        /* A = X cx, then B = Y cy, as MOs of rab */
        for (int j = 0; j < nx; j++, nmo++) {
            for (int p = 0; p < nx; p++)
                coef[(size_t)nmo * ntot + s1->slotmo[gx[p]]] = cx[p * nx + j];
            lmo[nmo] = l;
            mol[gx[j]] = nmo;
        }
        for (int j = 0; j < ny; j++, nmo++) {
            for (int p = 0; p < ny; p++)
                coef[(size_t)nmo * ntot + s1->rad.nmo + s2->slotmo[gy[p]]] = cy[p * ny + j];
            lmo[nmo] = l;
            mor[gy[j]] = nmo;
        }
    }
    const ac_radial *src[2] = {&s1->rad, &s2->rad};
    lap("bio: start");
    if (ac_radial_mix(&rab, 2, src, nmo, lmo, coef, err, errlen))
        goto out;
    lap("bio: biorthonormal orbitals");
    /* the CI vectors on A and B */
    double dev1, dev2, leak = 0.0;
    char what[160];
    if (ac_biorth_closure(&s1->ci, &dev1, what, sizeof what, err, errlen))
        goto out;
    if (dev1 > 1e-10)
        printf("  state 1 not closed under deexcitation: %s\n", what);
    if (ac_biorth_closure(&s2->ci, &dev2, what, sizeof what, err, errlen))
        goto out;
    if (dev2 > 1e-10)
        printf("  state 2 not closed under deexcitation: %s\n", what);
    printf("  closure under deexcitation: largest outside weight %.3e (state 1), %.3e (state 2)\n",
           dev1, dev2);
    lap("bio: closure");
    if ((dev1 > 1e-10 || dev2 > 1e-10) && !force) {
        snprintf(err, (size_t)errlen,
                 "CSF list not closed under deexcitation (biorthforce to go on anyway)");
        goto out;
    }
    if (ac_biorth_ops_build(&o1, &s1->ci, err, errlen) ||
        ac_biorth_ops_build(&o2, &s2->ci, err, errlen) ||
        transform_levels(&o1, &s1->ci, t1, k1, n1, &leak, err, errlen) ||
        transform_levels(&o2, &s2->ci, t2, k2, n2, &leak, err, errlen))
        goto out;
    lap("bio: counter-transformation");
    printf("  counter-transformation: %d and %d elements of E_ak; weight leaving a level's"
           " (S, L) %.1e\n",
           o1.n, o2.n, leak);
    /* one CSF list for both, state 2 after state 1 */
    cc = s1->ci;
    cc.thc = NULL;
    memset(&cc.in, 0, sizeof cc.in);
    cc.ncsf = s1->ci.ncsf + s2->ci.ncsf;
    cc.csf = malloc((size_t)cc.ncsf * sizeof *cc.csf);
    if (!cc.csf) {
        snprintf(err, (size_t)errlen, "out of memory");
        goto out;
    }
    memcpy(cc.csf, s1->ci.csf, (size_t)s1->ci.ncsf * sizeof *cc.csf);
    memcpy(cc.csf + s1->ci.ncsf, s2->ci.csf, (size_t)s2->ci.ncsf * sizeof *cc.csf);
    for (int q = 0; q < n2; q++) {
        int *ix = (int *)k2[q].v.idx;
        for (int p = 0; p < k2[q].v.dim; p++)
            ix[p] += s1->ci.ncsf;
    }
    reset_caches();
    nsl = ng;
    ctx_t xc = {&rab, mol, 0, 0, mor};
    /* the two-electron terms of M1 and of the Zeeman elements, bra orbitals A
       and ket orbitals B */
    if (o->m1r2e || o->oz2e)
        cc.in.roz = get_roz;
    if (o->m1r2e)
        cc.in.rm1 = get_rm1;
    if (o->gc2e)
        cc.in.rgc = get_rgc;
    if (o->rec2)
        cc.in.rrc = get_rrc;
    cc.in.zctx = &xc;
    printf("Between the levels of state 1 (A, bra) and state 2 (B, ket):\n");
    if (ac_lsj_print_cross(stdout, &cc, k1, n1, k2, n2, get_t, get_u, &xc, err, errlen))
        goto out;
    lap("bio: matrix elements between the states");
    rc = 0;
out:
    free(t1);
    free(t2);
    free(coef);
    free(cc.csf);
    ac_ci_free_cache(&cc);
    ac_biorth_ops_free(&o1);
    ac_biorth_ops_free(&o2);
    ac_radial_free(&rab);
    reset_caches();
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s CFPTBL atom.w configs [nlev] [noso] [mvd] [alpha=INV]\n",
                argv[0]);
        return 2;
    }
    opts_t o = {4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int nq = 0, biomode = AC_BIORTH_SYM, bioforce = 0, orth = 0;
    const char *writec = NULL, *bio = NULL, *rot[16], *lag = NULL;
    int nrot = 0;
    const char *qs[64];
    for (int i = 4; i < argc; i++) {
        if (!strncmp(argv[i], "q=", 2) && nq < 64)
            qs[nq++] = argv[i] + 2;
        else if (!strcmp(argv[i], "ls"))
            o.ls = 1;
        else if (!strncmp(argv[i], "delta12=", 8))
            ac_hfsrel_delta = atof(argv[i] + 8);
        else if (!strcmp(argv[i], "hfsrel"))
            o.hfsrel = 1;
        else if (!strncmp(argv[i], "lag=", 4))
            lag = argv[i] + 4;
        else if (!strncmp(argv[i], "lagthr=", 7))
            lagthr = atof(argv[i] + 7);
        else if (!strcmp(argv[i], "lagfit"))
            lagfit = 1;
        else if (!strcmp(argv[i], "lagfock"))
            lagfock = 1;
        else if (!strncmp(argv[i], "lagfock=", 8))
            lagfock = atoi(argv[i] + 8);
        else if (!strcmp(argv[i], "occslots"))
            occslots = 1;
        else if (!strcmp(argv[i], "allslots"))
            occslots = 0;
        else if (!strncmp(argv[i], "writec=", 7))
            writec = argv[i] + 7;
        else if (!strncmp(argv[i], "bio=", 4))
            bio = argv[i] + 4;
        else if (!strncmp(argv[i], "biomode=", 8))
            biomode = atoi(argv[i] + 8);
        else if (!strcmp(argv[i], "biorthforce"))
            bioforce = 1;
        else if (!strcmp(argv[i], "digits"))
            ac_lsj_digits = 1;
        else if (!strncmp(argv[i], "dav=", 4))
            ac_lsj_davidson = atoi(argv[i] + 4);
        else if (!strcmp(argv[i], "orth"))
            orth = 1;
        else if ((!strncmp(argv[i], "rot1=", 5) || !strncmp(argv[i], "rot2=", 5)) && nrot < 16)
            rot[nrot++] = argv[i]; /* applied in the order given */
        else if (!strcmp(argv[i], "noso"))
            o.noso = 1;
        else if (!strcmp(argv[i], "so2"))
            o.so2 = 1;
        else if (!strcmp(argv[i], "rkrel"))
            o.rkrel = 1;
        else if (!strcmp(argv[i], "oz2e"))
            o.oz2e = 1;
        else if (!strcmp(argv[i], "m1r2e"))
            o.m1r2e = 1;
        else if (!strncmp(argv[i], "m1r2epart=", 10))
            ac_lsj_m1_2e_parts = atoi(argv[i] + 10);
        else if (!strcmp(argv[i], "rec2"))
            o.rec2 = 1;
        else if (!strcmp(argv[i], "zoff"))
            ac_lsj_zoff = 1;
        else if (!strcmp(argv[i], "hfs")) {
            ac_lsj_hfs_rm3 = get_r3;
            ac_lsj_hfs_r00 = get_r00;
        }
        else if (!strcmp(argv[i], "trans"))
            ac_lsj_trans_rk = get_trk;
        else if (!strcmp(argv[i], "h6"))
            ac_lsj_h6_rk = get_trk;
        else if (!strcmp(argv[i], "is"))
            ac_lsj_is_rk = get_trk;
        else if (!strcmp(argv[i], "gc2e"))
            o.gc2e = 1;
        else if (!strncmp(argv[i], "gc2epart=", 9))
            ac_gc2e_parts = atoi(argv[i] + 9);
        else if (!strcmp(argv[i], "oo"))
            o.oo = 1;
        else if (!strcmp(argv[i], "ss"))
            o.ss = 1;
        else if (!strncmp(argv[i], "so2part=", 8))
            ac_so2_parts = atoi(argv[i] + 8);
        else if (!strcmp(argv[i], "atspcompat"))
            ac_atsp_compat = 1;
        else if (!strcmp(argv[i], "mvd"))
            o.mvd = 1;
        else if (!strncmp(argv[i], "alpha=", 6))
            ac_alpha = 1.0 / atof(argv[i] + 6);
        else if (i == 4)
            o.nlev = atoi(argv[i]);
        else {
            fprintf(stderr, "acatsp: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    char err[256];
    ac_cfptab t;
    double tset = omp_get_wtime();
    if (ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "acatsp: %s\n", err);
        return 1;
    }
    tset = since("setup: CFP table", tset);
    static state_t s1, s2;
    s1.wpath = argv[2];
    s1.cpath = argv[3];
    if (bio) {
        static char b2[1024];
        snprintf(b2, sizeof b2, "%s", bio);
        char *comma = strchr(b2, ',');
        if (!comma) {
            fprintf(stderr, "acatsp: bio=atom2.w,configs2\n");
            return 2;
        }
        *comma = 0;
        s2.wpath = b2;
        s2.cpath = comma + 1;
        if (o.ls) {
            fprintf(stderr, "acatsp: bio does not go with ls\n");
            return 2;
        }
    }
    ac_rgrid g;
    if (ac_rgrid_init(&g, AC_RGRID_RMIN, AC_RGRID_RMAX, AC_RGRID_H)) {
        fprintf(stderr, "acatsp: radial grid\n");
        return 1;
    }
    if (load_w(&s1, &g, err, sizeof err) || (bio && load_w(&s2, &g, err, sizeof err))) {
        fprintf(stderr, "acatsp: %s\n", err);
        return 1;
    }
    if (orth && (orthonormalize(&s1, err, sizeof err) ||
                 (bio && orthonormalize(&s2, err, sizeof err)))) {
        fprintf(stderr, "acatsp: %s\n", err);
        return 1;
    }
    for (int r = 0; r < nrot; r++)
        if ((rot[r][3] == '1' || bio) &&
            rotate(rot[r][3] == '1' ? &s1 : &s2, rot[r] + 5, err, sizeof err)) {
            fprintf(stderr, "acatsp: %s\n", err);
            return 1;
        }
    if (lag && (bio || add_laguerre(&s1, lag, err, sizeof err))) {
        fprintf(stderr, "acatsp: %s\n", bio ? "lag does not go with bio" : err);
        return 1;
    }
    tset = since("setup: orbitals", tset);
    const ac_atspw *w = &s1.w;
    const ac_radial *radp = &s1.rad;
    const int *mo = s1.mo;
    /* queries, in ATSP's notation (hf, "Additional parameters"):
         r,a,k        <a|r^k|a>; r,a,b,k: <a|r^k|b>
         F,k,a,b      F^k(a,b) = R^k(ab,ab)
         G,k,a,b      G^k(a,b) = R^k(ab,ba)
         R,k,a,b,c,d  R^k(ab,cd), a and c on electron 1
         D,a,b        int R_a (R_b' + (l_b+1) R_b/r) r^2 dr */
    for (int iq = 0; iq < nq; iq++) {
        char buf[128], *f[6];
        int nf = 0;
        snprintf(buf, sizeof buf, "%s", qs[iq]);
        for (char *tok = strtok(buf, ","); tok && nf < 6; tok = strtok(NULL, ","))
            f[nf++] = tok;
        if (nf < 3) {
            fprintf(stderr, "acatsp: bad query %s\n", qs[iq]);
            return 1;
        }
        int oo[4], no = 0;
        for (int j = (f[0][0] == 'r' || f[0][0] == 'D' ? 1 : 2); j < nf && no < 4; j++) {
            if (f[0][0] == 'r' && j == nf - 1)
                break;
            int s = w_index(w, f[j]);
            if (s < 0) {
                fprintf(stderr, "acatsp: query %s: unknown orbital %s\n", qs[iq], f[j]);
                return 1;
            }
            oo[no++] = mo[s];
        }
        double v;
        if (f[0][0] == 'r' && nf == 3)
            v = ac_radial_moment(radp, oo[0], oo[0], atoi(f[2]));
        else if (f[0][0] == 'r' && nf == 4)
            v = ac_radial_moment(radp, oo[0], oo[1], atoi(f[3]));
        else if (f[0][0] == 'F' && nf == 4)
            v = ac_radial_rk(radp, atoi(f[1]), oo[0], oo[0], oo[1], oo[1]);
        else if (f[0][0] == 'G' && nf == 4)
            v = ac_radial_rk(radp, atoi(f[1]), oo[0], oo[1], oo[1], oo[0]);
        else if (f[0][0] == 'R' && nf == 6)
            v = ac_radial_rk(radp, atoi(f[1]), oo[0], oo[2], oo[1], oo[3]);
        else if (f[0][0] == 'N' && nf == 6)
            v = ac_radial_nk(radp, atoi(f[1]), oo[0], oo[1], oo[2], oo[3]);
        else if (f[0][0] == 'V' && nf == 6)
            v = ac_radial_vk(radp, atoi(f[1]), oo[0], oo[1], oo[2], oo[3]);
        else if (f[0][0] == 'D' && nf == 3)
            v = ac_radial_rrc(radp, 2, oo[0], oo[1]);
        else {
            fprintf(stderr, "acatsp: bad query %s\n", qs[iq]);
            return 1;
        }
        printf("query %-20s %.10e\n", qs[iq], v);
    }

    tset = omp_get_wtime();
    if (load_csfs(&s1, &t, err, sizeof err) || (bio && load_csfs(&s2, &t, err, sizeof err))) {
        fprintf(stderr, "acatsp: %s\n", err);
        return 1;
    }
    since("setup: CSF list", tset);
    if (writec && !s1.csfmode) {
        int ident[MAXO];
        for (int s = 0; s < w->n; s++)
            ident[s] = s;
        if (write_atsp_c(&s1.ci, w, ident, writec)) {
            fprintf(stderr, "acatsp: cannot write %s_*.c\n", writec);
            return 1;
        }
    }
    ac_lsj_kept *k1 = NULL, *k2 = NULL;
    int n1 = 0, n2 = 0;
    lap("start");
    if (run_state(&s1, &o, bio ? &k1 : NULL, &n1, err, sizeof err) || (lap("state 1"), 0) ||
        (bio && run_state(&s2, &o, &k2, &n2, err, sizeof err)) || (bio && (lap("state 2"), 0)) ||
        (bio && bio_cross(&s1, &s2, k1, n1, k2, n2, biomode, bioforce, &o, err, sizeof err))) {
        fprintf(stderr, "acatsp: %s\n", err);
        return 1;
    }
    reset_caches();
    ac_lsj_kept_free(k1, n1);
    ac_lsj_kept_free(k2, n2);
    for (int k = 0; k < (bio ? 2 : 1); k++) {
        state_t *st = k ? &s2 : &s1;
        ac_ci_free(&st->ci);
        ac_radial_free(&st->rad);
        ac_atspw_free(&st->w);
    }
    ac_rgrid_free(&g);
    ac_cfp_free(&t);
    return 0;
}
