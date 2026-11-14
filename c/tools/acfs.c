/*
 * acfs -- fine structure and Zeeman factors of a set of configurations:
 * the Coulomb CI plus the one-electron spin-orbit H_SO1 in the LSJ basis
 * (stage 2, S2-M4).
 *
 *   acfs CFPTBL nfte nftri nfaii nftrc configs [nlev]
 *
 * nfte, nftri and configs are as for acterms (orbital list, closed shells,
 * addressing; integrals; "seq*occ ..." per line). nfaii and nftrc give the
 * basis and the MO coefficients, from which ac_radial computes
 *   zeta(a,b) = (alpha^2/2) Z <a|r^-3|b>, T(a,b) and Z <a|1/r|b>.
 * The MO of sequence orbital s is MO nwsq(s) of its l (the NFTRI order,
 * checked in S2-M3).
 *
 * Output, for each 2J, the lowest nlev levels (default 4): energy, the
 * leading LS term and its weight, g_J = g_L + g_s g_S, Delta g_RMC and
 * Delta g_GC (ac_lsj_gfactor); then all printed levels relative to the
 * lowest, in cm-1. Options after nlev: "noso" switches H_SO1 off; "mvd"
 * adds the mass-velocity and Darwin terms to h (variationally); "so2" adds
 * the two-electron spin-orbit H_SO2 (ac_ci_so2); "contact" the two-electron
 * Darwin + spin-spin contact terms (ac_radial_contact, added to R^k); "oo"
 * the orbit-orbit H_OO (ac_ci_oo); "ss" the spin-spin H_SS (ac_ci_ss,
 * tensor part); "gc2e" prints Delta g of the
 * two-electron gauge correction (ac_ci_gc2e; "gc2epart=1" spin-same-orbit,
 * "=2" spin-other-orbit only); "oz2e" that of the orbit-orbit Zeeman term
 * (ac_ci_oz2e); "rec2" the two-body recoil term <sum_{i!=j} (r_i x p_j)_z>/M_J
 * (ac_ci_rec2); "zoff" the Zeeman reduced matrix elements <A||mu||B>
 * between printed levels (ac_lsj_zeeman_reduced); "hfs" the hyperfine
 * <A||T^(K)||B> (ac_lsj_hfs_reduced) and "trans" the E1, M1, E2, M2, E3, M3 lines
 * (ac_lsj_trans_reduced), as in acatsp; "m1r2e" adds the two-electron terms
 * of the relativistic M1 to the M1r line (ac_ci_m1s2 and the orbit-orbit
 * Zeeman operator; "m1r2epart=1" spin only, "=2" orbital only); "h6" the
 * rank-2 part of Pachucki's H6 (Phys. Rev. A 69, 052502 (2004), Eq. (32))
 * per unit kappa (ac_lsj_g h6); "is" the
 * isotope-shift parameters K_NMS, K_SMS and rho(0) of every printed level
 * (ac_lsj_is_level).
 */
#include "ac_basis.h"
#include "ac_cfp.h"
#include "ac_ci.h"
#include "ac_ftnio.h"
#include "ac_ints.h"
#include "ac_lsj.h"
#include "ac_radial.h"
#include "ac_tape.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCFG 512


typedef struct {
    const ac_ci *ci;
    const ac_radial *rad;
    double (*h0)(void *, int, int); /* NFTRI h and R^k, for "mvd" */
    double (*rk0)(void *, int, int, int, int, int);
    void *h0ctx;                    /* their shared context */
    int mvd, contact;
} radctx;

static int mo_of(const radctx *x, int s)
{
    int l = x->ci->addr.nwlf[s] - 1;
    return x->rad->first[l] + x->ci->addr.nwsq[s] - 1;
}

static double get_zeta(void *p, int a, int b)
{
    const radctx *x = p;
    return ac_radial_zeta(x->rad, mo_of(x, a), mo_of(x, b));
}

static double get_t(void *p, int a, int b)
{
    const radctx *x = p;
    int i = mo_of(x, a), j = mo_of(x, b);
    return ac_radial_h(x->rad, i, j) + x->rad->z * ac_radial_moment(x->rad, i, j, -1);
}

static double get_u(void *p, int a, int b)
{
    const radctx *x = p;
    return x->rad->z * ac_radial_moment(x->rad, mo_of(x, a), mo_of(x, b), -1);
}

/* h (+ mass velocity + Darwin) */
static double get_h_mvd(void *p, int a, int b)
{
    const radctx *x = p;
    int i = mo_of(x, a), j = mo_of(x, b);
    double v = x->h0(x->h0ctx, a, b);
    if (x->mvd)
        v += ac_radial_mv(x->rad, i, j) + ac_radial_darwin(x->rad, i, j);
    return v;
}

/* Y^k potentials of MO pairs, computed once (H_SO2) */
#define YCACHE 4096
static _Thread_local struct {
    int k, c, d;
    double *y;
} ycache[YCACHE];
static _Thread_local int nycache;

static const double *ypot(const ac_radial *rad, int k, int c, int d)
{
    for (int i = 0; i < nycache; i++)
        if (ycache[i].k == k && ycache[i].c == c && ycache[i].d == d)
            return ycache[i].y;
    double *y = malloc((size_t)rad->g->n * sizeof(double));
    ac_radial_yk(rad, k, c, d, y);
    if (nycache < YCACHE) {
        ycache[nycache].k = k, ycache[nycache].c = c, ycache[nycache].d = d;
        ycache[nycache++].y = y;
    }
    return y;
}

static double get_rkg(void *p, int k, int a, int sa, int b, int sb, int c, int d)
{
    const radctx *x = p;
    return ac_radial_gdot(x->rad, mo_of(x, a), sa, mo_of(x, b), sb,
                          ypot(x->rad, k, mo_of(x, c), mo_of(x, d)));
}

/* R^k (+ the two-electron contact terms); wrapped also for "mvd" alone
   because h and R^k share ci->in.ctx */
static double get_rk_mvd(void *p, int k, int a, int b, int c, int d)
{
    const radctx *x = p;
    double v = x->rk0(x->h0ctx, k, a, b, c, d);
    if (x->contact)
        v += ac_radial_contact(x->rad, k, mo_of(x, a), mo_of(x, b), mo_of(x, c), mo_of(x, d));
    return v;
}

static double get_roo(void *p, int k, int g, int a, int b, int c, int d)
{
    const radctx *x = p;
    return ac_radial_oo(x->rad, k, g, mo_of(x, a), mo_of(x, b), mo_of(x, c), mo_of(x, d));
}

static double get_rgc(void *p, int k, int kp, int g, int a, int b, int c, int d)
{
    const radctx *x = p;
    return ac_radial_gc(x->rad, k, kp, g, mo_of(x, a), mo_of(x, b), mo_of(x, c), mo_of(x, d));
}

static double get_roz(void *p, int k, int g, int type, int a, int b, int c, int d)
{
    const radctx *x = p;
    return ac_radial_oz(x->rad, k, g, type, mo_of(x, a), mo_of(x, b), mo_of(x, c), mo_of(x, d));
}

static double get_rrc(void *p, int kind, int a, int b)
{
    const radctx *x = p;
    return ac_radial_rrc(x->rad, kind, mo_of(x, a), mo_of(x, b));
}

/* hyperfine: <r^-3> and the contact factor R_a(0) R_b(0) */
static double get_r3(void *p, int a, int b)
{
    const radctx *x = p;
    return ac_radial_moment(x->rad, mo_of(x, a), mo_of(x, b), -3);
}

static double get_r00(void *p, int a, int b)
{
    const radctx *x = p;
    return x->rad->r0[mo_of(x, a)] * x->rad->r0[mo_of(x, b)];
}

/* ac_lsj_trans_rk: kind 0 <r>, 1 and 2 the gradient factors, 3 <r^2>,
   4 <p^2>, 5 Z <1/r>, 6-9 ac_radial_dd (the M1 tensor term), 10 R_a(0) R_b(0),
   11 <r^3> */
static double get_trk(void *p, int kind, int a, int b)
{
    const radctx *x = p;
    int i = mo_of(x, a), j = mo_of(x, b);
    if (kind == 10)
        return get_r00(p, a, b);
    if (kind == 11)
        return ac_radial_moment(x->rad, i, j, 3);
    if (kind >= 6)
        return ac_radial_dd(x->rad, i, (kind - 6) & 2 ? 1 : -1, j, (kind - 6) & 1 ? 1 : -1);
    if (kind == 1 || kind == 2)
        return ac_radial_rrc(x->rad, kind, i, j);
    if (kind == 4)
        return 2.0 * get_t(p, a, b);
    if (kind == 5)
        return get_u(p, a, b);
    return ac_radial_moment(x->rad, i, j, kind == 0 ? 1 : 2);
}

/* H_SS radial integrals: int R_a R_b w_lam(cd) r^2 dr (as acatsp, uncached) */
static double get_rss(void *p, int k, int lam, int a, int b, int c, int d)
{
    const radctx *x = p;
    const ac_rgrid *g = x->rad->g;
    double *w = malloc((size_t)g->n * sizeof(double));
    ac_radial_ssw(x->rad, k, lam, mo_of(x, c), mo_of(x, d), w);
    const double *va = x->rad->val + (size_t)mo_of(x, a) * g->n;
    const double *vb = x->rad->val + (size_t)mo_of(x, b) * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * va[i] * vb[i] * g->r[i] * g->r[i] * w[i];
    free(w);
    return s;
}

/* the spin two-electron M1 integrals (ac_ci_m1s2): int R_a R_b u_lam(cd) r^2 dr,
   lam = -1 the plain R^k (uncached) */
static double get_rm1(void *p, int k, int lam, int a, int b, int c, int d)
{
    const radctx *x = p;
    if (lam < 0)
        return ac_radial_rk(x->rad, k, mo_of(x, a), mo_of(x, b), mo_of(x, c), mo_of(x, d));
    const ac_rgrid *g = x->rad->g;
    double *u = malloc((size_t)g->n * sizeof(double));
    ac_radial_m1w(x->rad, k, lam, mo_of(x, c), mo_of(x, d), u);
    const double *va = x->rad->val + (size_t)mo_of(x, a) * g->n;
    const double *vb = x->rad->val + (size_t)mo_of(x, b) * g->n;
    double s = 0.0;
    for (int i = 0; i < g->n; i++)
        s += g->w[i] * va[i] * vb[i] * g->r[i] * g->r[i] * u[i];
    free(u);
    return s;
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: %s CFPTBL nfte nftri nfaii nftrc configs [nlev [noso] [mvd]]\n",
                argv[0]);
        return 2;
    }
    int nlev = argc > 7 ? atoi(argv[7]) : 4;
    int noso = 0, mvd = 0, so2 = 0, contact = 0, oo = 0, gc2e = 0, oz2e = 0, ss = 0, rec2 = 0,
        m1r2e = 0;
    for (int i = 8; i < argc; i++) {
        if (!strcmp(argv[i], "noso"))
            noso = 1;
        else if (!strcmp(argv[i], "ss"))
            ss = 1;
        else if (!strcmp(argv[i], "mvd"))
            mvd = 1;
        else if (!strcmp(argv[i], "so2"))
            so2 = 1;
        else if (!strcmp(argv[i], "contact"))
            contact = 1;
        else if (!strcmp(argv[i], "oo"))
            oo = 1;
        else if (!strcmp(argv[i], "oz2e"))
            oz2e = 1;
        else if (!strcmp(argv[i], "rec2"))
            rec2 = 1;
        else if (!strcmp(argv[i], "zoff"))
            ac_lsj_zoff = 1;
        else if (!strcmp(argv[i], "gc2e"))
            gc2e = 1;
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
        else if (!strncmp(argv[i], "gc2epart=", 9))
            ac_gc2e_parts = atoi(argv[i] + 9);
        else if (!strcmp(argv[i], "m1r2e"))
            m1r2e = 1;
        else if (!strncmp(argv[i], "m1r2epart=", 10))
            ac_lsj_m1_2e_parts = atoi(argv[i] + 10);
        else {
            fprintf(stderr, "acfs: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    char err[256];
    ac_cfptab t;
    ac_ints ints;
    ac_nfte_head h;
    ac_ftn_file f;
    ac_basis bas;
    if (ac_cfp_load(&t, argv[1], err, sizeof err) ||
        ac_ints_load(&ints, argv[3], err, sizeof err) ||
        ac_basis_load(&bas, argv[4], argv[5], err, sizeof err)) {
        fprintf(stderr, "acfs: %s\n", err);
        return 1;
    }
    if (ac_ftn_open(&f, argv[2], "rb") || ac_nfte_read_head(&f, &h, err, sizeof err)) {
        fprintf(stderr, "acfs: %s: %s\n", argv[2], err);
        return 1;
    }
    ac_ftn_close(&f);

    static int seq[MAXCFG][AC_CSF_MAXSH], occ[MAXCFG][AC_CSF_MAXSH];
    int nocc[MAXCFG], ncfg = 0;
    FILE *cf = fopen(argv[6], "r");
    if (!cf) {
        perror(argv[6]);
        return 1;
    }
    char line[1024];
    while (fgets(line, sizeof line, cf) && ncfg < MAXCFG) {
        int n = 0;
        for (char *tok = strtok(line, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
            if (tok[0] == '#')
                break;
            if (n < AC_CSF_MAXSH && sscanf(tok, "%d*%d", &seq[ncfg][n], &occ[ncfg][n]) == 2)
                n++;
        }
        if (n > 0)
            nocc[ncfg++] = n;
    }
    fclose(cf);

    ac_ci ci;
    if (ac_ci_from_configs(&ci, &t, &ints, &h, ncfg, nocc, (const int (*)[AC_CSF_MAXSH])seq,
                           (const int (*)[AC_CSF_MAXSH])occ, err, sizeof err)) {
        fprintf(stderr, "acfs: %s\n", err);
        return 1;
    }
    ac_rgrid g;
    ac_radial rad;
    if (ac_rgrid_init(&g, AC_RGRID_RMIN, AC_RGRID_RMAX, AC_RGRID_H) || ac_radial_build(&rad, &bas, &g, err, sizeof err)) {
        fprintf(stderr, "acfs: radial: %s\n", err);
        return 1;
    }
    for (int s = 1; s <= ci.nwt; s++) {
        int l = ci.addr.nwlf[s] - 1;
        if (l >= bas.nl || ci.addr.nwsq[s] > bas.nob[l]) {
            fprintf(stderr, "acfs: orbital %d has no MO in the vector file\n", s);
            return 1;
        }
    }
    radctx rc = {&ci, &rad, ci.in.h, ci.in.rk, ci.in.ctx, mvd, contact};
    ci.in.zctx = &rc;
    if (!noso)
        ci.in.zeta = get_zeta;
    if (so2)
        ci.in.rkg = get_rkg;
    if (ss)
        ci.in.rss = get_rss;
    if (oo)
        ci.in.roo = get_roo;
    if (oz2e || m1r2e)
        ci.in.roz = get_roz;
    if (m1r2e)
        ci.in.rm1 = get_rm1;
    if (rec2)
        ci.in.rrc = get_rrc;
    if (gc2e)
        ci.in.rgc = get_rgc;
    if (mvd || contact) {
        ci.in.h = get_h_mvd;
        ci.in.rk = get_rk_mvd;
        ci.in.ctx = &rc;
    }
    printf("acfs: %d configurations, %d CSFs, Z = %g, H_SO1 %s, H_SO2 %s,"
           " mass velocity + Darwin %s\n",
           ncfg, ci.ncsf, bas.z, noso ? "off" : "on", so2 ? "on" : "off", mvd ? "on" : "off");
    for (int s = 1; mvd && s <= ci.nwt; s++)
        printf("mvd(%d,%d) = %.15e hartree   (mass velocity %.15e, Darwin %.15e)\n", s, s,
               get_h_mvd(&rc, s, s) - rc.h0(rc.h0ctx, s, s),
               ac_radial_mv(&rad, mo_of(&rc, s), mo_of(&rc, s)),
               ac_radial_darwin(&rad, mo_of(&rc, s), mo_of(&rc, s)));
    for (int s = 1; s <= ci.nwt; s++) {
        int l = ci.addr.nwlf[s] - 1;
        if (l > 0)
            printf("zeta(%d,%d) = %.15e hartree = %.6f cm-1   (l = %d, MO %d)\n", s, s,
                   get_zeta(&rc, s, s), get_zeta(&rc, s, s) * AC_HARTREE_CM, l, ci.addr.nwsq[s]);
    }

    if (ac_lsj_print_levels(stdout, &ci, nlev, get_t, get_u, &rc, err, sizeof err)) {
        fprintf(stderr, "acfs: %s\n", err);
        return 1;
    }

    ac_radial_free(&rad);
    ac_rgrid_free(&g);
    ac_ci_free(&ci);
    ac_basis_free(&bas);
    ac_nfte_head_free(&h);
    ac_ints_free(&ints);
    ac_cfp_free(&t);
    return 0;
}
