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
 *
 * For programs outside (none of these changes the output of a run without
 * them):
 *   info        prints, before everything else, the version and commit, the
 *               number of configurations and the number of CSFs of each
 *               (2S+1, L) and in total
 *   dump=PREFIX writes PREFIX.csf.tsv (the CSFs), PREFIX.J<2J>.{idx,vec,e,
 *               res}.npy (the levels of each J block; ac_lsj.h) and
 *               PREFIX.meta.json (options, conventions, the SHA-256 of the
 *               CSF list); implies info
 *   nlevJ=2J:n,2J:n,...  the number of levels of single J blocks (0 skips
 *               the block); the others keep nlev
 *   tol=X       the residual at which the Davidson iteration stops (default
 *               1e-11)
 *   dav=N       J blocks from dimension N on go to Davidson (default 500;
 *               0: always dsyev)
 *   ls          the LS levels instead: the Coulomb h alone (h_C), one block
 *               for each (2S+1, L), the same CSFs and phases; no property;
 *               with dump= the files are
 *               PREFIX.S<2S+1>L<L>.{idx,vec,e,res}.npy. The other terms
 *               (H_SO1 unless noso, so2, ss, oo, mvd, contact) never enter
 *               these levels; they are allowed with proj or dumpmat only
 *   nlevSL=4F:n,2[11]:n,...  with ls, the number of levels of single
 *               blocks (L as its letter or as [number])
 *   proj        (with ls and dump=) the terms between the LS levels:
 *               <a|h_T|b> for h_OO, h_MVD, h_CONTACT within each block and
 *               the reduced <a||w_T||b> for w_SO1, w_SO2, w_SS between the
 *               blocks, PREFIX.proj.<term>.<bra>.<ket>.npy; computed in a
 *               second pass that stores no matrix
 *   dumpmat     (with dump=) every term as the nonzero elements i <= j of
 *               its matrix, PREFIX.<term>.{i,j,v}.npy with <term> = h_C,
 *               h_OO, h_MVD, h_CONTACT, w_SO1, w_SO2, w_SS; refused above
 *               20000 CSFs unless dumpmat=force
 *   nog         the levels without the Zeeman factors
 * With proj or dumpmat the mass velocity + Darwin and the contact terms are
 * computed as matrices of their own and added to h (instead of being added
 * to the integrals): the levels then agree with those of a run without
 * these options to rounding only.
 */
#include "ac_basis.h"
#include "ac_cfp.h"
#include "ac_ci.h"
#include "ac_commit.h"
#include "ac_dump.h"
#include "ac_ftnio.h"
#include "ac_ints.h"
#include "ac_lsj.h"
#include "ac_radial.h"
#include "ac_tape.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char LNAME[] = "SPDFGHIKLMNOQRTUVWXYZ";

/* "2J:n,2J:n,..." into v[2J] = n (the others -1); returns the length of v */
static int parse_nlevj(const char *arg, int **v)
{
    int len = 0;
    *v = NULL;
    for (const char *p = arg; *p;) {
        int tj, n, used = 0;
        if (sscanf(p, "%d:%d%n", &tj, &n, &used) != 2 || tj < 0 || n < 0 || tj > 100000)
            return -1;
        if (tj >= len) {
            int *nv = realloc(*v, ((size_t)tj + 1) * sizeof **v);
            if (!nv)
                return -1;
            for (int i = len; i <= tj; i++)
                nv[i] = -1;
            *v = nv;
            len = tj + 1;
        }
        (*v)[tj] = n;
        p += used;
        if (*p == ',')
            p++;
        else if (*p)
            return -1;
    }
    return len;
}

/* "4F:n,2[11]:n,..." into triples (2S, 2L, n); returns their number */
static int parse_nlevsl(const char *arg, int **v)
{
    int cnt = 0;
    *v = NULL;
    for (const char *p = arg; *p;) {
        int mult, L = -1, n, used = 0;
        char c;
        if (sscanf(p, "%d[%d]:%d%n", &mult, &L, &n, &used) == 3 && used > 0)
            ;
        else if (sscanf(p, "%d%c:%d%n", &mult, &c, &n, &used) == 3 && strchr(LNAME, c))
            L = (int)(strchr(LNAME, c) - LNAME);
        else
            return -1;
        if (mult < 1 || L < 0 || n < 0)
            return -1;
        int *nv = realloc(*v, ((size_t)cnt + 1) * 3 * sizeof **v);
        if (!nv)
            return -1;
        *v = nv;
        nv[3 * cnt] = mult - 1, nv[3 * cnt + 1] = 2 * L, nv[3 * cnt + 2] = n;
        cnt++;
        p += used;
        if (*p == ',')
            p++;
        else if (*p)
            return -1;
    }
    return cnt;
}

static void json_str(FILE *f, const char *s)
{
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\')
            fputc('\\', f);
        if ((unsigned char)*s >= 32)
            fputc(*s, f);
    }
    fputc('"', f);
}

/* the shells and the coupling of CSF i as text:
   "seq,nel,l,code,2S+1,L;..." and "2S+1,L;..." for F_0 .. F_nsh */
static void csf_text(const ac_ci *ci, int i, char *shells, size_t ns, char *cpl, size_t nc)
{
    const ac_ci_csf *x = &ci->csf[i];
    size_t a = 0, b = 0;
    shells[0] = cpl[0] = 0;
    for (int k = 1; k <= x->c.nsh && a < ns; k++)
        a += (size_t)snprintf(shells + a, ns - a, "%s%d,%d,%d,%d,%d,%d", k > 1 ? ";" : "", x->seq[k],
                              x->c.nel[k], x->c.l[k], x->c.code[k], x->c.f[k].ts + 1,
                              x->c.f[k].tl / 2);
    for (int k = 0; k <= x->c.nsh && b < nc; k++)
        b += (size_t)snprintf(cpl + b, nc - b, "%s%d,%d", k ? ";" : "", x->c.cpl[k].ts + 1,
                              x->c.cpl[k].tl / 2);
}


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

/* The one-electron radial integrals are quadratures, and the matrix elements
   ask for the same pair of MOs once for every pair of CSFs (H_SO1 took 490
   thread-seconds of a list of 7482 CSFs, the Coulomb matrix 3). Each is
   computed once per thread and kept: a table [nmo][nmo] per kind, NaN until
   asked for; the tables belong to one ac_radial and start anew for another
   (bio=). The number kept is the number computed, so nothing changes in the
   results. */
enum {
    K1_ZETA, K1_T, K1_U, K1_MVD, K1_R3, K1_MOM1, K1_MOM2, K1_MOM3, K1_RRC, /* 3 kinds */
    K1_DD = K1_RRC + 3,                                                    /* 4 kinds */
    K1_N = K1_DD + 4
};
static _Thread_local struct {
    const ac_radial *rad;
    double *tab;
} keep1[K1_N];

static double radial1(const ac_radial *rad, int kind, int i, int j)
{
    switch (kind) {
    case K1_ZETA:
        return ac_radial_zeta(rad, i, j);
    case K1_T:
        return ac_radial_h(rad, i, j) + rad->z * ac_radial_moment(rad, i, j, -1);
    case K1_U:
        return rad->z * ac_radial_moment(rad, i, j, -1);
    case K1_MVD:
        return ac_radial_mv(rad, i, j) + ac_radial_darwin(rad, i, j);
    case K1_R3:
        return ac_radial_moment(rad, i, j, -3);
    case K1_MOM1:
    case K1_MOM2:
    case K1_MOM3:
        return ac_radial_moment(rad, i, j, kind - K1_MOM1 + 1);
    default:
        if (kind < K1_DD)
            return ac_radial_rrc(rad, kind - K1_RRC, i, j);
        return ac_radial_dd(rad, i, (kind - K1_DD) & 2 ? 1 : -1, j, (kind - K1_DD) & 1 ? 1 : -1);
    }
}

static double kept1(const ac_radial *rad, int kind, int i, int j)
{
    size_t n = (size_t)rad->nmo;
    if (keep1[kind].rad != rad) {
        free(keep1[kind].tab);
        keep1[kind].tab = malloc((n > 0 ? n * n : 1) * sizeof(double));
        keep1[kind].rad = keep1[kind].tab ? rad : NULL;
        for (size_t q = 0; keep1[kind].tab && q < n * n; q++)
            keep1[kind].tab[q] = NAN;
    }
    if (!keep1[kind].tab)
        return radial1(rad, kind, i, j);
    double *v = &keep1[kind].tab[(size_t)i * n + (size_t)j];
    if (isnan(*v))
        *v = radial1(rad, kind, i, j);
    return *v;
}

static double get_zeta(void *p, int a, int b)
{
    const radctx *x = p;
    return kept1(x->rad, K1_ZETA, mo_of(x, a), mo_of(x, b));
}

static double get_t(void *p, int a, int b)
{
    const radctx *x = p;
    return kept1(x->rad, K1_T, mo_of(x, a), mo_of(x, b));
}

static double get_u(void *p, int a, int b)
{
    const radctx *x = p;
    return kept1(x->rad, K1_U, mo_of(x, a), mo_of(x, b));
}

/* h (+ mass velocity + Darwin) */
static double get_h_mvd(void *p, int a, int b)
{
    const radctx *x = p;
    int i = mo_of(x, a), j = mo_of(x, b);
    double v = x->h0(x->h0ctx, a, b);
    if (x->mvd)
        v += kept1(x->rad, K1_MVD, i, j);
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

/* the terms kept apart (ac_lsj_extra): mass velocity + Darwin as h with no
   R^k, the contact terms as R^k with no h */
static double get_mvd_only(void *p, int a, int b)
{
    const radctx *x = p;
    return kept1(x->rad, K1_MVD, mo_of(x, a), mo_of(x, b));
}

static double get_contact_only(void *p, int k, int a, int b, int c, int d)
{
    const radctx *x = p;
    return ac_radial_contact(x->rad, k, mo_of(x, a), mo_of(x, b), mo_of(x, c), mo_of(x, d));
}

static double zero_h(void *p, int a, int b)
{
    (void)p, (void)a, (void)b;
    return 0.0;
}

static double zero_rk(void *p, int k, int a, int b, int c, int d)
{
    (void)p, (void)k, (void)a, (void)b, (void)c, (void)d;
    return 0.0;
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
    if (kind < 0 || kind > 2)
        return ac_radial_rrc(x->rad, kind, mo_of(x, a), mo_of(x, b));
    return kept1(x->rad, K1_RRC + kind, mo_of(x, a), mo_of(x, b));
}

/* hyperfine: <r^-3> and the contact factor R_a(0) R_b(0) */
static double get_r3(void *p, int a, int b)
{
    const radctx *x = p;
    return kept1(x->rad, K1_R3, mo_of(x, a), mo_of(x, b));
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
        return kept1(x->rad, K1_MOM3, i, j);
    if (kind >= 6)
        return kept1(x->rad, K1_DD + (kind - 6), i, j);
    if (kind == 1 || kind == 2)
        return kept1(x->rad, K1_RRC + kind, i, j);
    if (kind == 4)
        return 2.0 * get_t(p, a, b);
    if (kind == 5)
        return get_u(p, a, b);
    return kept1(x->rad, kind == 0 ? K1_MOM1 : K1_MOM2, i, j);
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
        fprintf(stderr, "usage: %s CFPTBL nfte nftri nfaii nftrc configs [nlev [options]]\n",
                argv[0]);
        return 2;
    }
    int nlev = argc > 7 ? atoi(argv[7]) : 4;
    int noso = 0, mvd = 0, so2 = 0, contact = 0, oo = 0, gc2e = 0, oz2e = 0, ss = 0, rec2 = 0,
        m1r2e = 0;
    int info = 0, ls = 0, props = 0, nsl = 0, proj = 0, dumpmat = 0;
    const char *dump = NULL;
    int *nlevj = NULL, *nlevsl = NULL;
    for (int i = 8; i < argc; i++) {
        if (!strcmp(argv[i], "info"))
            info = 1;
        else if (!strncmp(argv[i], "dump=", 5) && argv[i][5])
            dump = argv[i] + 5, info = 1;
        else if (!strcmp(argv[i], "ls"))
            ls = 1;
        else if (!strcmp(argv[i], "proj"))
            proj = 1;
        else if (!strcmp(argv[i], "dumpmat"))
            dumpmat = 1;
        else if (!strcmp(argv[i], "dumpmat=force"))
            dumpmat = 2;
        else if (!strcmp(argv[i], "nog"))
            ac_lsj_nog = 1;
        else if (!strncmp(argv[i], "tol=", 4) && atof(argv[i] + 4) > 0.0)
            ac_lsj_tol = atof(argv[i] + 4);
        else if (!strncmp(argv[i], "dav=", 4))
            ac_lsj_davidson = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "nlevJ=", 6)) {
            int len = parse_nlevj(argv[i] + 6, &nlevj);
            if (len < 0) {
                fprintf(stderr, "acfs: %s: expected nlevJ=2J:n,2J:n,...\n", argv[i]);
                return 2;
            }
            ac_lsj_nlev_j = nlevj;
            ac_lsj_nlev_j_len = len;
        } else if (!strncmp(argv[i], "nlevSL=", 7)) {
            nsl = parse_nlevsl(argv[i] + 7, &nlevsl);
            if (nsl < 0) {
                fprintf(stderr, "acfs: %s: expected nlevSL=4F:n,2[11]:n,...\n", argv[i]);
                return 2;
            }
            ac_lsj_nlev_sl = nlevsl;
            ac_lsj_nlev_sl_n = nsl;
        } else if (!strcmp(argv[i], "noso"))
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
        /* the properties (everything that is neither a term of H nor one of the options
           above): not with ls */
        static const char *const plain[] = {"info", "ls", "noso", "proj", "dumpmat", "dumpmat=force",
                                            "nog", "ss", "mvd", "so2", "contact", "oo"};
        int isplain = !strncmp(argv[i], "dump=", 5) || !strncmp(argv[i], "tol=", 4) ||
                      !strncmp(argv[i], "dav=", 4) || !strncmp(argv[i], "nlevJ=", 6) ||
                      !strncmp(argv[i], "nlevSL=", 7);
        for (size_t k = 0; k < sizeof plain / sizeof *plain; k++)
            isplain |= !strcmp(argv[i], plain[k]);
        props |= !isplain;
    }
    /* the terms are kept apart for the files of proj and dumpmat */
    int apart = proj || dumpmat;
    if ((proj || dumpmat) && !dump) {
        fprintf(stderr, "acfs: proj and dumpmat need dump=PREFIX\n");
        return 2;
    }
    if (proj && !ls) {
        fprintf(stderr, "acfs: proj needs ls (the terms between the LS levels)\n");
        return 2;
    }
    if (ls && (props || nlevj || ac_lsj_nog)) {
        fprintf(stderr, "acfs: ls takes no property, no nlevJ= and no nog\n");
        return 2;
    }
    if (ls && !apart && (so2 || ss || oo || mvd || contact)) {
        fprintf(stderr, "acfs: the LS levels are those of the Coulomb h alone; so2, ss, oo, mvd and "
                        "contact go with ls only for the files of proj or dumpmat\n");
        return 2;
    }
    /* ls alone: no H_SO1 either */
    if (ls && !apart)
        noso = 1;
    ac_lsj_proj = proj;
    ac_lsj_dumpmat = dumpmat;
    if (nlevsl && !ls) {
        fprintf(stderr, "acfs: nlevSL= needs ls\n");
        return 2;
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

    /* any number of configurations (the arrays grow) */
    int (*seq)[AC_CSF_MAXSH] = NULL, (*occ)[AC_CSF_MAXSH] = NULL;
    int *nocc = NULL, ncfg = 0, cfgcap = 0;
    FILE *cf = fopen(argv[6], "r");
    if (!cf) {
        perror(argv[6]);
        return 1;
    }
    char line[1024];
    while (fgets(line, sizeof line, cf)) {
        if (ncfg == cfgcap) {
            cfgcap = cfgcap ? 2 * cfgcap : 512;
            void *a = realloc(seq, (size_t)cfgcap * sizeof *seq);
            void *b = a ? realloc(occ, (size_t)cfgcap * sizeof *occ) : NULL;
            void *c = b ? realloc(nocc, (size_t)cfgcap * sizeof *nocc) : NULL;
            if (a)
                seq = a;
            if (b)
                occ = b;
            if (!c) {
                fprintf(stderr, "acfs: out of memory at configuration %d\n", ncfg + 1);
                return 1;
            }
            nocc = c;
        }
        int n = 0;
        for (char *tok = strtok(line, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
            int sq, oc;
            if (tok[0] == '#')
                break;
            if (sscanf(tok, "%d*%d", &sq, &oc) != 2)
                continue;
            if (n == AC_CSF_MAXSH) {
                fprintf(stderr, "acfs: configuration %d of %s lists more than %d orbitals\n",
                        ncfg + 1, argv[6], AC_CSF_MAXSH);
                return 1;
            }
            seq[ncfg][n] = sq, occ[ncfg][n] = oc;
            n++;
        }
        if (n > 0)
            nocc[ncfg++] = n;
    }
    fclose(cf);
    if (ncfg == 0) {
        fprintf(stderr, "acfs: %s has no configuration\n", argv[6]);
        return 1;
    }

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
    /* the CSFs of each (2S+1, L), in the order of 2S+1, then L */
    int ntab = 0, (*tab)[3] = NULL;
    for (int i = 0; i < ci.ncsf; i++) {
        ac_lab a = ci.csf[i].c.cpl[ci.csf[i].c.nsh];
        int k;
        for (k = 0; k < ntab && (tab[k][0] != a.ts || tab[k][1] != a.tl); k++)
            ;
        if (k == ntab) {
            void *nt = realloc(tab, ((size_t)ntab + 1) * sizeof *tab);
            if (!nt) {
                fprintf(stderr, "acfs: out of memory\n");
                return 1;
            }
            tab = nt;
            for (k = ntab++; k > 0 && (tab[k - 1][0] > a.ts ||
                                       (tab[k - 1][0] == a.ts && tab[k - 1][1] > a.tl));
                 k--)
                memcpy(tab[k], tab[k - 1], sizeof *tab);
            tab[k][0] = a.ts, tab[k][1] = a.tl, tab[k][2] = 0;
        }
        tab[k][2]++;
    }
    if (info) {
        printf("ATOMCI2 %s, commit %s\n", AC_VERSION, AC_COMMIT);
        printf("%d configurations, %d CSFs:", ncfg, ci.ncsf);
        for (int k = 0; k < ntab; k++) {
            if (tab[k][1] / 2 < (int)sizeof LNAME - 1)
                printf(" %d%c %d", tab[k][0] + 1, LNAME[tab[k][1] / 2], tab[k][2]);
            else
                printf(" %d[%d] %d", tab[k][0] + 1, tab[k][1] / 2, tab[k][2]);
        }
        printf("\n");
    }
    /* dump=: the CSFs, one line each, and the SHA-256 of their keys */
    char sha[65] = "";
    if (dump) {
        size_t plen = strlen(dump) + 32;
        char *path = malloc(plen);
        snprintf(path, plen, "%s.csf.tsv", dump);
        FILE *tf = fopen(path, "w");
        if (!tf) {
            perror(path);
            return 1;
        }
        ac_sha256 hs;
        ac_sha256_init(&hs);
        fprintf(tf, "i\tcfg\tconfig\tnsh\tshells\tcpl\tS2p1\tL\tkey\n");
        int c = 0;
        for (int i = 0; i < ci.ncsf; i++) {
            const ac_ci_csf *x = &ci.csf[i];
            /* its configuration: the first from c on whose listed orbitals have these
               occupations (the CSFs are in the order of the configurations) */
            for (; c < ncfg; c++) {
                int ok = 1;
                for (int k = 0; k < nocc[c] && ok; k++) {
                    int have = 0;
                    for (int q = 1; q <= x->c.nsh; q++)
                        if (x->seq[q] == seq[c][k])
                            have = x->c.nel[q];
                    ok = have == occ[c][k];
                }
                if (ok)
                    break;
            }
            if (c == ncfg) {
                fprintf(stderr, "acfs: CSF %d belongs to no configuration\n", i);
                return 1;
            }
            char shells[AC_CSF_MAXSH * 48], cpl[AC_CSF_MAXSH * 24];
            csf_text(&ci, i, shells, sizeof shells, cpl, sizeof cpl);
            ac_lab a = x->c.cpl[x->c.nsh];
            fprintf(tf, "%d\t%d\t", i, c);
            for (int k = 0; k < nocc[c]; k++)
                fprintf(tf, "%s%d*%d", k ? " " : "", seq[c][k], occ[c][k]);
            fprintf(tf, "\t%d\t%s\t%s\t%d\t%d\t%s|%s\n", x->c.nsh, shells, cpl, a.ts + 1, a.tl / 2,
                    shells, cpl);
            ac_sha256_add(&hs, shells, strlen(shells));
            ac_sha256_add(&hs, "|", 1);
            ac_sha256_add(&hs, cpl, strlen(cpl));
            ac_sha256_add(&hs, "\n", 1);
        }
        if (fclose(tf)) {
            perror(path);
            return 1;
        }
        free(path);
        ac_sha256_hex(&hs, sha);
        printf("CSF list sha256 %s\n", sha);
        ac_lsj_dump = dump;
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
    static ac_lsj_hterm extra[2];
    if (apart) {
        if (mvd)
            extra[ac_lsj_nextra++] = (ac_lsj_hterm){"h_MVD", get_mvd_only, zero_rk, &rc};
        if (contact)
            extra[ac_lsj_nextra++] = (ac_lsj_hterm){"h_CONTACT", zero_h, get_contact_only, &rc};
        ac_lsj_extra = extra;
    } else if (mvd || contact) {
        ci.in.h = get_h_mvd;
        ci.in.rk = get_rk_mvd;
        ci.in.ctx = &rc;
    }
    printf("acfs: %d configurations, %d CSFs, Z = %g, H_SO1 %s, H_SO2 %s,"
           " mass velocity + Darwin %s\n",
           ncfg, ci.ncsf, bas.z, noso ? "off" : "on", so2 ? "on" : "off", mvd ? "on" : "off");
    for (int s = 1; mvd && s <= ci.nwt; s++)
        printf("mvd(%d,%d) = %.15e hartree   (mass velocity %.15e, Darwin %.15e)\n", s, s,
               apart ? get_mvd_only(&rc, s, s) : get_h_mvd(&rc, s, s) - rc.h0(rc.h0ctx, s, s),
               ac_radial_mv(&rad, mo_of(&rc, s), mo_of(&rc, s)),
               ac_radial_darwin(&rad, mo_of(&rc, s), mo_of(&rc, s)));
    for (int s = 1; s <= ci.nwt; s++) {
        int l = ci.addr.nwlf[s] - 1;
        if (l > 0)
            printf("zeta(%d,%d) = %.15e hartree = %.6f cm-1   (l = %d, MO %d)\n", s, s,
                   get_zeta(&rc, s, s), get_zeta(&rc, s, s) * AC_HARTREE_CM, l, ci.addr.nwsq[s]);
    }

    if (ls ? ac_lsj_print_ls(stdout, &ci, nlev, err, sizeof err)
           : ac_lsj_print_levels(stdout, &ci, nlev, get_t, get_u, &rc, err, sizeof err)) {
        fprintf(stderr, "acfs: %s\n", err);
        return 1;
    }
    if (dump) {
        size_t plen = strlen(dump) + 32;
        char *path = malloc(plen);
        snprintf(path, plen, "%s.meta.json", dump);
        FILE *mf = fopen(path, "w");
        if (!mf) {
            perror(path);
            return 1;
        }
        fprintf(mf, "{\n  \"program\": \"acfs\",\n  \"version\": \"%s\",\n  \"commit\": \"%s\",\n",
                AC_VERSION, AC_COMMIT);
        fprintf(mf, "  \"options\": [");
        for (int i = 7; i < argc; i++) {
            fprintf(mf, i > 7 ? ", " : "");
            json_str(mf, argv[i]);
        }
        fprintf(mf, "],\n  \"mode\": \"%s\",\n  \"nlev\": %d,\n", ls ? "ls" : "lsj", nlev);
        fprintf(mf, "  \"davidson_tolerance\": %.3e,\n  \"davidson_from_dimension\": %d,\n",
                ac_lsj_tol, ac_lsj_davidson);
        fprintf(mf, "  \"units\": \"hartree\",\n  \"alpha\": %.17g,\n  \"g_s\": %.17g,\n  \"Z\": %g,\n",
                AC_ALPHA, AC_GS, bas.z);
        fprintf(mf,
                "  \"terms\": {\"coulomb\": true, \"so1\": %s, \"so2\": %s, \"ss\": %s, \"oo\": %s, "
                "\"mvd\": %s, \"contact\": %s},\n",
                !noso ? "true" : "false", so2 ? "true" : "false", ss ? "true" : "false",
                oo ? "true" : "false", mvd ? "true" : "false", contact ? "true" : "false");
        fprintf(mf, "  \"n_configurations\": %d,\n  \"n_csf\": %d,\n  \"csf_per_term\": [", ncfg,
                ci.ncsf);
        for (int k = 0; k < ntab; k++)
            fprintf(mf, "%s{\"S2p1\": %d, \"L\": %d, \"n\": %d}", k ? ", " : "", tab[k][0] + 1,
                    tab[k][1] / 2, tab[k][2]);
        fprintf(mf, "],\n  \"csf_file\": \"csf.tsv\",\n  \"csf_sha256\": \"%s\",\n", sha);
        fprintf(mf, "  \"csf_sha256_of\": \"the key column of csf.tsv, each key followed by a "
                    "newline, in the order of the CSFs\",\n");
        fprintf(mf, "  \"blocks\": [\n%s\n  ],\n", ac_lsj_dump_blocks());
        fprintf(mf, "  \"terms_in_the_levels\": \"%s\",\n",
                ls ? "h_C alone (the LS levels)" : "all the terms switched on");
        if (dumpmat)
            fprintf(mf, "  \"dumpmat\": \"PREFIX.<term>.i.npy, .j.npy (int64) and .v.npy (float64): "
                        "the nonzero elements i <= j of each term switched on, sorted by i, then "
                        "j; <term> = h_C, h_OO, h_MVD, h_CONTACT (terms of h, symmetric), w_SO1, "
                        "w_SO2 (terms of w), w_SS (= w2); the element (j, i) of a term of w or "
                        "w2 follows from conventions.transpose\",\n");
        if (proj)
            fprintf(mf, "  \"proj_files\": \"PREFIX.<file>: float64 [nroot of bra][nroot of ket]; "
                        "for h_OO, h_MVD, h_CONTACT the elements <a|h_T|b> between the LS levels "
                        "of one block; for w_SO1, w_SO2, w_SS the reduced <a||w_T||b> = sum_ij "
                        "c_a(i) <i||W_T||j> c_b(j) between the levels a of the bra block and b "
                        "of the ket block, which enter a J block as w and w2 do in "
                        "conventions.j_block\",\n  \"proj\": [\n%s\n  ],\n",
                    ac_lsj_dump_proj());
        fprintf(mf, "  \"block_files\": \"PREFIX.<tag>.idx.npy int64 [dim] (the CSF of each basis "
                    "function, from 0), .vec.npy float64 [nroot][dim], .e.npy float64 [nroot] "
                    "(hartree), .res.npy float64 [nroot] (|H c - E c|)\",\n");
        fprintf(mf, "  \"conventions\": {\n");
        fprintf(mf, "    \"csf\": \"the CSFs and their phases depend on the configurations only, "
                    "not on the terms switched on; shells are coupled in the order of the sequence "
                    "orbitals; csf.tsv: shells = seq,nel,l,code,2S+1,L of each shell, cpl = 2S+1,L "
                    "of F_0 .. F_nsh\",\n");
        fprintf(mf, "    \"basis_of_J\": \"every CSF whose total (S, L) couples to J, in the order "
                    "of the CSFs; |S L J M> = sum <S M_S L M_L|J M> |S L M_S M_L> (spin first)\",\n");
        fprintf(mf, "    \"j_block\": \"<i J|H|j J> = delta(S L, S' L') h_ij + (-1/sqrt 3) "
                    "(-1)^(S'+L+J) {J L S; 1 S' L'} w_ij + (1/sqrt 5) (-1)^(S'+L+J) "
                    "{J L S; 2 S' L'} w2_ij, with (S, L) of CSF i and (S', L') of CSF j "
                    "(Edmonds 1957, Eq. (7.1.6))\",\n");
        fprintf(mf, "    \"h\": \"h_ij = <i|H|j> = h_C + h_OO + h_MVD + h_CONTACT: the Coulomb "
                    "Hamiltonian (with the one-electron part) plus the orbit-orbit, mass velocity "
                    "+ Darwin and two-electron contact terms switched on\",\n");
        fprintf(mf, "    \"w\": \"w_ij = w_SO1 + w_SO2 = <i||W||j>, reduced in spin and in orbit "
                    "(Edmonds), of the double tensor of spin rank 1 and orbital rank 1 whose "
                    "J-scalar is H_SO1 + H_SO2 (c/include/ac_ci.h, ac_ci_so1 and ac_ci_so2)\",\n");
        fprintf(mf, "    \"w2\": \"w2_ij = <i||W_SS||j> of the double tensor of spin rank 2 and "
                    "orbital rank 2 whose J-scalar is H_SS (ac_ci_ss)\",\n");
        fprintf(mf, "    \"transpose\": \"<j||W||i> = (-1)^((S_i - S_j) + (L_i - L_j)) <i||W||j>, "
                    "for W and W_SS\"\n  }\n}\n");
        if (fclose(mf)) {
            perror(path);
            return 1;
        }
        free(path);
    }
    free(tab);
    free(nlevj);
    free(nlevsl);
    free(seq);
    free(occ);
    free(nocc);

    ac_radial_free(&rad);
    ac_rgrid_free(&g);
    ac_ci_free(&ci);
    ac_basis_free(&bas);
    ac_nfte_head_free(&h);
    ac_ints_free(&ints);
    ac_cfp_free(&t);
    return 0;
}
