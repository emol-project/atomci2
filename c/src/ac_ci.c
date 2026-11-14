/*
 * ac_ci.c -- the non-relativistic CI of stage 2 (see ac_ci.h).
 *
 * The Coulomb CI follows the Fortran ATOMCI; the Breit-Pauli and property
 * operators were derived anew in the tensor form of the technical note (Secs. 4.2
 * and 5), not taken from another program. Sources cited below by
 * author and year:
 *   METECC (1993)    F. Sasaki, M. Sekiya, T. Noro, K. Ohtsuki, Y. Osanai,
 *                    "Non-relativistic configuration interaction calculations
 *                    for many-electron atoms: ATOMCI", in Methods and
 *                    Techniques in Computational Chemistry: METECC-94,
 *                    E. Clementi (ed.), Cagliari (1993), pp. 163-219,
 *                    Eqs. (44), (45)
 *   Bethe-Salpeter (1957)  H. A. Bethe and E. E. Salpeter, Quantum Mechanics
 *                    of One- and Two-Electron Atoms, Springer (1957), sections
 *                    38-39 (the Breit-Pauli operators)
 *   Edmonds (1957)   A. R. Edmonds, Angular Momentum in Quantum Mechanics,
 *                    Princeton University Press (1957), Eq. (7.1.1)
 *   Racah (1942)     G. Racah, Phys. Rev. 62, 438 (1942)
 *   Pachucki (2004)  K. Pachucki, Phys. Rev. A 69, 052502 (2004): Eq. (32)
 *                    (alpha^2 corrections to g_J), Eqs. (60), (62)
 *                    (relativistic M1)
 *   Beltran-Lopez (1968)  V. Beltran-Lopez, E. Ley Koo, N. Segovia,
 *                    E. Blaisten, Phys. Rev. 172, 44 (1968), Eq. (3) (the
 *                    recoil term of g_L)
 *   Hegstrom (1973)  R. A. Hegstrom, Phys. Rev. A 7, 451 (1973): Eq. (21)
 *                    (H3, H4 spin-orbit, H6 orbit-orbit, with
 *                    pi = p - e A), Eq. (23) (nuclear-mass terms)
 *   Phillips (1949)  M. Phillips, Phys. Rev. 76, 1803 (1949) (the two-body
 *                    recoil term T_H1 of the orbital moment)
 *   Davidson (1975)  E. R. Davidson, J. Comput. Phys. 17, 87 (1975)
 *   Liu (1978)       B. Liu, in Numerical Algorithms in Chemistry: Algebraic
 *                    Methods, C. Moler and I. Shavitt (eds.), LBL-8158
 *                    (1978), p. 49
 */
#include "ac_ci.h"
#include "ac_csfgen.h"
#include "ac_factor.h"
#include "ac_optree.h"
#include "ac_radial.h"
#include "ac_wigner.h"

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

/* the first state of shell l with n electrons: the 1S vacuum or closed shell */
static int scalar_code(const ac_cfptab *t, int l, int n)
{
    const int *codes;
    return ac_cfp_states(t, l, n, &codes) > 0 ? codes[0] : 0;
}

/* NFTRI values at their ac_addr addresses */
static double nftri_h(void *ctx, int a, int b)
{
    const ac_ci *ci = ctx;
    return ci->ints->x[ac_addr_one(&ci->addr, a, b)];
}

static double nftri_rk(void *ctx, int k, int a, int b, int c, int d)
{
    const ac_ci *ci = ctx;
    return ci->ints->x[ac_addr_two(&ci->addr, 2 * k + 1, a, b, c, d)];
}

/* orbitals and integral addressing from a tape header */
static int setup_orbitals(ac_ci *ci, const ac_cfptab *t, const ac_ints *ints,
                          const ac_nfte_head *h, char *err, int errlen)
{
    memset(ci, 0, sizeof *ci);
    ci->t = t;
    ci->ints = ints;
    ci->in.h = nftri_h;
    ci->in.rk = nftri_rk;
    ci->in.ctx = ci;
    ci->nwt = h->nwt;
    if (h->nwt > AC_ADDR_MAXORB)
        return fail(err, errlen, "too many orbitals (%d)", h->nwt);
    for (int i = 1; i <= h->nwt; i++)
        ci->lorb[i] = (h->nwtn[i - 1] % 256 - 1) / 2;
    int32_t *nwth = malloc((size_t)h->nwt * sizeof *nwth);
    if (!nwth)
        return fail(err, errlen, "out of memory");
    ac_nfte_nwth(t, h, nwth);
    int r = ac_addr_setup(&ci->addr, t, 0, h->nwt, nwth, NULL, h->nwsq);
    free(nwth);
    if (r)
        return fail(err, errlen, "ac_addr_setup failed");
    if (ci->addr.igtad != ints->n)
        return fail(err, errlen, "%ld integrals on NFTRI, %d addresses expected",
                    ints->n, ci->addr.igtad);
    return 0;
}

int ac_ci_from_configs(ac_ci *ci, const ac_cfptab *t, const ac_ints *ints,
                       const ac_nfte_head *h, int ncfg, const int *nocc,
                       const int (*seq)[AC_CSF_MAXSH], const int (*occ)[AC_CSF_MAXSH],
                       char *err, int errlen)
{
    if (setup_orbitals(ci, t, ints, h, err, errlen))
        return -1;
    ac_csflist all;
    ac_csflist_init(&all);
    int cap = 0;
    for (int c = 0; c < ncfg; c++) {
        /* slots: the listed orbitals and the default closed shells, in
           sequence order */
        int sseq[AC_CSF_MAXSH], sl[AC_CSF_MAXSH], socc[AC_CSF_MAXSH], n = 0;
        for (int s = 1; s <= h->nwt; s++) {
            int k = -1, closed = 0;
            for (int i = 0; i < nocc[c]; i++)
                if (seq[c][i] == s)
                    k = i;
            for (int j = 0; k < 0 && j < h->nwf; j++)
                closed |= h->nwtc[j] == s;
            if (k < 0 && !closed)
                continue;
            if (n == AC_CSF_MAXSH) {
                ac_csflist_free(&all);
                return fail(err, errlen, "configuration %d has too many shells", c);
            }
            sseq[n] = s;
            sl[n] = ci->lorb[s];
            socc[n] = k >= 0 ? occ[c][k] : 4 * ci->lorb[s] + 2;
            n++;
        }
        int before = all.len;
        if (ac_csfgen(t, n, sl, socc, &all) < 0) {
            ac_csflist_free(&all);
            return fail(err, errlen, "cannot generate the CSFs of configuration %d", c);
        }
        if (all.len > cap) {
            ac_ci_csf *nc = realloc(ci->csf, (size_t)all.len * sizeof *nc);
            if (!nc) {
                ac_csflist_free(&all);
                return fail(err, errlen, "out of memory");
            }
            ci->csf = nc;
            cap = all.len;
        }
        for (int i = before; i < all.len; i++) {
            ac_ci_csf *x = &ci->csf[ci->ncsf++];
            memset(x, 0, sizeof *x);
            x->c = all.v[i];
            for (int k = 1; k <= n; k++)
                x->seq[k] = sseq[k - 1];
        }
    }
    ac_csflist_free(&all);
    return 0;
}

int ac_ci_from_nfte(ac_ci *ci, const ac_cfptab *t, const ac_ints *ints,
                    const ac_nfte_head *h, char *err, int errlen)
{
    if (setup_orbitals(ci, t, ints, h, err, errlen))
        return -1;

    /* count the CSFs of the image */
    const int16_t *W = h->nwfsg - 1;
    int ncsf = 0;
    for (int js = 1; js < h->nint && W[js] > 0; js += W[js] * (W[js + 1] + 3) + 2)
        ncsf += W[js + 1];
    ci->csf = calloc((size_t)ncsf, sizeof *ci->csf);
    if (!ci->csf)
        return fail(err, errlen, "out of memory");

    for (int js = 1; js < h->nint && W[js] > 0; js += W[js] * (W[js + 1] + 3) + 2) {
        int ns = W[js], nrps = W[js + 1];
        const int16_t *nen = &W[js + 2], *neq = nen + ns, *nesl = neq + ns, *rps = nesl + ns;
        for (int r2 = 0; r2 < nrps; r2++) {
            ac_ci_csf *x = &ci->csf[ci->ncsf++];
            int nslot = 0, oi = 0;
            ac_lab prev = {0, 0};
            /* merge the open shells (in sequence order) with the default
               closed shells that the CSF does not list */
            for (int s = 1; s <= h->nwt; s++) {
                int open = oi < ns && rps[r2 * ns + oi] == s;
                int closed = 0;
                for (int j = 0; !open && j < h->nwf; j++)
                    closed |= h->nwtc[j] == s;
                if (!open && !closed)
                    continue;
                if (++nslot > AC_CSF_MAXSH)
                    return fail(err, errlen, "a CSF has more than %d shells", AC_CSF_MAXSH);
                int l = ci->lorb[s];
                x->seq[nslot] = s;
                x->c.l[nslot] = l;
                if (open) {
                    ac_shell_qn q;
                    if (ac_cfp_decode(t, nen[oi], &q) || q.tl != 2 * l)
                        return fail(err, errlen, "bad shell state code %d", nen[oi]);
                    x->c.code[nslot] = nen[oi];
                    x->c.nel[nslot] = neq[oi];
                    x->c.f[nslot].ts = q.ts;
                    x->c.f[nslot].tl = q.tL;
                    prev.ts = nesl[oi] / 256 - 1;
                    prev.tl = nesl[oi] % 256 - 1;
                    oi++;
                } else {
                    x->c.code[nslot] = scalar_code(t, l, 4 * l + 2);
                    x->c.nel[nslot] = 4 * l + 2;
                    x->c.f[nslot].ts = x->c.f[nslot].tl = 0;
                }
                x->c.cpl[nslot] = prev;
            }
            if (oi != ns)
                return fail(err, errlen, "open shells of a CSF out of sequence order");
            x->c.nsh = nslot;
        }
    }
    return 0;
}

int ac_ci_from_orbitals(ac_ci *ci, const ac_cfptab *t, int nwt, const int *lorb,
                        int ncfg, const int *occ, char *err, int errlen)
{
    memset(ci, 0, sizeof *ci);
    ci->t = t;
    if (nwt < 1 || nwt > AC_CSF_MAXSH)
        return fail(err, errlen, "%d orbitals (1..%d allowed)", nwt, AC_CSF_MAXSH);
    ci->nwt = nwt;
    for (int s = 1; s <= nwt; s++)
        ci->lorb[s] = lorb[s - 1];
    ac_csflist all;
    ac_csflist_init(&all);
    for (int c = 0; c < ncfg; c++)
        if (ac_csfgen(t, nwt, lorb, occ + (size_t)c * nwt, &all) < 0) {
            ac_csflist_free(&all);
            return fail(err, errlen, "cannot generate the CSFs of configuration %d", c);
        }
    ci->csf = calloc((size_t)(all.len ? all.len : 1), sizeof *ci->csf);
    if (!ci->csf) {
        ac_csflist_free(&all);
        return fail(err, errlen, "out of memory");
    }
    for (int i = 0; i < all.len; i++) {
        ci->csf[i].c = all.v[i];
        for (int s = 1; s <= nwt; s++)
            ci->csf[i].seq[s] = s;
    }
    ci->ncsf = all.len;
    ac_csflist_free(&all);
    return 0;
}

int ac_ci_from_orbitals_occ(ac_ci *ci, const ac_cfptab *t, int nwt, const int *lorb,
                            int ncfg, const int *occ, char *err, int errlen)
{
    memset(ci, 0, sizeof *ci);
    ci->t = t;
    if (nwt < 1 || nwt > AC_ADDR_MAXORB)
        return fail(err, errlen, "%d orbitals (1..%d allowed)", nwt, AC_ADDR_MAXORB);
    ci->nwt = nwt;
    for (int s = 1; s <= nwt; s++)
        ci->lorb[s] = lorb[s - 1];
    ac_csflist all;
    ac_csflist_init(&all);
    int cap = 0, *seqs = NULL; /* [csf][AC_CSF_MAXSH]: the sequence orbital of each slot */
    for (int c = 0; c < ncfg; c++) {
        int lo[AC_CSF_MAXSH], oc[AC_CSF_MAXSH], sq[AC_CSF_MAXSH], n = 0;
        for (int s = 0; s < nwt; s++)
            if (occ[(size_t)c * nwt + s]) {
                if (n == AC_CSF_MAXSH) {
                    ac_csflist_free(&all);
                    free(seqs);
                    return fail(err, errlen, "configuration %d: more than %d open orbitals", c,
                                AC_CSF_MAXSH);
                }
                lo[n] = lorb[s], oc[n] = occ[(size_t)c * nwt + s], sq[n++] = s + 1;
            }
        int before = all.len;
        if (n == 0 || ac_csfgen(t, n, lo, oc, &all) < 0) {
            ac_csflist_free(&all);
            free(seqs);
            return fail(err, errlen, "cannot generate the CSFs of configuration %d", c);
        }
        if (all.len > cap) {
            cap = 2 * all.len + 16;
            int *ns = realloc(seqs, (size_t)cap * AC_CSF_MAXSH * sizeof *seqs);
            if (!ns) {
                ac_csflist_free(&all);
                free(seqs);
                return fail(err, errlen, "out of memory");
            }
            seqs = ns;
        }
        for (int i = before; i < all.len; i++)
            memcpy(seqs + (size_t)i * AC_CSF_MAXSH, sq, sizeof sq);
    }
    ci->csf = calloc((size_t)(all.len ? all.len : 1), sizeof *ci->csf);
    if (!ci->csf) {
        ac_csflist_free(&all);
        free(seqs);
        return fail(err, errlen, "out of memory");
    }
    for (int i = 0; i < all.len; i++) {
        ci->csf[i].c = all.v[i];
        for (int s = 1; s <= all.v[i].nsh; s++)
            ci->csf[i].seq[s] = seqs[(size_t)i * AC_CSF_MAXSH + s - 1];
    }
    ci->ncsf = all.len;
    ac_csflist_free(&all);
    free(seqs);
    return 0;
}

/* Theta cache: the coefficients of the Breit-Pauli two-body terms depend
   only on the orbitals (a, b, c, d), k and g, not on the CSF pair. Open
   addressing on a 64-bit key (0 = empty). */
typedef struct {
    unsigned long long key;
    double v;
} thc_ent;

typedef struct {
    thc_ent *e;
    size_t cap, n;
} thc_tab;

enum { THC_SO2 = 1, THC_SS, THC_OO, THC_GC, /* THC_GC + kp + 1, kp = -1..2 */
       THC_OZ1 = THC_GC + 4, THC_OZ2, THC_REC, THC_M1S0, THC_M1S2, THC_SMS };

static unsigned long long thc_key(int kind, int a, int b, int c, int d, int k, int g)
{
    return ((((((unsigned long long)kind * 256 + a) * 256 + b) * 256 + c) * 256 + d) * 64 + k) *
               64 + (g + 1) + 1;
}

static size_t thc_slot(const thc_tab *t, unsigned long long key)
{
    unsigned long long h = key * 0x9E3779B97F4A7C15ULL;
    size_t i = (size_t)(h >> 20) & (t->cap - 1);
    while (t->e[i].key && t->e[i].key != key)
        i = (i + 1) & (t->cap - 1);
    return i;
}

static int thc_get(ac_ci *ci, unsigned long long key, double *v)
{
    thc_tab *t = ci->thc;
    if (!t)
        return 0;
    size_t i = thc_slot(t, key);
    if (!t->e[i].key)
        return 0;
    *v = t->e[i].v;
    return 1;
}

static void thc_put(ac_ci *ci, unsigned long long key, double v)
{
    thc_tab *t = ci->thc;
    if (!t) {
        t = ci->thc = calloc(1, sizeof *t);
        if (!t)
            return;
    }
    if (2 * (t->n + 1) > t->cap) {
        thc_tab nt = {calloc(t->cap ? 2 * t->cap : 4096, sizeof(thc_ent)),
                      t->cap ? 2 * t->cap : 4096, 0};
        if (!nt.e)
            return;
        for (size_t i = 0; i < t->cap; i++)
            if (t->e[i].key) {
                nt.e[thc_slot(&nt, t->e[i].key)] = t->e[i];
                nt.n++;
            }
        free(t->e);
        *t = nt;
    }
    size_t i = thc_slot(t, key);
    if (!t->e[i].key) {
        t->e[i].key = key;
        t->n++;
    }
    t->e[i].v = v;
}

void ac_ci_free_cache(ac_ci *ci)
{
    if (ci->thc) {
        free(((thc_tab *)ci->thc)->e);
        free(ci->thc);
    }
    ci->thc = NULL;
}

void ac_ci_free(ac_ci *ci)
{
    free(ci->csf);
    ac_ci_free_cache(ci);
    memset(ci, 0, sizeof *ci);
}

/* the two CSFs on the union of their slots */
static int pair_csfs(const ac_ci *ci, const ac_ci_csf *F, const ac_ci_csf *G, ac_csf *bf,
                     ac_csf *bg, int *seq)
{
    return ac_pair_union(ci->t, F, G, bf, bg, seq);
}

/* slots other than those in `used` must carry the same shell state */
static int spectators_match(const ac_csf *bf, const ac_csf *bg, int a, int b, int c, int d)
{
    return ac_pair_spectators_match(bf, bg, a, b, c, d);
}

/* ac_ci_hij: the expression of hexpr-atomci with the integrals put in. A zero
   R^k leaves its term out before the angular part is computed. */
typedef struct {
    const ac_ci_integrals *in;
    double rk, sum;
} hij_sum;

static int hij_skip(void *p, int k, int a, int b, int c, int d)
{
    hij_sum *x = p;
    x->rk = x->in->rk(x->in->ctx, k, a, b, c, d);
    return x->rk == 0.0;
}

static void hij_add(void *p, int k, int a, int b, int c, int d, double coef)
{
    hij_sum *x = p;
    (void)c, (void)d;
    x->sum += coef * (k < 0 ? x->in->h(x->in->ctx, a, b) : x->rk);
}

double ac_ci_hij(ac_ci *ci, int i, int j, int *err)
{
    hij_sum x = {&ci->in, 0.0, 0.0};
    *err = ac_pair_terms(ci->t, &ci->csf[i], &ci->csf[j], hij_skip, hij_add, &x,
                         &ci->ops_evaluated);
    return *err ? 0.0 : x.sum;
}

static int tri2(int a, int b, int c)
{
    return (a + b + c) % 2 == 0 && c <= a + b && c >= abs(a - b);
}

double ac_ci_one_coef(double spin_reduced, double orbital_reduced, int ks, int kl)
{
    return spin_reduced * orbital_reduced / -sqrt((double)(ks + 1) * (kl + 1));
}

typedef struct {
    double (*f)(void *ctx, int a, int b, int l);
    void *ctx;
} one_eq;

static double one_eq_f(void *p, int a, int b, int la, int lb)
{
    const one_eq *o = p;
    return la == lb ? o->f(o->ctx, a, b, la) : 0.0;
}

double ac_ci_one(ac_ci *ci, int i, int j, int ks, int kl,
                 double (*f)(void *ctx, int a, int b, int l), void *ctx, int *err)
{
    one_eq o = {f, ctx};
    return ac_ci_one_ll(ci, i, j, ks, kl, one_eq_f, &o, err);
}

double ac_ci_one_ll(ac_ci *ci, int i, int j, int ks, int kl,
                    double (*f)(void *ctx, int a, int b, int la, int lb), void *ctx, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (!tri2(FT.ts, ks, GT.ts) || !tri2(FT.tl, kl, GT.tl))
        return 0.0;

    /* a one-body operator moves at most one electron between shells */
    int u = 0, v = 0, nu = 0, nv = 0, ndiff = 0;
    for (int s = 1; s <= n; s++) {
        int d = bf.nel[s] - bg.nel[s];
        ndiff += abs(d);
        if (d == 1) u = s, nu++;
        else if (d == -1) v = s, nv++;
    }
    if (ndiff > 2 || (ndiff == 2 && (nu != 1 || nv != 1)))
        return 0.0;

    double sum = 0.0;
    for (int a = 1; a <= n; a++)
        for (int b = 1; b <= n; b++) {
            if (ndiff == 0 ? a != b : !(a == u && b == v))
                continue;
            int l = bf.l[a], lb = bf.l[b];
            if (bg.nel[b] == 0 || !spectators_match(&bf, &bg, a, b, 0, 0))
                continue;
            if (kl > 2 * (l + lb) || kl < 2 * abs(l - lb))
                continue;
            /* a closed shell carries only the rank-(0,0) density */
            if (a == b && bg.nel[a] == 4 * l + 2 && (ks || kl))
                continue;
            double z = f(ctx, seq[a], seq[b], l, lb);
            if (z == 0.0)
                continue;
            ac_otree tr;
            ac_lab p = {ks, kl};
            ac_ot_one_body(&tr, a, l, b, lb, p);
            double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 0, err);
            ci->ops_evaluated++;
            if (*err)
                return 0.0;
            sum += z * inv;
        }
    return (ks / 2 + kl / 2) % 2 ? -sum : sum;
}

/* <l||C^k||l'> (Racah (1942)): (-1)^l sqrt([l][l']) (l k l'; 0 0 0) */
static double cred(int l, int k, int lp)
{
    if ((l + k + lp) % 2 || k < abs(l - lp) || k > l + lp)
        return 0.0;
    int g = (l + k + lp) / 2;
    return ((l + g) % 2 ? -1.0 : 1.0) * sqrt((2.0 * l + 1) * (2.0 * lp + 1)) * ac_fabc(l, k, lp);
}

int ac_so2_parts = 3;
int ac_atsp_compat = 0;

/* CSFs i and j differ by a double excitation ab -> cd over four distinct
   orbitals (ATSP's bpci has no two-body Breit-Pauli elements there) */
static int four_shell_double(const ac_ci *ci, int i, int j)
{
    int occ[AC_ADDR_MAXORB + 1] = {0}, nrem = 0, nadd = 0;
    for (int s = 1; s <= ci->csf[i].c.nsh; s++)
        occ[ci->csf[i].seq[s]] += ci->csf[i].c.nel[s];
    for (int s = 1; s <= ci->csf[j].c.nsh; s++)
        occ[ci->csf[j].seq[s]] -= ci->csf[j].c.nel[s];
    for (int o = 0; o <= AC_ADDR_MAXORB; o++) {
        nrem += occ[o] > 0;
        nadd += occ[o] < 0;
    }
    return nrem == 2 && nadd == 2;
}

/* Theta x invariant of :((b_a b~_b)^(s1 g) (b_c b~_d)^(s2 k))^(1 1): for both
   spin assignments: (1, 0) spin-same-orbit, (0, 1) spin-other-orbit (x 2) */
static double so2_term(ac_ci *ci, const ac_csf *bf, const ac_csf *bg, int a, int b, int c,
                       int d, int g, int k, double val, int *err)
{
    double sum = 0.0;
    for (int s1 = 0; s1 <= 1; s1++) {
        int s2 = 1 - s1;
        if (!(ac_so2_parts & (s1 ? 1 : 2)))
            continue;
        ac_otree tr;
        ac_lab p = {2 * s1, 2 * g}, q = {2 * s2, 2 * k}, tot = {2, 2};
        ac_ot_two_body(&tr, a, bf->l[a], b, bf->l[b], p, c, bf->l[c], d, bf->l[d], q, tot);
        double inv = ac_factor_op(ci->t, bf, bg, &tr, 1, err);
        ci->ops_evaluated++;
        if (*err)
            return 0.0;
        sum += (s2 ? 2.0 : 1.0) * val * inv;
    }
    return sum;
}

/* H_SO2 (Bethe-Salpeter (1957), section 39) with its Theta from the
   technical note, Sec. 4.2 */
double ac_ci_so2(ac_ci *ci, int i, int j, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    const ac_ci_integrals *in = &ci->in;
    if (!in->rkg || (ac_atsp_compat && four_shell_double(ci, i, j)))
        return 0.0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (!tri2(FT.ts, 2, GT.ts) || !tri2(FT.tl, 2, GT.tl))
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    double sum = 0.0, a2 = ac_alpha * ac_alpha;
    void *zx = in->zctx;
    /* ordered (a, b, c, d): b_a, b_c create, b~_b, b~_d annihilate */
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                if (!spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                for (int k = 0; k <= lc + ld && k <= la + lb + 2; k++) {
                    double cc = cred(lc, k, ld);
                    if (cc == 0.0)
                        continue;
                    /* the gradient radial integrals of the four (lam_a, lam_b), on demand */
                    double rg[2][2];
                    int have_rg = 0;
                    for (int g = k - 1; g <= k + 1; g++) {
                        if (g < 0 || g < abs(la - lb) || g > la + lb)
                            continue;
                        unsigned long long key =
                            thc_key(THC_SO2, seq[a], seq[b], seq[c], seq[d], k, g);
                        double theta;
                        if (!thc_get(ci, key, &theta)) {
                            if (!have_rg) {
                                for (int ia = 0; ia < 2; ia++)
                                    for (int ib = 0; ib < 2; ib++) {
                                        int lama = la + (ia ? 1 : -1), lamb = lb + (ib ? 1 : -1);
                                        rg[ia][ib] =
                                            lama < 0 || lamb < 0 || cred(lama, k, lamb) == 0.0
                                                ? 0.0
                                                : in->rkg(zx, k, seq[a], ia ? 1 : -1, seq[b],
                                                          ib ? 1 : -1, seq[c], seq[d]);
                                    }
                                have_rg = 1;
                            }
                            double u = 0.0;
                            for (int ia = 0; ia < 2; ia++)
                                for (int ib = 0; ib < 2; ib++) {
                                    int lama = la + (ia ? 1 : -1), lamb = lb + (ib ? 1 : -1);
                                    if (rg[ia][ib] == 0.0)
                                        continue;
                                    u += sqrt(6.0 * (2 * la + 1) * (2 * lb + 1) * (2 * g + 1)) *
                                         ac_ninej(2 * lama, 2 * lamb, 2 * k, 2, 2, 2, 2 * la,
                                                  2 * lb, 2 * g) *
                                         cred(lama, k, lamb) * rg[ia][ib];
                                }
                            theta = 0.5 * a2 * (k % 2 ? -1.0 : 1.0) / sqrt(2.0 * k + 1) * cc * u;
                            thc_put(ci, key, theta);
                        }
                        if (theta == 0.0)
                            continue;
                        sum += so2_term(ci, &bf, &bg, a, b, c, d, g, k, theta, err);
                        if (*err)
                            return 0.0;
                    }
                }
            }
        }
    return sum;
}

/* H_SS (Bethe-Salpeter (1957), section 39); [grad x grad]^(2) by Edmonds
   (1957), Eq. (7.1.1); Theta from the technical note, Sec. 4.2 */
double ac_ci_ss(ac_ci *ci, int i, int j, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    const ac_ci_integrals *in = &ci->in;
    if (!in->rss || (ac_atsp_compat && four_shell_double(ci, i, j)))
        return 0.0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (!tri2(FT.ts, 4, GT.ts) || !tri2(FT.tl, 4, GT.tl))
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    double sum = 0.0, a2 = ac_alpha * ac_alpha;
    void *zx = in->zctx;
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                if (!spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                for (int k = 0; k <= lc + ld; k++) {
                    double cc = cred(lc, k, ld);
                    if (cc == 0.0)
                        continue;
                    for (int lam = k - 2; lam <= k + 2; lam += 2) {
                        if (lam < 0 || lam < abs(k - 2))
                            continue;
                        double ca = cred(la, lam, lb);
                        if (ca == 0.0)
                            continue;
                        unsigned long long key =
                            thc_key(THC_SS, seq[a], seq[b], seq[c], seq[d], k, lam);
                        double theta;
                        if (!thc_get(ci, key, &theta)) {
                            double rad = in->rss(zx, k, lam, seq[a], seq[b], seq[c], seq[d]);
                            theta = -0.25 * a2 * (k % 2 ? -1.0 : 1.0) / (2 * k + 1) * ca * cc * rad;
                            thc_put(ci, key, theta);
                        }
                        if (theta == 0.0)
                            continue;
                        ac_otree tr;
                        ac_lab p = {2, 2 * lam}, q = {2, 2 * k}, tot = {4, 4};
                        ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                        double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                        ci->ops_evaluated++;
                        if (*err)
                            return 0.0;
                        sum += theta * inv;
                    }
                }
            }
        }
    return sum;
}

/* H_OO (Bethe-Salpeter (1957), section 39) in the current/divergence form
   (technical note, Sec. 4.2), Edmonds (1957), Eq. (7.1.1) */
double ac_ci_oo(ac_ci *ci, int i, int j, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    const ac_ci_integrals *in = &ci->in;
    if (!in->roo)
        return 0.0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (FT.ts != GT.ts || FT.tl != GT.tl)
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    double sum = 0.0, a2 = ac_alpha * ac_alpha;
    void *zx = in->zctx;
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                if (!spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                for (int k = 0; k <= la + lb + 1 && k <= lc + ld + 1; k++) {
                    for (int g = k - 1; g <= k + 1; g++) {
                        if (g < 0)
                            continue;
                        unsigned long long key =
                            thc_key(THC_OO, seq[a], seq[b], seq[c], seq[d], k, g);
                        double theta = 0.0;
                        if (thc_get(ci, key, &theta))
                            goto oo_have;
                        if ((la + lb + k + 1) % 2 == 0 && (lc + ld + k + 1) % 2 == 0 &&
                            tri2(2 * la, 2 * g, 2 * lb) && tri2(2 * lc, 2 * g, 2 * ld)) {
                            double nj = ac_ninej(2 * k, 2 * k, 0, 2, 2, 0, 2 * g, 2 * g, 0);
                            if (nj != 0.0) {
                                double rad = in->roo(zx, k, g, seq[a], seq[b], seq[c], seq[d]);
                                theta = a2 * (k % 2 ? -1.0 : 1.0) * sqrt(2.0 * k + 1) *
                                        -sqrt(3.0) * nj * rad;
                            }
                        }
                        if (g == k) {
                            double cc = cred(la, k, lb) * cred(lc, k, ld);
                            if (cc != 0.0)
                                theta += 0.5 * a2 * (k % 2 ? -1.0 : 1.0) * cc *
                                         in->roo(zx, k, -1, seq[a], seq[b], seq[c], seq[d]) /
                                         sqrt(2.0 * k + 1);
                        }
                        thc_put(ci, key, theta);
                    oo_have:
                        if (theta == 0.0)
                            continue;
                        ac_otree tr;
                        ac_lab p = {0, 2 * g}, tot = {0, 0};
                        ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, p, tot);
                        double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                        ci->ops_evaluated++;
                        if (*err)
                            return 0.0;
                        sum += theta * inv;
                    }
                }
            }
        }
    return sum / sqrt((double)(FT.ts + 1) * (FT.tl + 1));
}

int ac_gc2e_parts = 3;

/* the two-electron gauge correction of g_J: the field-linear part of H3,
   H4 of Hegstrom (1973), Eq. (21) (spin-other-orbit twice spin-same-orbit;
   cf. Pachucki (2004), Eq. (32)); derivation in the technical note,
   Sec. 5.1 */
double ac_ci_gc2e(ac_ci *ci, int i, int j, int kp, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    const ac_ci_integrals *in = &ci->in;
    if (!in->rgc || kp < 0 || kp > 2)
        return 0.0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (!tri2(FT.ts, 2, GT.ts) || !tri2(FT.tl, 2 * kp, GT.tl))
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    /* c'_K' (spin first) of the [R x grad]^(K') terms; the density term
       (K' = 0) has -2 */
    static const double cp[3] = {1.1547005383792515, 1.0, -1.2909944487358056};
    double sum = 0.0, a2 = ac_alpha * ac_alpha;
    void *zx = in->zctx;
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                if (!spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                for (int k = 0; k <= lc + ld; k++) {
                    double cc = cred(lc, k, ld);
                    if (cc == 0.0 || (la + lb + k) % 2)
                        continue;
                    /* g = k - kp .. k + kp; the density term only for kp = 0 */
                    for (int g = abs(k - kp); g <= k + kp; g++) {
                        if (g < abs(la - lb) || g > la + lb)
                            continue;
                        unsigned long long key =
                            thc_key(THC_GC + kp + 1, seq[a], seq[b], seq[c], seq[d], k, g);
                        double theta;
                        if (!thc_get(ci, key, &theta)) {
                            double rad = in->rgc(zx, k, kp, g, seq[a], seq[b], seq[c], seq[d]);
                            theta = 0.5 * a2 * cp[kp] * (k % 2 ? -1.0 : 1.0) * cc * rad /
                                    sqrt((2.0 * kp + 1) * (2 * k + 1));
                            if (kp == 0 && g == k) {
                                double cd = cred(la, k, lb);
                                if (cd != 0.0)
                                    theta += 0.5 * a2 * -2.0 * (k % 2 ? -1.0 : 1.0) * cc * cd *
                                             in->rgc(zx, k, -1, k, seq[a], seq[b], seq[c],
                                                     seq[d]) /
                                             sqrt(2.0 * k + 1);
                            }
                            thc_put(ci, key, theta);
                        }
                        if (theta == 0.0)
                            continue;
                        for (int s1 = 0; s1 <= 1; s1++) {
                            if (!(ac_gc2e_parts & (s1 ? 1 : 2)))
                                continue;
                            ac_otree tr;
                            ac_lab p = {2 * s1, 2 * g}, q = {2 * (1 - s1), 2 * k},
                                   tot = {2, 2 * kp};
                            ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                            double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                            ci->ops_evaluated++;
                            if (*err)
                                return 0.0;
                            sum += (s1 ? 1.0 : 2.0) * theta * inv;
                        }
                    }
                }
            }
        }
    /* Edmonds = (-1)^(ks + kl) x invariant */
    return (1 + kp) % 2 ? -sum : sum;
}

/* Pachucki (2004), Eqs. (60), (62): the spin two-electron part of M1 */
double ac_ci_m1s2(ac_ci *ci, int i, int j, int K, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    const ac_ci_integrals *in = &ci->in;
    if (!in->rm1 || (K != 0 && K != 2))
        return 0.0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (!tri2(FT.ts, 2, GT.ts) || !tri2(FT.tl, 2 * K, GT.tl))
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    double sum = 0.0, a2 = ac_alpha * ac_alpha;
    void *zx = in->zctx;
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                if (!spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                for (int k = 0; k <= lc + ld; k++) {
                    double cc = cred(lc, k, ld);
                    if (cc == 0.0)
                        continue;
                    for (int lam = K ? k - 2 : k; lam <= (K ? k + 2 : k); lam += 2) {
                        if (lam < 0 || (K && lam < abs(k - 2)))
                            continue;
                        double ca = cred(la, lam, lb);
                        if (ca == 0.0)
                            continue;
                        unsigned long long key = thc_key(K ? THC_M1S2 : THC_M1S0, seq[a], seq[b],
                                                         seq[c], seq[d], k, lam);
                        double theta;
                        if (!thc_get(ci, key, &theta)) {
                            double ph = k % 2 ? -1.0 : 1.0;
                            theta = K ? a2 * ph * ca * cc *
                                            in->rm1(zx, k, lam, seq[a], seq[b], seq[c], seq[d]) /
                                            (sqrt(3.0) * (2 * k + 1))
                                      : -2.0 / 3.0 * a2 * ph * ca * cc *
                                            in->rm1(zx, k, -1, seq[a], seq[b], seq[c], seq[d]) /
                                            sqrt(2.0 * k + 1);
                            thc_put(ci, key, theta);
                        }
                        if (theta == 0.0)
                            continue;
                        ac_otree tr;
                        ac_lab p = {2, 2 * lam}, q = {0, 2 * k}, tot = {2, 2 * K};
                        ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                        double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                        ci->ops_evaluated++;
                        if (*err)
                            return 0.0;
                        sum += theta * inv;
                    }
                }
            }
        }
    /* Edmonds = (-1)^(ks + kl) x invariant, (1, K) */
    return -sum;
}

/* <l_a||[C^k x C^1]^(g)||l_b> (Edmonds 7.1.1) */
static double cc1_red(int la, int lb, int k, int g)
{
    double s = 0.0;
    for (int lam = abs(la - k); lam <= la + k; lam++) {
        double c = cred(la, k, lam) * cred(lam, 1, lb);
        if (c != 0.0)
            s += ac_sixj(2 * k, 2, 2 * g, 2 * lb, 2 * la, 2 * lam) * c;
    }
    return ((la + g + lb) % 2 ? -1.0 : 1.0) * sqrt(2.0 * g + 1) * s;
}

/* the orbit-orbit Zeeman term: the field-linear part of H6 of Hegstrom
   (1973), Eq. (21) (cf. Pachucki (2004), Eq. (32), H5); derivation in
   the technical note, Sec. 5.1 */
double ac_ci_oz2e(ac_ci *ci, int i, int j, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    const ac_ci_integrals *in = &ci->in;
    if (!in->roz)
        return 0.0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (FT.ts != GT.ts || !tri2(FT.tl, 2, GT.tl))
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    double sum = 0.0, a2 = ac_alpha * ac_alpha;
    void *zx = in->zctx;
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                if (!spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                for (int k = 0; k <= lc + ld + 1; k++) {
                    /* term 1: [C^k x R]^(g1)(1) x [C^k x J]^(g2)(2), both of parity (-1)^(k+1) */
                    if ((la + lb + k + 1) % 2 == 0 && (lc + ld + k + 1) % 2 == 0)
                        for (int g1 = abs(k - 1); g1 <= k + 1; g1++)
                            for (int g2 = abs(k - 1); g2 <= k + 1; g2++) {
                                if (!tri2(2 * la, 2 * g1, 2 * lb) || !tri2(2 * lc, 2 * g2, 2 * ld) ||
                                    !tri2(2 * g1, 2 * g2, 2))
                                    continue;
                                unsigned long long key = thc_key(THC_OZ1, seq[a], seq[b], seq[c],
                                                                 seq[d], k, 8 * g1 + g2);
                                double theta;
                                if (!thc_get(ci, key, &theta)) {
                                    double nj = ac_ninej(2 * k, 2 * k, 0, 2, 2, 2, 2 * g1, 2 * g2, 2);
                                    double a1 = nj == 0.0 ? 0.0 : cc1_red(la, lb, k, g1);
                                    theta = a1 == 0.0 ? 0.0
                                                      : 2.0 * sqrt(6.0) * a2 * (k % 2 ? -1.0 : 1.0) *
                                                            sqrt(2.0 * k + 1) * nj * a1 *
                                                            in->roz(zx, k, g2, 0, seq[a], seq[b],
                                                                    seq[c], seq[d]);
                                    thc_put(ci, key, theta);
                                }
                                if (theta == 0.0)
                                    continue;
                                ac_otree tr;
                                ac_lab p = {0, 2 * g1}, q = {0, 2 * g2}, tot = {0, 2};
                                ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                                double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                                ci->ops_evaluated++;
                                if (*err)
                                    return 0.0;
                                sum += theta * inv;
                            }
                    /* term 2: [C^k x [R x grad]^(1)]^(g) f (1) x C^k Q (2), parity (-1)^k */
                    double cc = cred(lc, k, ld);
                    if (cc == 0.0 || (la + lb + k) % 2)
                        continue;
                    for (int g = abs(k - 1); g <= k + 1; g++) {
                        if (!tri2(2 * la, 2 * g, 2 * lb))
                            continue;
                        unsigned long long key =
                            thc_key(THC_OZ2, seq[a], seq[b], seq[c], seq[d], k, g);
                        double theta;
                        if (!thc_get(ci, key, &theta)) {
                            theta = a2 * sqrt(2.0 / 3.0) * (k % 2 ? -1.0 : 1.0) * cc *
                                    in->roz(zx, k, g, 1, seq[a], seq[b], seq[c], seq[d]) /
                                    sqrt(2.0 * k + 1);
                            thc_put(ci, key, theta);
                        }
                        if (theta == 0.0)
                            continue;
                        ac_otree tr;
                        ac_lab p = {0, 2 * g}, q = {0, 2 * k}, tot = {0, 2};
                        ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                        double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                        ci->ops_evaluated++;
                        if (*err)
                            return 0.0;
                        sum += theta * inv;
                    }
                }
            }
        }
    /* Edmonds = (-1)^(0 + 1) x invariant */
    return -sum;
}

/* the two-body recoil term sum_{i != j} r_i x p_j of g_L (Phillips (1949),
   T_H1; Hegstrom (1973), Eq. (23); Beltran-Lopez (1968), Eq. (3);
   technical note, Sec. 5.1) */
double ac_ci_rec2(ac_ci *ci, int i, int j, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    const ac_ci_integrals *in = &ci->in;
    if (!in->rrc)
        return 0.0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (FT.ts != GT.ts || !tri2(FT.tl, 2, GT.tl))
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    double sum = 0.0;
    void *zx = in->zctx;
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                double ca = cred(la, 1, lb), cc = cred(lc, 1, ld);
                if (ca == 0.0 || cc == 0.0 || !spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                double theta;
                unsigned long long key = thc_key(THC_REC, seq[a], seq[b], seq[c], seq[d], 0, 0);
                if (!thc_get(ci, key, &theta)) {
                    double grad = cc * in->rrc(zx, lc == ld + 1 ? 1 : 2, seq[c], seq[d]);
                    theta = -sqrt(2.0) * (2.0 / 3.0) * ca * in->rrc(zx, 0, seq[a], seq[b]) * grad;
                    thc_put(ci, key, theta);
                }
                if (theta == 0.0)
                    continue;
                ac_otree tr;
                ac_lab p = {0, 2}, q = {0, 2}, tot = {0, 2};
                ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                ci->ops_evaluated++;
                if (*err)
                    return 0.0;
                sum += theta * inv;
            }
        }
    /* Edmonds = (-1)^(0 + 1) x invariant */
    return -sum;
}

/* the specific mass shift sum_{i<j} p_i.p_j (technical note, Sec. 5.5) */
double ac_ci_sms(ac_ci *ci, int i, int j, double (*grad)(void *, int, int, int), void *ctx,
                 int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    *err = 0;
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return 0.0;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (FT.ts != GT.ts || FT.tl != GT.tl)
        return 0.0;
    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0.0;

    double sum = 0.0;
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0)
                    bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                double ca = cred(la, 1, lb), cc = cred(lc, 1, ld);
                if (ca == 0.0 || cc == 0.0 || !spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                double theta;
                unsigned long long key = thc_key(THC_SMS, seq[a], seq[b], seq[c], seq[d], 0, 0);
                if (!thc_get(ci, key, &theta)) {
                    theta = ca * grad(ctx, la == lb + 1 ? 1 : 2, seq[a], seq[b]) * cc *
                            grad(ctx, lc == ld + 1 ? 1 : 2, seq[c], seq[d]) / sqrt(3.0);
                    thc_put(ci, key, theta);
                }
                if (theta == 0.0)
                    continue;
                ac_otree tr;
                ac_lab p = {0, 2}, q = {0, 2}, tot = {0, 0};
                ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 1, err);
                ci->ops_evaluated++;
                if (*err)
                    return 0.0;
                sum += theta * inv;
            }
        }
    /* Edmonds = (-1)^(0 + 0) x invariant */
    return sum;
}

static double so1_f(void *ctx, int a, int b, int l)
{
    ac_ci *ci = ctx;
    if (l == 0)
        return 0.0;
    return ci->in.zeta(ci->in.zctx, a, b) * sqrt(l * (l + 1) * (2 * l + 1) / 2.0);
}

/* H_SO1 (Bethe-Salpeter (1957), section 39) as a double tensor (technical
   note, Sec. 4.2) */
double ac_ci_so1(ac_ci *ci, int i, int j, int *err)
{
    *err = 0;
    if (!ci->in.zeta)
        return 0.0;
    return ac_ci_one(ci, i, j, 2, 2, so1_f, ci, err);
}

/* LAPACK */
void dsyev_(const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
            double *w, double *work, const int *lwork, int *info);

static int syev(const char *jobz, int n, double *H, double *w)
{
    int info = 0, lwork = -1;
    double q;
    dsyev_(jobz, "L", &n, H, &n, w, &q, &lwork, &info);
    if (info)
        return info;
    lwork = (int)q;
    double *work = malloc((size_t)lwork * sizeof *work);
    if (!work)
        return -1;
    dsyev_(jobz, "L", &n, H, &n, w, work, &lwork, &info);
    free(work);
    return info;
}

int ac_ci_eigvals(int n, double *H, double *w)
{
    return syev("N", n, H, w);
}

int ac_ci_eigvecs(int n, double *H, double *w)
{
    return syev("V", n, H, w);
}

/* AV = H V for m columns; H symmetric, so row r is the contiguous column r */
static void hmul(int n, const double *H, int m, const double *V, double *AV)
{
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n; r++) {
        const double *h = H + (size_t)r * n;
        for (int j = 0; j < m; j++) {
            const double *v = V + (size_t)j * n;
            double s = 0.0;
            for (int i = 0; i < n; i++)
                s += h[i] * v[i];
            AV[(size_t)j * n + r] = s;
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

typedef struct {
    double d;
    int i;
} diag_t;

static int by_diag(const void *a, const void *b)
{
    double d = ((const diag_t *)a)->d - ((const diag_t *)b)->d;
    return (d > 0) - (d < 0);
}

/* Davidson (1975) in the block form of Liu (1978) */
int ac_ci_davidson(int n, const double *H, int k, double tol, double *w, double *x, int *niter)
{
    if (k < 1 || k > n)
        return -1;
    int mmax = 4 * k > 2 * k + 48 ? 4 * k : 2 * k + 48, m0 = 2 * k > 200 ? 2 * k : 200;
    if (mmax > n)
        mmax = n;
    if (m0 > n)
        m0 = n;
    size_t N = (size_t)n;
    double *V = malloc(N * mmax * sizeof *V), *AV = malloc(N * mmax * sizeof *AV);
    double *X = malloc(N * k * sizeof *X), *AX = malloc(N * k * sizeof *AX);
    double *G = malloc((size_t)mmax * mmax * sizeof *G);
    double *th = malloc((size_t)(mmax > m0 ? mmax : m0) * sizeof *th);
    double *B = malloc((size_t)m0 * m0 * sizeof *B), *R = malloc(N * k * sizeof *R);
    diag_t *dg = malloc(N * sizeof *dg);
    int rc = -1, it = 0;
    if (!V || !AV || !X || !AX || !G || !th || !B || !R || !dg)
        goto out;
    /* start: the lowest eigenvectors of the block of the m0 lowest diagonal elements */
    for (int i = 0; i < n; i++)
        dg[i].d = H[N * i + i], dg[i].i = i;
    qsort(dg, N, sizeof *dg, by_diag);
    for (int b = 0; b < m0; b++)
        for (int a = 0; a < m0; a++)
            B[(size_t)b * m0 + a] = H[N * dg[b].i + dg[a].i];
    if ((rc = syev("V", m0, B, th)) != 0)
        goto out;
    memset(V, 0, N * k * sizeof *V);
    for (int j = 0; j < k; j++)
        for (int a = 0; a < m0; a++)
            V[N * j + dg[a].i] = B[(size_t)j * m0 + a];
    hmul(n, H, k, V, AV);
    int m = k;
    rc = 1;
    for (it = 1; it <= 1000; it++) {
        /* Rayleigh-Ritz in span V */
        for (int b = 0; b < m; b++)
            for (int a = 0; a <= b; a++)
                G[(size_t)b * m + a] = G[(size_t)a * m + b] =
                    0.5 * (dot(n, V + N * a, AV + N * b) + dot(n, V + N * b, AV + N * a));
        if (syev("V", m, G, th)) {
            rc = -1;
            goto out;
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) {
                double s = 0.0, t = 0.0;
                for (int a = 0; a < m; a++) {
                    s += V[N * a + i] * G[(size_t)j * m + a];
                    t += AV[N * a + i] * G[(size_t)j * m + a];
                }
                X[N * j + i] = s, AX[N * j + i] = t;
                R[N * j + i] = t - th[j] * s;
            }
        int nnew = 0, conv = 1;
        double rmax = 0.0;
        for (int j = 0; j < k; j++) {
            double rn = sqrt(dot(n, R + N * j, R + N * j));
            rmax = rn > rmax ? rn : rmax;
            if (rn >= tol)
                conv = 0;
        }
        if (getenv("AC_DAVDEBUG"))
            fprintf(stderr, "davidson %d: m %d, w0 %.15f, |r| %.2e\n", it, m, th[0], rmax);
        if (conv) {
            rc = 0;
            break;
        }
        if (m + k > mmax) { /* restart from the Ritz vectors */
            memcpy(V, X, N * k * sizeof *V);
            memcpy(AV, AX, N * k * sizeof *AV);
            m = k;
        }
        /* corrections (th - D)^-1 r of the unconverged roots, orthonormalized */
        for (int j = 0; j < k; j++) {
            double *r = R + N * j, *t = V + N * (m + nnew);
            if (sqrt(dot(n, r, r)) < tol)
                continue;
            for (int i = 0; i < n; i++) {
                double d = th[j] - H[N * i + i];
                t[i] = r[i] / (fabs(d) > 1e-8 ? d : (d < 0 ? -1e-8 : 1e-8));
            }
            double nt = sqrt(dot(n, t, t));
            for (int i = 0; i < n; i++)
                t[i] /= nt;
            for (int pass = 0; pass < 2; pass++)
                for (int a = 0; a < m + nnew; a++) {
                    double c = dot(n, V + N * a, t);
                    for (int i = 0; i < n; i++)
                        t[i] -= c * V[N * a + i];
                }
            nt = sqrt(dot(n, t, t));
            if (nt < 1e-8) /* (nearly) in span V already */
                continue;
            for (int i = 0; i < n; i++)
                t[i] /= nt;
            nnew++;
        }
        if (nnew == 0)
            break; /* stalled */
        hmul(n, H, nnew, V + N * m, AV + N * m);
        m += nnew;
    }
    for (int j = 0; j < k; j++)
        w[j] = th[j];
    memcpy(x, X, N * k * sizeof *x);
out:
    if (niter)
        *niter = it;
    free(V);
    free(AV);
    free(X);
    free(AX);
    free(G);
    free(th);
    free(B);
    free(R);
    free(dg);
    return rc;
}
