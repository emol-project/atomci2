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
#include <omp.h>
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

size_t ac_ci_cache_bytes(const ac_ci *ci, size_t *n)
{
    const thc_tab *t = ci->thc;
    if (n)
        *n = t ? t->n : 0;
    return t ? t->cap * sizeof(thc_ent) : 0;
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

static int tri2(int a, int b, int c);

/* nonzero when some pair (i0 + i, j0 + j) has total labels that an operator
   of the ranks ks/2 (spin) and kl/2 (orbital) can connect; a block without
   such a pair has no elements and is not set up */
static int any_total_pair(const ac_ci *ci, int i0, int ni, int j0, int nj, int ks, int kl)
{
    for (int i = i0; i < i0 + ni; i++) {
        ac_lab a = ci->csf[i].c.cpl[ci->csf[i].c.nsh];
        for (int j = j0; j < j0 + nj; j++) {
            ac_lab b = ci->csf[j].c.cpl[ci->csf[j].c.nsh];
            if (tri2(a.ts, ks, b.ts) && tri2(a.tl, kl, b.tl))
                return 1;
        }
    }
    return 0;
}

/* ac_ci_hij for all the CSF pairs of two configurations: the terms of
   ac_pair_terms_block, with the integrals put in, in the order of ac_ci_hij */
typedef struct {
    const ac_ci_integrals *in;
    double rk, *sum;
    size_t nn;
} hij_bsum;

static int hij_bskip(void *p, int k, int a, int b, int c, int d)
{
    hij_bsum *x = p;
    x->rk = x->in->rk(x->in->ctx, k, a, b, c, d);
    return x->rk == 0.0;
}

static void hij_badd(void *p, int k, int a, int b, int c, int d, const double *coef)
{
    hij_bsum *x = p;
    (void)c, (void)d;
    double val = k < 0 ? x->in->h(x->in->ctx, a, b) : x->rk;
    for (size_t pq = 0; pq < x->nn; pq++)
        if (coef[pq] != 0.0)
            x->sum[pq] += coef[pq] * val;
}

/* out[p * ng + q] = <F_p|H|G_q> for nf CSFs F of one configuration and ng
   CSFs G of one configuration (upper: only q >= p, the rest 0) */
static void hij_sub(ac_ci *ci, int nf, const ac_ci_csf *F, int ng, const ac_ci_csf *G, int upper,
                    double *out, int *err)
{
    ac_pair_block pb;
    hij_bsum x = {&ci->in, 0.0, out, (size_t)nf * (size_t)ng};
    for (size_t pq = 0; pq < x.nn; pq++)
        out[pq] = 0.0;
    *err = ac_pair_block_make(&pb, ci->t, nf, F, ng, G, upper);
    if (*err)
        return;
    *err = ac_pair_terms_block(ci->t, &pb, hij_bskip, hij_badd, &x, &ci->ops_evaluated);
    ac_pair_block_free(&pb);
}

int ac_ci_hij_list(ac_ci *ci, int nf, const int *fi, int ng, const int *gj, double *out, int *err)
{
    ac_ci_csf *F = malloc((size_t)(nf > 0 ? nf : 1) * sizeof *F);
    ac_ci_csf *G = malloc((size_t)(ng > 0 ? ng : 1) * sizeof *G);
    *err = 0;
    if (!F || !G)
        *err = 1;
    else {
        for (int p = 0; p < nf; p++)
            F[p] = ci->csf[fi[p]];
        for (int q = 0; q < ng; q++)
            G[q] = ci->csf[gj[q]];
        hij_sub(ci, nf, F, ng, G, 0, out, err);
    }
    free(F);
    free(G);
    return *err ? -1 : 0;
}

/* H connects only CSFs of the same total (S, L), and a run of one
   configuration holds several: the CSFs of each (S, L) are taken out of the
   two runs and their pairs evaluated together, so that no work goes into
   pairs without an element. Within a pair the terms come in the order of
   ac_ci_hij. */
int ac_ci_hij_block(ac_ci *ci, int i0, int ni, int j0, int nj, int upper, double *out, int *err)
{
    size_t nn = (size_t)ni * (size_t)nj;
    for (size_t pq = 0; pq < nn; pq++)
        out[pq] = 0.0;
    *err = 0;
    if (ni < 1 || nj < 1 || !any_total_pair(ci, i0, ni, j0, nj, 0, 0))
        return 0;
    int *fi = malloc((size_t)ni * sizeof *fi), *gj = malloc((size_t)nj * sizeof *gj);
    ac_ci_csf *F = malloc((size_t)ni * sizeof *F), *G = malloc((size_t)nj * sizeof *G);
    double *sub = malloc(nn * sizeof *sub);
    unsigned char *seen = calloc((size_t)ni, 1);
    if (!fi || !gj || !F || !G || !sub || !seen) {
        *err = 1;
        goto done;
    }
    for (int i = 0; i < ni && !*err; i++) {
        if (seen[i])
            continue;
        ac_lab a = ci->csf[i0 + i].c.cpl[ci->csf[i0 + i].c.nsh];
        int nf = 0, ng = 0;
        for (int p = i; p < ni; p++) {
            ac_lab b = ci->csf[i0 + p].c.cpl[ci->csf[i0 + p].c.nsh];
            if (!seen[p] && a.ts == b.ts && a.tl == b.tl) {
                seen[p] = 1;
                fi[nf] = p;
                F[nf++] = ci->csf[i0 + p];
            }
        }
        for (int q = 0; q < nj; q++) {
            ac_lab b = ci->csf[j0 + q].c.cpl[ci->csf[j0 + q].c.nsh];
            if (a.ts == b.ts && a.tl == b.tl) {
                gj[ng] = q;
                G[ng++] = ci->csf[j0 + q];
            }
        }
        if (!ng)
            continue;
        /* upper: both runs are the same, so fi and gj are the same list and
           j >= i is the same condition on the positions in it */
        hij_sub(ci, nf, F, ng, G, upper, sub, err);
        if (*err)
            break;
        for (int p = 0; p < nf; p++)
            for (int q = 0; q < ng; q++)
                out[(size_t)fi[p] * (size_t)nj + (size_t)gj[q]] = sub[(size_t)p * (size_t)ng + q];
    }
done:
    free(fi);
    free(gj);
    free(F);
    free(G);
    free(sub);
    free(seen);
    return *err ? -1 : 0;
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

int ac_ci_one_ll_n(ac_ci *ci, int i, int j, int ks, int kl,
                   double (*f)(void *ctx, int a, int b, int la, int lb), int nf,
                   void *const *ctx, double *out, int *err)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1];
    double z[AC_CI_ONE_N];
    *err = 0;
    for (int x = 0; x < nf; x++)
        out[x] = 0.0;
    if (nf < 1 || nf > AC_CI_ONE_N) {
        *err = 1;
        return -1;
    }
    int n = pair_csfs(ci, &ci->csf[i], &ci->csf[j], &bf, &bg, seq);
    if (n < 0) {
        *err = 1;
        return -1;
    }
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (!tri2(FT.ts, ks, GT.ts) || !tri2(FT.tl, kl, GT.tl))
        return 0;
    int u = 0, v = 0, nu = 0, nv = 0, ndiff = 0;
    for (int s = 1; s <= n; s++) {
        int d = bf.nel[s] - bg.nel[s];
        ndiff += abs(d);
        if (d == 1) u = s, nu++;
        else if (d == -1) v = s, nv++;
    }
    if (ndiff > 2 || (ndiff == 2 && (nu != 1 || nv != 1)))
        return 0;
    for (int a = 1; a <= n; a++)
        for (int b = 1; b <= n; b++) {
            if (ndiff == 0 ? a != b : !(a == u && b == v))
                continue;
            int l = bf.l[a], lb = bf.l[b], any = 0;
            if (bg.nel[b] == 0 || !spectators_match(&bf, &bg, a, b, 0, 0))
                continue;
            if (kl > 2 * (l + lb) || kl < 2 * abs(l - lb))
                continue;
            if (a == b && bg.nel[a] == 4 * l + 2 && (ks || kl))
                continue;
            for (int x = 0; x < nf; x++) {
                z[x] = f(ctx[x], seq[a], seq[b], l, lb);
                any |= z[x] != 0.0;
            }
            if (!any)
                continue;
            ac_otree tr;
            ac_lab p = {ks, kl};
            ac_ot_one_body(&tr, a, l, b, lb, p);
            double inv = ac_factor_op(ci->t, &bf, &bg, &tr, 0, err);
            ci->ops_evaluated++;
            if (*err)
                return -1;
            for (int x = 0; x < nf; x++)
                if (z[x] != 0.0)
                    out[x] += z[x] * inv;
        }
    if ((ks / 2 + kl / 2) % 2)
        for (int x = 0; x < nf; x++)
            out[x] = -out[x];
    return 0;
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

/* ---- H_SO2 and H_SS for all the CSF pairs of two configurations --------
   The CSFs i0 .. i0+ni-1 share their orbitals and occupations, and so do
   j0 .. j0+nj-1. Then the operators (a, b, c, d; k, g or lam), their trees
   and their Theta are the same for every pair, and only the shell states and
   the couplings differ: the loops of ac_ci_so2 and ac_ci_ss run once, and
   each operator is evaluated for all the pairs together
   (ac_factor_op_block). What ac_ci_so2 and ac_ci_ss decide pair by pair
   (the triangle of the total labels, more than four shells that differ,
   the spectators of an operator) is kept per pair, and the sums are formed
   in the same order, so every element is the one they return, bit for bit. */
typedef struct {
    ac_pair_block pb;          /* owns what the next lines point to          */
    int ni, nj, n;
    const ac_csf *bf, *bg;     /* the CSFs on the union of the slots         */
    ac_csf_set B, K;
    const int *seq, *delta;
    const unsigned long long *mis; /* per pair: bit s when shell s differs   */
    const unsigned char *ok;   /* per pair: wanted, at most 4 shells differ  */
    unsigned char *sel;        /* per pair: ok and the total labels allow an
                                  element of the operator at hand (bp2_select) */
    double *inv[2], *work;
} bp2;

static void bp2_end(bp2 *x)
{
    ac_pair_block_free(&x->pb);
    free(x->sel);
    free(x->inv[0]);
    free(x->inv[1]);
    free(x->work);
}

/* 0, or the error of ac_ci_so2 (1: too many slots; here also out of memory) */
static int bp2_begin(bp2 *x, const ac_ci *ci, int i0, int ni, int j0, int nj, int upper)
{
    size_t nn = (size_t)ni * (size_t)nj;
    memset(x, 0, sizeof *x);
    x->sel = malloc(nn ? nn : 1);
    x->inv[0] = malloc((nn ? nn : 1) * sizeof(double));
    x->inv[1] = malloc((nn ? nn : 1) * sizeof(double));
    x->work = malloc((nn ? nn : 1) * sizeof(double));
    if (!x->sel || !x->inv[0] || !x->inv[1] || !x->work)
        return 1;
    if (ac_pair_block_make(&x->pb, ci->t, ni, &ci->csf[i0], nj, &ci->csf[j0], upper))
        return 1;
    x->ni = ni;
    x->nj = nj;
    x->n = x->pb.n;
    x->bf = x->pb.bf;
    x->bg = x->pb.bg;
    x->B = x->pb.B;
    x->K = x->pb.K;
    x->seq = x->pb.seq;
    x->delta = x->pb.delta;
    x->mis = x->pb.mis;
    x->ok = x->pb.ok;
    return 0;
}

/* sel for an operator of the ranks ks/2 (spin) and kl/2 (orbital): the
   triangles of the total labels; the number of selected pairs */
static long bp2_select(bp2 *x, int ks, int kl)
{
    long np = 0;
    for (int i = 0; i < x->ni; i++)
        for (int j = 0; j < x->nj; j++) {
            size_t pq = (size_t)i * x->nj + j;
            ac_lab FT = x->bf[i].cpl[x->n], GT = x->bg[j].cpl[x->n];
            x->sel[pq] = x->ok[pq] && tri2(FT.ts, ks, GT.ts) && tri2(FT.tl, kl, GT.tl);
            np += x->sel[pq];
        }
    return np;
}

/* the selected pairs whose spectators (the shells outside a, b, c, d) match */
static long bp2_count(const bp2 *x, unsigned long long opm)
{
    size_t nn = (size_t)x->ni * (size_t)x->nj;
    long np = 0;
    for (size_t p = 0; p < nn; p++)
        np += x->sel[p] && !(x->mis[p] & ~opm);
    return np;
}

static void bp2_so2(ac_ci *ci, bp2 *x, int i0, int j0, double *out, int *err)
{
    size_t nn = (size_t)x->ni * (size_t)x->nj;
    const ac_ci_integrals *in = &ci->in;
    const ac_csf *bf = &x->bf[0];
    const int *seq = x->seq, *delta = x->delta;
    int n = x->n;
    for (size_t p = 0; p < nn; p++)
        out[p] = 0.0;
    if (!in->rkg || (ac_atsp_compat && four_shell_double(ci, i0, j0)) || !bp2_select(x, 2, 2))
        return;
    double a2 = ac_alpha * ac_alpha;
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
                unsigned long long opm = 1ULL << a | 1ULL << b | 1ULL << c | 1ULL << d;
                long npass = bp2_count(x, opm);
                if (!npass)
                    continue;
                int la = bf->l[a], lb = bf->l[b], lc = bf->l[c], ld = bf->l[d];
                for (int k = 0; k <= lc + ld && k <= la + lb + 2; k++) {
                    double cc = cred(lc, k, ld);
                    if (cc == 0.0)
                        continue;
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
                        /* so2_term for all the pairs */
                        for (int s1 = 0; s1 <= 1; s1++) {
                            int s2 = 1 - s1;
                            if (!(ac_so2_parts & (s1 ? 1 : 2)))
                                continue;
                            ac_otree tr;
                            ac_lab p = {2 * s1, 2 * g}, q = {2 * s2, 2 * k}, tot = {2, 2};
                            ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, q, tot);
                            ac_factor_op_block(ci->t, &x->B, &x->K, &tr, 1, x->inv[s1], x->work,
                                               err);
                            ci->ops_evaluated += npass;
                            if (*err)
                                return;
                        }
                        for (size_t pq = 0; pq < nn; pq++) {
                            if (!x->sel[pq] || (x->mis[pq] & ~opm))
                                continue;
                            double sum = 0.0;
                            for (int s1 = 0; s1 <= 1; s1++) {
                                int s2 = 1 - s1;
                                if (!(ac_so2_parts & (s1 ? 1 : 2)))
                                    continue;
                                sum += (s2 ? 2.0 : 1.0) * theta * x->inv[s1][pq];
                            }
                            out[pq] += sum;
                        }
                    }
                }
            }
        }
}

static void bp2_ss(ac_ci *ci, bp2 *x, int i0, int j0, double *out, int *err)
{
    size_t nn = (size_t)x->ni * (size_t)x->nj;
    const ac_ci_integrals *in = &ci->in;
    const ac_csf *bf = &x->bf[0];
    const int *seq = x->seq, *delta = x->delta;
    int n = x->n;
    for (size_t p = 0; p < nn; p++)
        out[p] = 0.0;
    if (!in->rss || (ac_atsp_compat && four_shell_double(ci, i0, j0)) || !bp2_select(x, 4, 4))
        return;
    double a2 = ac_alpha * ac_alpha;
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
                unsigned long long opm = 1ULL << a | 1ULL << b | 1ULL << c | 1ULL << d;
                long npass = bp2_count(x, opm);
                if (!npass)
                    continue;
                int la = bf->l[a], lb = bf->l[b], lc = bf->l[c], ld = bf->l[d];
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
                        ac_factor_op_block(ci->t, &x->B, &x->K, &tr, 1, x->inv[0], x->work, err);
                        ci->ops_evaluated += npass;
                        if (*err)
                            return;
                        for (size_t pq = 0; pq < nn; pq++)
                            if (x->sel[pq] && !(x->mis[pq] & ~opm))
                                out[pq] += theta * x->inv[0][pq];
                    }
                }
            }
        }
}

int ac_ci_bp2_block(ac_ci *ci, int i0, int ni, int j0, int nj, int upper, double *so2, double *ss,
                    double *tsec, int *err)
{
    bp2 x;
    double t0 = tsec ? omp_get_wtime() : 0.0, t1;
    size_t nn = (size_t)ni * (size_t)nj;
    /* a pair of runs whose total labels allow neither operator */
    if (so2 && !(ci->in.rkg && any_total_pair(ci, i0, ni, j0, nj, 2, 2))) {
        for (size_t pq = 0; pq < nn; pq++)
            so2[pq] = 0.0;
        so2 = NULL;
    }
    if (ss && !(ci->in.rss && any_total_pair(ci, i0, ni, j0, nj, 4, 4))) {
        for (size_t pq = 0; pq < nn; pq++)
            ss[pq] = 0.0;
        ss = NULL;
    }
    *err = 0;
    if (!so2 && !ss)
        return 0;
    *err = bp2_begin(&x, ci, i0, ni, j0, nj, upper);
    if (!*err && so2)
        bp2_so2(ci, &x, i0, j0, so2, err);
    if (tsec) {
        t1 = omp_get_wtime();
        tsec[0] += t1 - t0;
        t0 = t1;
    }
    if (!*err && ss)
        bp2_ss(ci, &x, i0, j0, ss, err);
    if (tsec)
        tsec[1] += omp_get_wtime() - t0;
    bp2_end(&x);
    return *err ? -1 : 0;
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

/* H_OO for all the CSF pairs of two configurations, as ac_ci_bp2_block does
   for H_SO2 and H_SS: the loops of ac_ci_oo run once and each operator is
   evaluated for all the pairs together; the sums are formed in the same
   order, so every element is the one ac_ci_oo returns. out[ni * nj]; the
   pairs j < i are 0 when upper is set. */
int ac_ci_oo_block(ac_ci *ci, int i0, int ni, int j0, int nj, int upper, double *out, int *err)
{
    size_t nn = (size_t)ni * (size_t)nj;
    const ac_ci_integrals *in = &ci->in;
    *err = 0;
    for (size_t p = 0; p < nn; p++)
        out[p] = 0.0;
    if (!in->roo || !any_total_pair(ci, i0, ni, j0, nj, 0, 0))
        return 0;
    bp2 x;
    *err = bp2_begin(&x, ci, i0, ni, j0, nj, upper);
    if (*err || !bp2_select(&x, 0, 0)) {
        bp2_end(&x);
        return *err ? -1 : 0;
    }
    const ac_csf *bf = &x.bf[0];
    const int *seq = x.seq, *delta = x.delta;
    int n = x.n;
    double a2 = ac_alpha * ac_alpha;
    void *zx = in->zctx;
    for (int a = 1; a <= n && !*err; a++)
        for (int c = 1; c <= n && !*err; c++) {
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
            for (int ord = 0; ord < 2 && !*err; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                unsigned long long opm = 1ULL << a | 1ULL << b | 1ULL << c | 1ULL << d;
                long npass = bp2_count(&x, opm);
                if (!npass)
                    continue;
                int la = bf->l[a], lb = bf->l[b], lc = bf->l[c], ld = bf->l[d];
                for (int k = 0; k <= la + lb + 1 && k <= lc + ld + 1 && !*err; k++) {
                    for (int g = k - 1; g <= k + 1; g++) {
                        if (g < 0)
                            continue;
                        unsigned long long key =
                            thc_key(THC_OO, seq[a], seq[b], seq[c], seq[d], k, g);
                        double theta = 0.0;
                        if (!thc_get(ci, key, &theta)) {
                            if ((la + lb + k + 1) % 2 == 0 && (lc + ld + k + 1) % 2 == 0 &&
                                tri2(2 * la, 2 * g, 2 * lb) && tri2(2 * lc, 2 * g, 2 * ld)) {
                                double nj9 = ac_ninej(2 * k, 2 * k, 0, 2, 2, 0, 2 * g, 2 * g, 0);
                                if (nj9 != 0.0) {
                                    double rad = in->roo(zx, k, g, seq[a], seq[b], seq[c], seq[d]);
                                    theta = a2 * (k % 2 ? -1.0 : 1.0) * sqrt(2.0 * k + 1) *
                                            -sqrt(3.0) * nj9 * rad;
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
                        }
                        if (theta == 0.0)
                            continue;
                        ac_otree tr;
                        ac_lab p = {0, 2 * g}, tot = {0, 0};
                        ac_ot_two_body(&tr, a, la, b, lb, p, c, lc, d, ld, p, tot);
                        ac_factor_op_block(ci->t, &x.B, &x.K, &tr, 1, x.inv[0], x.work, err);
                        ci->ops_evaluated += npass;
                        if (*err)
                            break;
                        for (size_t pq = 0; pq < nn; pq++)
                            if (x.sel[pq] && !(x.mis[pq] & ~opm))
                                out[pq] += theta * x.inv[0][pq];
                    }
                }
            }
        }
    for (int i = 0; i < ni && !*err; i++) {
        ac_lab FT = x.bf[i].cpl[n];
        double norm = sqrt((double)(FT.ts + 1) * (FT.tl + 1));
        for (int j = 0; j < nj; j++)
            out[(size_t)i * nj + j] = x.sel[(size_t)i * nj + j] ? out[(size_t)i * nj + j] / norm : 0.0;
    }
    bp2_end(&x);
    return *err ? -1 : 0;
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

/* ac_ci_gc2e and ac_ci_oz2e for all the CSF pairs of two configurations,
   as ac_ci_bp2_block does for H_SO2 and H_SS: the same loops, run once */
static void bp2_gc2e(ac_ci *ci, bp2 *x, int kp, double *out, int *err)
{
    size_t nn = (size_t)x->ni * (size_t)x->nj;
    const ac_ci_integrals *in = &ci->in;
    const ac_csf *bf = &x->bf[0];
    const int *seq = x->seq, *delta = x->delta;
    int n = x->n;
    for (size_t p = 0; p < nn; p++)
        out[p] = 0.0;
    if (!in->rgc || kp < 0 || kp > 2 || !bp2_select(x, 2, 2 * kp))
        return;
    static const double cp[3] = {1.1547005383792515, 1.0, -1.2909944487358056};
    double a2 = ac_alpha * ac_alpha;
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
                unsigned long long opm = 1ULL << a | 1ULL << b | 1ULL << c | 1ULL << d;
                long npass = bp2_count(x, opm);
                if (!npass)
                    continue;
                int la = bf->l[a], lb = bf->l[b], lc = bf->l[c], ld = bf->l[d];
                for (int k = 0; k <= lc + ld; k++) {
                    double cc = cred(lc, k, ld);
                    if (cc == 0.0 || (la + lb + k) % 2)
                        continue;
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
                            ac_factor_op_block(ci->t, &x->B, &x->K, &tr, 1, x->inv[s1], x->work,
                                               err);
                            ci->ops_evaluated += npass;
                            if (*err)
                                return;
                        }
                        for (size_t pq = 0; pq < nn; pq++) {
                            if (!x->sel[pq] || (x->mis[pq] & ~opm))
                                continue;
                            for (int s1 = 0; s1 <= 1; s1++)
                                if (ac_gc2e_parts & (s1 ? 1 : 2))
                                    out[pq] += (s1 ? 1.0 : 2.0) * theta * x->inv[s1][pq];
                        }
                    }
                }
            }
        }
    /* Edmonds = (-1)^(ks + kl) x invariant; a pair without an element has 0 */
    if ((1 + kp) % 2)
        for (size_t pq = 0; pq < nn; pq++)
            if (x->sel[pq])
                out[pq] = -out[pq];
}

static void bp2_oz2e(ac_ci *ci, bp2 *x, double *out, int *err)
{
    size_t nn = (size_t)x->ni * (size_t)x->nj;
    const ac_ci_integrals *in = &ci->in;
    const ac_csf *bf = &x->bf[0];
    const int *seq = x->seq, *delta = x->delta;
    int n = x->n;
    for (size_t p = 0; p < nn; p++)
        out[p] = 0.0;
    if (!in->roz || !bp2_select(x, 0, 2))
        return;
    double a2 = ac_alpha * ac_alpha;
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
                unsigned long long opm = 1ULL << a | 1ULL << b | 1ULL << c | 1ULL << d;
                long npass = bp2_count(x, opm);
                if (!npass)
                    continue;
                int la = bf->l[a], lb = bf->l[b], lc = bf->l[c], ld = bf->l[d];
                for (int k = 0; k <= lc + ld + 1; k++) {
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
                                ac_factor_op_block(ci->t, &x->B, &x->K, &tr, 1, x->inv[0], x->work,
                                                   err);
                                ci->ops_evaluated += npass;
                                if (*err)
                                    return;
                                for (size_t pq = 0; pq < nn; pq++)
                                    if (x->sel[pq] && !(x->mis[pq] & ~opm))
                                        out[pq] += theta * x->inv[0][pq];
                            }
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
                        ac_factor_op_block(ci->t, &x->B, &x->K, &tr, 1, x->inv[0], x->work, err);
                        ci->ops_evaluated += npass;
                        if (*err)
                            return;
                        for (size_t pq = 0; pq < nn; pq++)
                            if (x->sel[pq] && !(x->mis[pq] & ~opm))
                                out[pq] += theta * x->inv[0][pq];
                    }
                }
            }
        }
    /* Edmonds = (-1)^(0 + 1) x invariant; a pair without an element has 0 */
    for (size_t pq = 0; pq < nn; pq++)
        if (x->sel[pq])
            out[pq] = -out[pq];
}

int ac_ci_prop2_block(ac_ci *ci, int kp, int i0, int ni, int j0, int nj, double *out, int *err)
{
    bp2 x;
    *err = 0;
    if (kp < -1 || kp > 2)
        return -2;
    if (!any_total_pair(ci, i0, ni, j0, nj, kp < 0 ? 0 : 2, kp < 0 ? 2 : 2 * kp)) {
        for (size_t pq = 0; pq < (size_t)ni * (size_t)nj; pq++)
            out[pq] = 0.0;
        return 0;
    }
    *err = bp2_begin(&x, ci, i0, ni, j0, nj, 0);
    if (!*err) {
        if (kp == -1)
            bp2_oz2e(ci, &x, out, err);
        else
            bp2_gc2e(ci, &x, kp, out, err);
    }
    bp2_end(&x);
    return *err ? -1 : 0;
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

void dsytrf_(const char *uplo, const int *n, double *a, const int *lda, int *ipiv, double *work,
             const int *lwork, int *info);

int ac_ci_count_below(int n, double *H, double sigma, int *count)
{
    int info = 0, lwork = -1, neg = 0;
    double q;
    *count = 0;
    if (n < 1)
        return 0;
    for (int i = 0; i < n; i++)
        H[(size_t)i * n + i] -= sigma;
    int *ipiv = malloc((size_t)n * sizeof *ipiv);
    if (!ipiv)
        return -1;
    dsytrf_("L", &n, H, &n, ipiv, &q, &lwork, &info);
    lwork = (int)q;
    double *work = info ? NULL : malloc((size_t)lwork * sizeof *work);
    if (!work) {
        free(ipiv);
        return info ? info : -1;
    }
    dsytrf_("L", &n, H, &n, ipiv, work, &lwork, &info);
    free(work);
    /* info > 0: a block of D is exactly singular (sigma is an eigenvalue);
       the factorization is complete, and a zero is not counted as below */
    if (info < 0) {
        free(ipiv);
        return info;
    }
    for (int k = 0; k < n; k++) {
        double a = H[(size_t)k * n + k];
        if (ipiv[k] > 0 || k == n - 1)
            neg += a < 0.0;
        else {
            /* the 2 x 2 block (a b; b c) of D in rows k, k+1: the signs of
               its eigenvalues from its determinant and trace */
            double b = H[(size_t)k * n + k + 1], c = H[(size_t)(k + 1) * n + k + 1];
            double det = a * c - b * b;
            neg += det < 0.0 ? 1 : det > 0.0 && a + c < 0.0 ? 2 : (det == 0.0 && a + c < 0.0);
            k++;
        }
    }
    free(ipiv);
    *count = neg;
    return 0;
}

/* a.b in eight partial sums: one running sum waits for each addition and
   cannot be vectorized (the order of the additions is fixed), which made H x
   of the Davidson iteration several times slower than it has to be */
static double dot(int n, const double *a, const double *b)
{
    double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int u = 0; u < 8; u++)
            s[u] += a[i + u] * b[i + u];
    double t = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    for (; i < n; i++)
        t += a[i] * b[i];
    return t;
}

/* out[j] = dot(n, a, b_j) for four vectors at once: each sum is formed
   exactly as in dot, but the four do not wait for one another, and a is read
   once */
static void dot4(int n, const double *a, const double *b0, const double *b1, const double *b2,
                 const double *b3, double *out)
{
    double s0[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double s1[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double s2[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double s3[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int u = 0; u < 8; u++) {
            double x = a[i + u];
            s0[u] += x * b0[i + u];
            s1[u] += x * b1[i + u];
            s2[u] += x * b2[i + u];
            s3[u] += x * b3[i + u];
        }
    double t0 = ((s0[0] + s0[4]) + (s0[2] + s0[6])) + ((s0[1] + s0[5]) + (s0[3] + s0[7]));
    double t1 = ((s1[0] + s1[4]) + (s1[2] + s1[6])) + ((s1[1] + s1[5]) + (s1[3] + s1[7]));
    double t2 = ((s2[0] + s2[4]) + (s2[2] + s2[6])) + ((s2[1] + s2[5]) + (s2[3] + s2[7]));
    double t3 = ((s3[0] + s3[4]) + (s3[2] + s3[6])) + ((s3[1] + s3[5]) + (s3[3] + s3[7]));
    for (; i < n; i++) {
        t0 += a[i] * b0[i];
        t1 += a[i] * b1[i];
        t2 += a[i] * b2[i];
        t3 += a[i] * b3[i];
    }
    out[0] = t0, out[1] = t1, out[2] = t2, out[3] = t3;
}

/* AV = H V for m columns; H symmetric, so row r is the contiguous column r.
   With part: the rows of part[r] == cur only, the others are 0 (the vectors
   are 0 there and H does not couple the parts) */
static void hmul(int n, const double *H, int m, const double *V, double *AV, const int *part,
                 int cur)
{
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n; r++) {
        const double *h = H + (size_t)r * n;
        int j = 0;
        if (part && part[r] != cur) {
            for (; j < m; j++)
                AV[(size_t)j * n + r] = 0.0;
            continue;
        }
        for (; j + 4 <= m; j += 4) {
            double o[4];
            dot4(n, h, V + (size_t)j * n, V + (size_t)(j + 1) * n, V + (size_t)(j + 2) * n,
                 V + (size_t)(j + 3) * n, o);
            for (int u = 0; u < 4; u++)
                AV[(size_t)(j + u) * n + r] = o[u];
        }
        for (; j < m; j++)
            AV[(size_t)j * n + r] = dot(n, h, V + (size_t)j * n);
    }
}

/* the matrix of the Davidson iteration: dense (H, both triangles), or
   (H NULL) given by its product with vectors and its single elements */
typedef struct {
    const double *H;
    void (*mul)(void *ctx, int m, const double *V, double *AV);
    double (*elem)(void *ctx, int a, int b);
    void *ctx;
    /* not NULL: the iteration stays in the rows of part[i] == cur, a part of
       the matrix that H does not couple to the rest (mul must then give 0
       in the other rows) */
    const int *part;
    int cur;
} dav_mat;

/* the element (a, b) */
static double dav_elem(int n, const dav_mat *A, int a, int b)
{
    return A->H ? A->H[(size_t)n * b + a] : A->elem(A->ctx, a, b);
}

static void dav_mul(int n, const dav_mat *A, int m, const double *V, double *AV)
{
    if (A->H)
        hmul(n, A->H, m, V, AV, A->part, A->cur);
    else
        A->mul(A->ctx, m, V, AV);
}

/* rows handled together in the sums over the subspace vectors */
#define DAV_STRIP 64

typedef struct {
    double d;
    int i;
} diag_t;

static int by_diag(const void *a, const void *b)
{
    double d = ((const diag_t *)a)->d - ((const diag_t *)b)->d;
    return (d > 0) - (d < 0);
}

/* Davidson (1975) in the block form of Liu (1978). The subspace holds up to
   mmax vectors; when it is full it is restarted from the keep lowest Ritz
   vectors (keep > k: restarting from the k wanted ones alone loses most of
   what the subspace has learned and doubles the number of H x; for 9 roots
   120 and 27 were the fastest of the sizes tried on Ti blocks of dimension
   533 to 3184). The projected matrix is extended by the new vectors only. */
static int dav_core(int n, const dav_mat *A, int k, double tol, double *w, double *x, int *niter)
{
    /* ne: the rows the iteration lives in (all, or those of the part) */
    int ne = n;
    if (A->part) {
        ne = 0;
        for (int i = 0; i < n; i++)
            ne += A->part[i] == A->cur;
    }
    if (k < 1 || k > ne)
        return -1;
    int mmax = 8 * k + 48, keep = 3 * k, m0 = 2 * k > 200 ? 2 * k : 200;
    if (mmax > ne)
        mmax = ne;
    if (mmax < 2 * k)
        mmax = 2 * k < ne ? 2 * k : ne;
    if (keep > mmax - k)
        keep = mmax - k;
    if (keep < k)
        keep = k;
    if (m0 > ne)
        m0 = ne;
    size_t N = (size_t)n, M = (size_t)mmax;
    double *V = malloc(N * M * sizeof *V), *AV = malloc(N * M * sizeof *AV);
    double *X = malloc(N * k * sizeof *X), *AX = malloc(N * k * sizeof *AX);
    double *P = malloc(M * M * sizeof *P); /* the projected matrix V^T H V */
    double *G = malloc(M * M * sizeof *G); /* its eigenvectors */
    double *th = malloc((size_t)(mmax > m0 ? mmax : m0) * sizeof *th);
    double *B = malloc((size_t)m0 * m0 * sizeof *B), *R = malloc(N * k * sizeof *R);
    double *cf = malloc(M * sizeof *cf), *rn = malloc((size_t)k * sizeof *rn);
    diag_t *dg = malloc(N * sizeof *dg);
    double *dd = malloc(N * sizeof *dd); /* the diagonal */
    int rc = -1, it = 0;
    /* AC_TIME: where the time goes (start, H x, Rayleigh-Ritz, Ritz vectors
       and residuals, corrections and restarts), the H x count and the restarts */
    double tm[6] = {0}, t0 = omp_get_wtime(), t1;
    long nmul = 0;
    int nrestart = 0, nstall = 0;
#define LAP(k) (t1 = omp_get_wtime(), tm[k] += t1 - t0, t0 = t1)
    if (!V || !AV || !X || !AX || !P || !G || !th || !B || !R || !cf || !rn || !dg || !dd)
        goto out;
    /* start: the lowest eigenvectors of the block of the m0 lowest diagonal elements */
    for (int i = 0, e = 0; i < n; i++) {
        dd[i] = dav_elem(n, A, i, i);
        if (!A->part || A->part[i] == A->cur)
            dg[e].d = dd[i], dg[e++].i = i;
    }
    qsort(dg, (size_t)ne, sizeof *dg, by_diag);
    for (int b = 0; b < m0; b++)
        for (int a = 0; a < m0; a++)
            B[(size_t)b * m0 + a] = dav_elem(n, A, dg[a].i, dg[b].i);
    if ((rc = syev("V", m0, B, th)) != 0)
        goto out;
    memset(V, 0, N * k * sizeof *V);
    for (int j = 0; j < k; j++)
        for (int a = 0; a < m0; a++)
            V[N * j + dg[a].i] = B[(size_t)j * m0 + a];
    LAP(0);
    dav_mul(n, A, k, V, AV);
    nmul += k;
    LAP(1);
    int m = 0, mnew = k; /* vectors already in P; vectors added since */
    rc = 1;
    for (it = 1; it <= 1000; it++) {
        /* Rayleigh-Ritz in span V: the rows and columns of the new vectors */
#pragma omp parallel for schedule(dynamic, 8)
        for (int a = 0; a < m + mnew; a++)
            for (int b = a < m ? m : a; b < m + mnew; b++)
                P[(size_t)b * M + a] = P[(size_t)a * M + b] =
                    0.5 * (dot(n, V + N * a, AV + N * b) + dot(n, V + N * b, AV + N * a));
        m += mnew;
        LAP(2);
        for (int b = 0; b < m; b++)
            memcpy(G + (size_t)b * m, P + (size_t)b * M, (size_t)m * sizeof *G);
        if (syev("V", m, G, th)) {
            rc = -1;
            goto out;
        }
        LAP(5);
        /* the Ritz vectors X = V G, with A X and the residuals, a strip of
           rows at a time: for each row the sum over the subspace vectors
           runs in the same order as before, but the strip stays in the
           cache while it does */
#pragma omp parallel for schedule(static)
        for (int i0 = 0; i0 < n; i0 += DAV_STRIP) {
            int ns = n - i0 < DAV_STRIP ? n - i0 : DAV_STRIP;
            double s[DAV_STRIP], t[DAV_STRIP];
            for (int j = 0; j < k; j++) {
                for (int i = 0; i < ns; i++)
                    s[i] = t[i] = 0.0;
                for (int a = 0; a < m; a++) {
                    double g = G[(size_t)j * m + a];
                    const double *va = V + N * a + i0, *wa = AV + N * a + i0;
                    for (int i = 0; i < ns; i++) {
                        s[i] += va[i] * g;
                        t[i] += wa[i] * g;
                    }
                }
                for (int i = 0; i < ns; i++) {
                    X[N * j + i0 + i] = s[i], AX[N * j + i0 + i] = t[i];
                    R[N * j + i0 + i] = t[i] - th[j] * s[i];
                }
            }
        }
        int nnew = 0, conv = 1, nopen = 0;
        double rmax = 0.0;
        for (int j = 0; j < k; j++) {
            rn[j] = sqrt(dot(n, R + N * j, R + N * j));
            rmax = rn[j] > rmax ? rn[j] : rmax;
            if (rn[j] >= tol)
                conv = 0, nopen++;
        }
        if (getenv("AC_DAVDEBUG")) {
            fprintf(stderr, "davidson %d: m %d, w0 %.15f, |r| %.2e, open %d, |r_j|", it, m, th[0],
                    rmax, nopen);
            for (int j = 0; j < k; j++)
                fprintf(stderr, " %.1e", rn[j]);
            fprintf(stderr, "\n");
        }
        LAP(3);
        if (conv) {
            rc = 0;
            break;
        }
        if (m + k > mmax) {
            /* restart from the keep lowest Ritz vectors: V <- V G, AV <- AV G,
               row by row; the projected matrix becomes diag(th) */
#pragma omp parallel
            {
                double *row = malloc(2 * M * sizeof *row);
#pragma omp for schedule(static)
                for (int i = 0; i < n; i++) {
                    if (!row)
                        continue;
                    for (int a = 0; a < m; a++)
                        row[a] = V[N * a + i], row[M + a] = AV[N * a + i];
                    for (int j = 0; j < keep; j++) {
                        const double *g = G + (size_t)j * m;
                        double s = 0.0, t = 0.0;
                        for (int a = 0; a < m; a++)
                            s += row[a] * g[a], t += row[M + a] * g[a];
                        V[N * j + i] = s, AV[N * j + i] = t;
                    }
                }
#pragma omp critical(ac_dav_fail)
                if (!row)
                    rc = -1;
                free(row);
            }
            if (rc < 0)
                goto out;
            for (int b = 0; b < keep; b++)
                for (int a = 0; a < keep; a++)
                    P[(size_t)b * M + a] = a == b ? th[a] : 0.0;
            m = keep;
            nrestart++;
        }
        /* corrections (th - D)^-1 r of the unconverged roots, orthonormalized
           (Gram-Schmidt against all of V twice) */
        for (int j = 0; j < k; j++) {
            double *r = R + N * j, *t = V + N * (m + nnew);
            if (rn[j] < tol)
                continue;
            for (int i = 0; i < n; i++) {
                double d = th[j] - dd[i];
                t[i] = r[i] / (fabs(d) > 1e-8 ? d : (d < 0 ? -1e-8 : 1e-8));
            }
            double nt = sqrt(dot(n, t, t));
            for (int i = 0; i < n; i++)
                t[i] /= nt;
            int nv = m + nnew;
            for (int pass = 0; pass < 2; pass++) {
#pragma omp parallel for schedule(static)
                for (int a = 0; a < nv; a++)
                    cf[a] = dot(n, V + N * a, t);
                /* a strip of rows at a time, as for the Ritz vectors */
#pragma omp parallel for schedule(static)
                for (int i0 = 0; i0 < n; i0 += DAV_STRIP) {
                    int ns = n - i0 < DAV_STRIP ? n - i0 : DAV_STRIP;
                    double s[DAV_STRIP];
                    for (int i = 0; i < ns; i++)
                        s[i] = 0.0;
                    for (int a = 0; a < nv; a++) {
                        double ca = cf[a];
                        const double *va = V + N * a + i0;
                        for (int i = 0; i < ns; i++)
                            s[i] += ca * va[i];
                    }
                    for (int i = 0; i < ns; i++)
                        t[i0 + i] -= s[i];
                }
            }
            nt = sqrt(dot(n, t, t));
            if (nt < 1e-8) /* (nearly) in span V already */
                continue;
            for (int i = 0; i < n; i++)
                t[i] /= nt;
            nnew++;
        }
        LAP(4);
        if (nnew == 0) {
            nstall = 1;
            break; /* stalled */
        }
        dav_mul(n, A, nnew, V + N * m, AV + N * m);
        nmul += nnew;
        LAP(1);
        mnew = nnew;
    }
    if (getenv("AC_TIME"))
        fprintf(stderr,
                "davidson: n %d, %d roots, %d iterations, %ld H x, %d restarts (%d of %d vectors kept)%s;"
                " start %.2f, H x %.2f, Rayleigh-Ritz %.2f (dsyev %.2f), residuals %.2f, "
                "corrections %.2f s\n",
                n, k, it > 1000 ? 1000 : it, nmul, nrestart, keep, mmax, nstall ? ", stalled" : "",
                tm[0], tm[1], tm[2] + tm[5], tm[5], tm[3], tm[4]);
#undef LAP
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
    free(P);
    free(G);
    free(th);
    free(B);
    free(R);
    free(cf);
    free(rn);
    free(dg);
    free(dd);
    return rc;
}

int ac_ci_davidson(int n, const double *H, int k, double tol, double *w, double *x, int *niter)
{
    dav_mat A = {H, NULL, NULL, NULL, NULL, 0};
    return dav_core(n, &A, k, tol, w, x, niter);
}

int ac_ci_davidson_op(int n, void (*mul)(void *ctx, int m, const double *V, double *AV),
                      double (*elem)(void *ctx, int a, int b), void *ctx, int k, double tol,
                      double *w, double *x, int *niter)
{
    dav_mat A = {NULL, mul, elem, ctx, NULL, 0};
    return dav_core(n, &A, k, tol, w, x, niter);
}

/* a part of at most this many rows is solved by dsyev */
#define DAV_PART_DENSE 400

/* The k lowest levels of a matrix of np parts that H does not couple: each
   part alone (its min(k, size) lowest levels; Davidson within the part, or
   dsyev when it is small), and of all those the k lowest. sel, when not
   NULL, tells the operator the part before its mul is used. */
static int dav_parts(int n, dav_mat *A, void (*sel)(void *ctx, const int *part, int cur), int np,
                     const int *part, int k, double tol, double *w, double *x, int *niter)
{
    if (k < 1 || k > n)
        return -1;
    size_t N = (size_t)n;
    double *tw = malloc((size_t)k * sizeof *tw), *tx = malloc(N * k * sizeof *tx);
    double *mw = malloc((size_t)k * sizeof *mw), *mx = malloc(N * k * sizeof *mx);
    int *mem = malloc(N * sizeof *mem);
    int rc = -1, nb = 0, its = 0;
    /* AC_DAV_PART_DENSE of the environment: another limit (for tests) */
    const char *sd = getenv("AC_DAV_PART_DENSE");
    int dense = sd ? atoi(sd) : DAV_PART_DENSE;
    if (!tw || !tx || !mw || !mx || !mem)
        goto out;
    for (int c = 0; c < np; c++) {
        int nc = 0;
        for (int i = 0; i < n; i++)
            if (part[i] == c)
                mem[nc++] = i;
        if (nc == 0)
            continue;
        int kc = k < nc ? k : nc;
        if (nc <= dense || 4 * kc >= nc) {
            double *B = malloc((size_t)nc * nc * sizeof *B), *th = malloc((size_t)nc * sizeof *th);
            int bad = !B || !th;
            if (!bad) {
                for (int b = 0; b < nc; b++)
                    for (int a = 0; a <= b; a++)
                        B[(size_t)b * nc + a] = B[(size_t)a * nc + b] =
                            dav_elem(n, A, mem[a], mem[b]);
                bad = syev("V", nc, B, th) != 0;
            }
            if (!bad) {
                memset(tx, 0, N * kc * sizeof *tx);
                for (int j = 0; j < kc; j++) {
                    tw[j] = th[j];
                    for (int a = 0; a < nc; a++)
                        tx[N * j + mem[a]] = B[(size_t)j * nc + a];
                }
            }
            free(B);
            free(th);
            if (bad)
                goto out;
        } else {
            int it = 0;
            A->part = part, A->cur = c;
            if (sel)
                sel(A->ctx, part, c);
            int e = dav_core(n, A, kc, tol, tw, tx, &it);
            A->part = NULL;
            if (sel)
                sel(A->ctx, NULL, 0);
            its += it;
            if (e) {
                rc = e;
                goto out;
            }
        }
        /* the k lowest of those kept and the new ones (those kept first
           among equals) */
        int a = 0, b = 0, m = 0;
        while (m < k && (a < nb || b < kc)) {
            int old = b >= kc || (a < nb && w[a] <= tw[b]);
            mw[m] = old ? w[a] : tw[b];
            memcpy(mx + N * m, old ? x + N * a : tx + N * b, N * sizeof *mx);
            m++, a += old, b += !old;
        }
        nb = m;
        memcpy(w, mw, (size_t)nb * sizeof *w);
        memcpy(x, mx, N * nb * sizeof *x);
    }
    rc = nb == k ? 0 : -1;
out:
    if (niter)
        *niter = its;
    free(tw);
    free(tx);
    free(mw);
    free(mx);
    free(mem);
    return rc;
}

int ac_ci_davidson_parts(int n, const double *H, int np, const int *part, int k, double tol,
                         double *w, double *x, int *niter)
{
    dav_mat A = {H, NULL, NULL, NULL, NULL, 0};
    if (np <= 1)
        return dav_core(n, &A, k, tol, w, x, niter);
    return dav_parts(n, &A, NULL, np, part, k, tol, w, x, niter);
}

int ac_ci_davidson_op_parts(int n, void (*mul)(void *ctx, int m, const double *V, double *AV),
                            double (*elem)(void *ctx, int a, int b),
                            void (*sel)(void *ctx, const int *part, int cur), void *ctx, int np,
                            const int *part, int k, double tol, double *w, double *x, int *niter)
{
    dav_mat A = {NULL, mul, elem, ctx, NULL, 0};
    if (np <= 1)
        return dav_core(n, &A, k, tol, w, x, niter);
    return dav_parts(n, &A, sel, np, part, k, tol, w, x, niter);
}