/*
 * ac_terms.c -- the expression of a Hamiltonian matrix element (see ac_terms.h).
 *
 * Sources: F. Sasaki, M. Sekiya, T. Noro, K. Ohtsuki, Y. Osanai, "Non-relativistic
 * configuration interaction calculations for many-electron atoms: ATOMCI", in
 * Methods and Techniques in Computational Chemistry: METECC-94, E. Clementi (ed.),
 * Cagliari (1993), pp. 163-219, Eqs. (44), (45).
 */
#include "ac_terms.h"
#include "ac_csfgen.h"
#include "ac_factor.h"
#include "ac_optree.h"
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

int ac_orb_csfs(const ac_cfptab *t, int nwt, const int *lorb, int ncfg, const int *occ,
                ac_orb_csf **out, int *nout, char *err, int errlen)
{
    *out = NULL;
    *nout = 0;
    if (nwt < 1)
        return fail(err, errlen, "%d orbitals", nwt);
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
                    return fail(err, errlen, "configuration %d: more than %d occupied orbitals", c,
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
            memcpy(seqs + (size_t)i * AC_CSF_MAXSH, sq, (size_t)n * sizeof *sq);
    }
    ac_orb_csf *v = calloc((size_t)(all.len ? all.len : 1), sizeof *v);
    if (!v) {
        ac_csflist_free(&all);
        free(seqs);
        return fail(err, errlen, "out of memory");
    }
    for (int i = 0; i < all.len; i++) {
        v[i].c = all.v[i];
        for (int s = 1; s <= all.v[i].nsh; s++)
            v[i].seq[s] = seqs[(size_t)i * AC_CSF_MAXSH + s - 1];
    }
    *out = v;
    *nout = all.len;
    ac_csflist_free(&all);
    free(seqs);
    return 0;
}

/* the first state of shell l with n electrons: the 1S vacuum or closed shell */
static int scalar_code(const ac_cfptab *t, int l, int n)
{
    const int *codes;
    return ac_cfp_states(t, l, n, &codes) > 0 ? codes[0] : 0;
}

int ac_pair_union(const ac_cfptab *t, const ac_orb_csf *F, const ac_orb_csf *G, ac_csf *bf,
                  ac_csf *bg, int *seq)
{
    int i = 1, j = 1, n = 0;
    ac_lab pf = {0, 0}, pg = {0, 0};
    while (i <= F->c.nsh || j <= G->c.nsh) {
        int sf = i <= F->c.nsh ? F->seq[i] : 1 << 30;
        int sg = j <= G->c.nsh ? G->seq[j] : 1 << 30;
        int s = sf < sg ? sf : sg;
        if (++n > AC_CSF_MAXSH)
            return -1;
        int l = sf == s ? F->c.l[i] : G->c.l[j];
        seq[n] = s;
        bf->l[n] = bg->l[n] = l;
        if (sf == s) {
            bf->code[n] = F->c.code[i];
            bf->nel[n] = F->c.nel[i];
            bf->f[n] = F->c.f[i];
            pf = bf->cpl[n] = F->c.cpl[i];
            i++;
        } else {
            bf->code[n] = scalar_code(t, l, 0);
            bf->nel[n] = 0;
            bf->f[n].ts = bf->f[n].tl = 0;
            bf->cpl[n] = pf;
        }
        if (sg == s) {
            bg->code[n] = G->c.code[j];
            bg->nel[n] = G->c.nel[j];
            bg->f[n] = G->c.f[j];
            pg = bg->cpl[n] = G->c.cpl[j];
            j++;
        } else {
            bg->code[n] = scalar_code(t, l, 0);
            bg->nel[n] = 0;
            bg->f[n].ts = bg->f[n].tl = 0;
            bg->cpl[n] = pg;
        }
    }
    bf->nsh = bg->nsh = n;
    bf->cpl[0].ts = bf->cpl[0].tl = bg->cpl[0].ts = bg->cpl[0].tl = 0;
    return n;
}

int ac_pair_spectators_match(const ac_csf *bf, const ac_csf *bg, int a, int b, int c, int d)
{
    for (int s = 1; s <= bf->nsh; s++) {
        if (s == a || s == b || s == c || s == d)
            continue;
        if (bf->code[s] != bg->code[s] || bf->nel[s] != bg->nel[s])
            return 0;
    }
    return 1;
}

int ac_pair_terms(const ac_cfptab *t, const ac_orb_csf *F, const ac_orb_csf *G, ac_term_skip skip,
                  ac_term_emit emit, void *ctx, long *nops)
{
    ac_csf bf, bg;
    int seq[AC_CSF_MAXSH + 1], err = 0;
    long ops = 0;
    int n = ac_pair_union(t, F, G, &bf, &bg, seq);
    if (n < 0)
        return 1;
    ac_lab FT = bf.cpl[n], GT = bg.cpl[n];
    if (FT.ts != GT.ts || FT.tl != GT.tl)
        return 0;

    int delta[AC_CSF_MAXSH + 1], ndiff = 0, mism = 0;
    for (int s = 1; s <= n; s++) {
        delta[s] = bf.nel[s] - bg.nel[s];
        ndiff += abs(delta[s]);
        mism += bf.code[s] != bg.code[s] || bf.nel[s] != bg.nel[s];
    }
    if (ndiff > 4 || mism > 4)
        return 0;

    /* the engine's invariant becomes the element of a component */
    double norm = sqrt((double)(FT.ts + 1) * (FT.tl + 1));

    /* one-electron part, METECC (44): -sqrt[2l] h(a,b) (b_a b~_b)^(1S) */
    int u = 0, v = 0, nu = 0, nv = 0;
    for (int s = 1; s <= n; s++) {
        if (delta[s] == 1) u = s, nu++;
        else if (delta[s] == -1) v = s, nv++;
        else if (delta[s] != 0) nu = nv = 9;
    }
    for (int a = 1; a <= n; a++)
        for (int b = 1; b <= n; b++) {
            if (ndiff == 0 ? a != b : !(nu == 1 && nv == 1 && a == u && b == v))
                continue;
            if (bf.l[a] != bf.l[b] || bg.nel[b] == 0 ||
                !ac_pair_spectators_match(&bf, &bg, a, b, 0, 0))
                continue;
            ac_otree tr;
            ac_lab p = {0, 0};
            ac_ot_one_body(&tr, a, bf.l[a], b, bf.l[b], p);
            double inv = ac_factor_op(t, &bf, &bg, &tr, 0, &err);
            ops++;
            if (err)
                goto done;
            if (inv != 0.0)
                emit(ctx, -1, seq[a], seq[b], 0, 0, -sqrt(2.0 * (2 * bf.l[a] + 1)) * inv / norm);
        }

    /* two-electron part, METECC (45): sum over ordered (a,b,c,d) and k of
       H^k R^k(ab;cd) :((b_a b~_b)^(1k)(b_c b~_d)^(1k))^(1S): */
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0) bad = 1;
                for (int r = 0; r < need[s] && nl < 2; r++)
                    list[nl++] = s;
            }
            if (bad || nl != 2)
                continue;
            for (int ord = 0; ord < 2; ord++) {
                if (ord == 1 && list[0] == list[1])
                    break;
                int b = list[ord], d = list[1 - ord];
                if (!ac_pair_spectators_match(&bf, &bg, a, b, c, d))
                    continue;
                int la = bf.l[a], lb = bf.l[b], lc = bf.l[c], ld = bf.l[d];
                int kmin = abs(la - lb) > abs(lc - ld) ? abs(la - lb) : abs(lc - ld);
                int kmax = la + lb < lc + ld ? la + lb : lc + ld;
                for (int k = kmin; k <= kmax; k++) {
                    if ((la + lb + k) % 2 || (lc + ld + k) % 2)
                        continue;
                    double hk = ac_hk(k, la, lb, lc, ld);
                    if (hk == 0.0)
                        continue;
                    if (skip && skip(ctx, k, seq[a], seq[b], seq[c], seq[d]))
                        continue;
                    ac_otree tr;
                    ac_lab pk = {0, 2 * k}, tot = {0, 0};
                    ac_ot_two_body(&tr, a, la, b, lb, pk, c, lc, d, ld, pk, tot);
                    double inv = ac_factor_op(t, &bf, &bg, &tr, 1, &err);
                    ops++;
                    if (err)
                        goto done;
                    if (inv != 0.0)
                        emit(ctx, k, seq[a], seq[b], seq[c], seq[d], hk * inv / norm);
                }
            }
        }
done:
    if (nops)
        *nops += ops;
    return err;
}

/* ---- all the CSF pairs of two configurations at once (ac_terms.h) ------ */

void ac_pair_block_free(ac_pair_block *pb)
{
    ac_csf_set_free(&pb->B);
    ac_csf_set_free(&pb->K);
    free(pb->bf);
    free(pb->bg);
    free(pb->mis);
    free(pb->ok);
    pb->bf = pb->bg = NULL;
    pb->mis = NULL;
    pb->ok = NULL;
}

int ac_pair_block_make(ac_pair_block *pb, const ac_cfptab *t, int ni, const ac_orb_csf *F, int nj,
                       const ac_orb_csf *G, int upper)
{
    size_t nn = (size_t)ni * (size_t)nj;
    memset(pb, 0, sizeof *pb);
    pb->ni = ni;
    pb->nj = nj;
    pb->bf = malloc((size_t)(ni > 0 ? ni : 1) * sizeof *pb->bf);
    pb->bg = malloc((size_t)(nj > 0 ? nj : 1) * sizeof *pb->bg);
    pb->mis = malloc((nn ? nn : 1) * sizeof *pb->mis);
    pb->ok = malloc(nn ? nn : 1);
    if (!pb->bf || !pb->bg || !pb->mis || !pb->ok || ni < 1 || nj < 1)
        goto bad;
    ac_csf tmp;
    int n = 0;
    for (int i = 0; i < ni; i++) {
        n = ac_pair_union(t, &F[i], &G[0], &pb->bf[i], &tmp, pb->seq);
        if (n < 0)
            goto bad;
    }
    for (int j = 0; j < nj; j++)
        if (ac_pair_union(t, &F[0], &G[j], &tmp, &pb->bg[j], pb->seq) != n)
            goto bad;
    pb->n = n;
    if (ac_csf_set_make(&pb->B, ni, pb->bf) || ac_csf_set_make(&pb->K, nj, pb->bg))
        goto bad;
    for (int s = 1; s <= n; s++) {
        pb->delta[s] = pb->bf[0].nel[s] - pb->bg[0].nel[s];
        pb->ndiff += abs(pb->delta[s]);
    }
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++) {
            const ac_csf *bf = &pb->bf[i], *bg = &pb->bg[j];
            unsigned long long m = 0;
            int mism = 0;
            for (int s = 1; s <= n; s++)
                if (bf->code[s] != bg->code[s] || bf->nel[s] != bg->nel[s]) {
                    m |= 1ULL << s;
                    mism++;
                }
            pb->mis[(size_t)i * nj + j] = m;
            pb->ok[(size_t)i * nj + j] = pb->ndiff <= 4 && mism <= 4 && !(upper && j < i);
        }
    return 0;
bad:
    ac_pair_block_free(pb);
    return 1;
}

int ac_pair_terms_block(const ac_cfptab *t, const ac_pair_block *pb, ac_term_skip skip,
                        ac_term_emit_block emit, void *ctx, long *nops)
{
    int ni = pb->ni, nj = pb->nj, n = pb->n, err = 0;
    size_t nn = (size_t)ni * (size_t)nj;
    const ac_csf *bf = &pb->bf[0], *bg = &pb->bg[0];
    const int *seq = pb->seq, *delta = pb->delta;
    long ops = 0, nsel = 0;
    unsigned char *sel = malloc(nn ? nn : 1);
    double *inv = malloc((nn ? nn : 1) * sizeof *inv), *work = malloc((nn ? nn : 1) * sizeof *work);
    double *coef = malloc((nn ? nn : 1) * sizeof *coef);
    double *norm = malloc((size_t)(ni > 0 ? ni : 1) * sizeof *norm);
    if (!sel || !inv || !work || !coef || !norm) {
        err = 1;
        goto done;
    }
    /* the pairs with the same total (S, L); the engine's invariant becomes
       the element of a component */
    for (int i = 0; i < ni; i++) {
        ac_lab FT = pb->bf[i].cpl[n];
        norm[i] = sqrt((double)(FT.ts + 1) * (FT.tl + 1));
        for (int j = 0; j < nj; j++) {
            ac_lab GT = pb->bg[j].cpl[n];
            size_t pq = (size_t)i * nj + j;
            sel[pq] = pb->ok[pq] && FT.ts == GT.ts && FT.tl == GT.tl;
            nsel += sel[pq];
        }
    }
    if (!nsel)
        goto done;

    /* one-electron part */
    int ndiff = pb->ndiff, u = 0, v = 0, nu = 0, nv = 0;
    for (int s = 1; s <= n; s++) {
        if (delta[s] == 1) u = s, nu++;
        else if (delta[s] == -1) v = s, nv++;
        else if (delta[s] != 0) nu = nv = 9;
    }
    for (int a = 1; a <= n; a++)
        for (int b = 1; b <= n; b++) {
            if (ndiff == 0 ? a != b : !(nu == 1 && nv == 1 && a == u && b == v))
                continue;
            if (bf->l[a] != bf->l[b] || bg->nel[b] == 0)
                continue;
            unsigned long long opm = 1ULL << a | 1ULL << b;
            long npass = 0;
            for (size_t pq = 0; pq < nn; pq++)
                npass += sel[pq] && !(pb->mis[pq] & ~opm);
            if (!npass)
                continue;
            ac_otree tr;
            ac_lab p = {0, 0};
            ac_ot_one_body(&tr, a, bf->l[a], b, bf->l[b], p);
            ac_factor_op_block(t, &pb->B, &pb->K, &tr, 0, inv, work, &err);
            ops += npass;
            if (err)
                goto done;
            long nz = 0;
            for (size_t pq = 0; pq < nn; pq++) {
                coef[pq] = 0.0;
                if (sel[pq] && !(pb->mis[pq] & ~opm) && inv[pq] != 0.0) {
                    coef[pq] = -sqrt(2.0 * (2 * bf->l[a] + 1)) * inv[pq] / norm[pq / (size_t)nj];
                    nz++;
                }
            }
            if (nz)
                emit(ctx, -1, seq[a], seq[b], 0, 0, coef);
        }

    /* two-electron part */
    for (int a = 1; a <= n; a++)
        for (int c = 1; c <= n; c++) {
            int need[AC_CSF_MAXSH + 1], bad = 0, list[2], nl = 0;
            for (int s = 1; s <= n; s++) {
                need[s] = (s == a) + (s == c) - delta[s];
                if (need[s] < 0) bad = 1;
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
                long npass = 0;
                for (size_t pq = 0; pq < nn; pq++)
                    npass += sel[pq] && !(pb->mis[pq] & ~opm);
                if (!npass)
                    continue;
                int la = bf->l[a], lb = bf->l[b], lc = bf->l[c], ld = bf->l[d];
                int kmin = abs(la - lb) > abs(lc - ld) ? abs(la - lb) : abs(lc - ld);
                int kmax = la + lb < lc + ld ? la + lb : lc + ld;
                for (int k = kmin; k <= kmax; k++) {
                    if ((la + lb + k) % 2 || (lc + ld + k) % 2)
                        continue;
                    double hk = ac_hk(k, la, lb, lc, ld);
                    if (hk == 0.0)
                        continue;
                    if (skip && skip(ctx, k, seq[a], seq[b], seq[c], seq[d]))
                        continue;
                    ac_otree tr;
                    ac_lab pk = {0, 2 * k}, tot = {0, 0};
                    ac_ot_two_body(&tr, a, la, b, lb, pk, c, lc, d, ld, pk, tot);
                    ac_factor_op_block(t, &pb->B, &pb->K, &tr, 1, inv, work, &err);
                    ops += npass;
                    if (err)
                        goto done;
                    long nz = 0;
                    for (size_t pq = 0; pq < nn; pq++) {
                        coef[pq] = 0.0;
                        if (sel[pq] && !(pb->mis[pq] & ~opm) && inv[pq] != 0.0) {
                            coef[pq] = hk * inv[pq] / norm[pq / (size_t)nj];
                            nz++;
                        }
                    }
                    if (nz)
                        emit(ctx, k, seq[a], seq[b], seq[c], seq[d], coef);
                }
            }
        }
done:
    free(sel);
    free(inv);
    free(work);
    free(coef);
    free(norm);
    if (nops)
        *nops += ops;
    return err;
}

void ac_expr_init(ac_expr *l)
{
    memset(l, 0, sizeof *l);
}

void ac_expr_free(ac_expr *l)
{
    free(l->v);
    ac_expr_init(l);
}

void ac_term_canon(int *k, int *a, int *b, int *c, int *d)
{
    int t;
    if (*a > *b) t = *a, *a = *b, *b = t;
    if (*k < 0) {
        *k = -1;
        *c = *d = 0;
        return;
    }
    if (*c > *d) t = *c, *c = *d, *d = t;
    if (*a > *c || (*a == *c && *b > *d)) {
        t = *a, *a = *c, *c = t;
        t = *b, *b = *d, *d = t;
    }
}

typedef struct {
    ac_expr *l;
    int oom;
} collect_ctx;

static void collect(void *p, int k, int a, int b, int c, int d, double coef)
{
    collect_ctx *x = p;
    ac_expr *l = x->l;
    if (x->oom)
        return;
    if (l->len == l->cap) {
        int cap = l->cap ? 2 * l->cap : 64;
        ac_expr_term *nv = realloc(l->v, (size_t)cap * sizeof *nv);
        if (!nv) {
            x->oom = 1;
            return;
        }
        l->v = nv;
        l->cap = cap;
    }
    ac_term_canon(&k, &a, &b, &c, &d);
    l->v[l->len++] = (ac_expr_term){k, a, b, c, d, coef};
}

static int same_label(const ac_expr_term *x, const ac_expr_term *y)
{
    return x->k == y->k && x->a == y->a && x->b == y->b && x->c == y->c && x->d == y->d;
}

static int cmp_label(const void *p, const void *q)
{
    const ac_expr_term *x = p, *y = q;
    if (x->k != y->k) return x->k < y->k ? -1 : 1;
    if (x->a != y->a) return x->a < y->a ? -1 : 1;
    if (x->b != y->b) return x->b < y->b ? -1 : 1;
    if (x->c != y->c) return x->c < y->c ? -1 : 1;
    if (x->d != y->d) return x->d < y->d ? -1 : 1;
    return 0;
}

int ac_pair_expr(const ac_cfptab *t, const ac_orb_csf *F, const ac_orb_csf *G, double tol,
                 ac_expr *l)
{
    collect_ctx x = {l, 0};
    l->len = 0;
    int r = ac_pair_terms(t, F, G, NULL, collect, &x, NULL);
    if (r)
        return r;
    if (x.oom)
        return -1;
    if (l->len)
        qsort(l->v, (size_t)l->len, sizeof *l->v, cmp_label);
    int n = 0;
    for (int i = 0; i < l->len;) {
        ac_expr_term m = l->v[i];
        int j = i + 1;
        for (; j < l->len && same_label(&l->v[j], &m); j++)
            m.coef += l->v[j].coef;
        if (fabs(m.coef) > tol)
            l->v[n++] = m;
        i = j;
    }
    l->len = n;
    return 0;
}
