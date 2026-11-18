/*
 * ac_factor.c -- factorized matrix elements (see ac_factor.h).
 */
#include "ac_factor.h"
#include "ac_labmath.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const ac_cfptab *t;
    int err;
} ctx;

/* a shell state: code, electron count, label */
typedef struct {
    int code, nel;
    ac_lab lab;
} sstate;

static double shell_me(ctx *c, const ac_otree *tr, int node, int l,
                       const sstate *f, const sstate *g);

/* <f~ (b~ b)^x g> through (b~ b)^x = -(-1)^{2b-x} (b b~)^x + d_x0 sqrt[b],
   for a truncated shell whose intermediate N+1 states do not exist */
static double ann_cre_by_anticommutation(ctx *c, const ac_otree *tr, int node,
                                         int l, const sstate *f, const sstate *g)
{
    const ac_onode *x = &tr->n[node];
    ac_otree tmp;
    ac_ot_init(&tmp);
    int cre = ac_ot_leaf(&tmp, AC_OP_CRE, tr->n[x->left].shell, l);
    int ann = ac_ot_leaf(&tmp, AC_OP_ANN, tr->n[x->left].shell, l);
    int top = ac_ot_node(&tmp, cre, ann, x->lab);
    ac_lab b = tmp.n[cre].lab;
    double v = -phase2(2 * lab_e2(b) - lab_e2(x->lab)) * shell_me(c, &tmp, top, l, f, g);
    if (x->lab.ts == 0 && x->lab.tl == 0 && f->code == g->code && f->nel == g->nel)
        v += sqrt(lab_dim(b)) * sqrt(lab_dim(f->lab));
    return v;
}

/* ---- memo of the single-shell elements ---------------------------------
   <f~ h g> of a coupled subtree h depends only on the subtree (the kinds
   and labels of its nodes in pre-order), l and the two shell states, and a
   CI asks for the same few thousand values tens of millions of times. They
   are kept per thread, keyed exactly (no value is ever taken from a
   different key): word 0 holds the shell states, l and the node count,
   words 1-3 three bytes per node, word 4 the labels of the two states. Subtrees of more than SME_NODES nodes or
   with labels above 255 are computed as before. The values are those of
   shell_me_sum, bit for bit. */
#define SME_NODES 8

#define SME_WORDS 5

typedef struct {
    uint64_t k[SME_WORDS];
    double v;
} sme_entry;

static _Thread_local sme_entry *sme_tab;
static _Thread_local size_t sme_cap, sme_len;
static _Thread_local const ac_cfptab *sme_t;

/* pre-order (kind, 2S, 2L) of the subtree; the node count, or -1 */
static int sme_encode(const ac_otree *tr, int node, unsigned char *out, int n)
{
    const ac_onode *x = &tr->n[node];
    if (n >= SME_NODES || x->lab.ts < 0 || x->lab.ts > 255 || x->lab.tl < 0 || x->lab.tl > 255)
        return -1;
    out[3 * n] = (unsigned char)x->kind;
    out[3 * n + 1] = (unsigned char)x->lab.ts;
    out[3 * n + 2] = (unsigned char)x->lab.tl;
    n++;
    if (x->kind != AC_OP_NODE)
        return n;
    n = sme_encode(tr, x->left, out, n);
    return n < 0 ? -1 : sme_encode(tr, x->right, out, n);
}

static int sme_key(const ac_otree *tr, int node, int l, const sstate *f, const sstate *g,
                   uint64_t *k)
{
    unsigned char sub[3 * SME_NODES] = {0};
    int nn = sme_encode(tr, node, sub, 0);
    if (nn < 0 || f->code < 0 || f->code >= 1 << 20 || g->code < 0 || g->code >= 1 << 20 ||
        f->nel < 0 || f->nel > 63 || g->nel < 0 || g->nel > 63 || l < 0 || l > 15 ||
        f->lab.ts < 0 || f->lab.ts > 65535 || f->lab.tl < 0 || f->lab.tl > 65535 ||
        g->lab.ts < 0 || g->lab.ts > 65535 || g->lab.tl < 0 || g->lab.tl > 65535)
        return -1;
    k[0] = (uint64_t)f->code | (uint64_t)g->code << 20 | (uint64_t)f->nel << 40 |
           (uint64_t)g->nel << 46 | (uint64_t)l << 52 | (uint64_t)nn << 56;
    memcpy(k + 1, sub, sizeof sub);
    k[4] = (uint64_t)f->lab.ts | (uint64_t)f->lab.tl << 16 | (uint64_t)g->lab.ts << 32 |
           (uint64_t)g->lab.tl << 48;
    return 0;
}

static size_t sme_hash(const uint64_t *k)
{
    uint64_t h = k[0] * 0x9E3779B97F4A7C15ULL;
    for (int i = 1; i < SME_WORDS; i++)
        h = (h ^ (h >> 29) ^ k[i]) * 0xBF58476D1CE4E5B9ULL;
    return (size_t)(h ^ (h >> 32));
}

/* the slot of the key (its k[0] is 0 when the key is new), or NULL */
static sme_entry *sme_slot(const ac_cfptab *t, const uint64_t *k)
{
    if (t != sme_t) { /* another CFP table: its values are not these */
        if (sme_tab)
            memset(sme_tab, 0, sme_cap * sizeof *sme_tab);
        sme_len = 0;
        sme_t = t;
    }
    if (2 * (sme_len + 1) > sme_cap) {
        size_t ncap = sme_cap ? 2 * sme_cap : 1 << 12;
        sme_entry *nt = calloc(ncap, sizeof *nt);
        if (!nt)
            return NULL;
        for (size_t i = 0; i < sme_cap; i++)
            if (sme_tab[i].k[0]) {
                size_t j = sme_hash(sme_tab[i].k) & (ncap - 1);
                while (nt[j].k[0])
                    j = (j + 1) & (ncap - 1);
                nt[j] = sme_tab[i];
            }
        free(sme_tab);
        sme_tab = nt;
        sme_cap = ncap;
    }
    size_t j = sme_hash(k) & (sme_cap - 1);
    while (sme_tab[j].k[0] && memcmp(sme_tab[j].k, k, sizeof sme_tab[j].k))
        j = (j + 1) & (sme_cap - 1);
    return &sme_tab[j];
}

static double shell_me_sum(ctx *c, const ac_otree *tr, int node, int l,
                           const sstate *f, const sstate *g);

/* <f~ h g> for the subtree h at `node`, all operators in one shell of l */
static double shell_me(ctx *c, const ac_otree *tr, int node, int l,
                       const sstate *f, const sstate *g)
{
    const ac_onode *x = &tr->n[node];
    switch (x->kind) {
    case AC_OP_ONE:
        return (f->code == g->code && f->nel == g->nel) ? sqrt(lab_dim(f->lab)) : 0.0;
    case AC_OP_CRE:
        if (f->nel != g->nel + 1)
            return 0.0;
        return ac_cfp_xcfp(c->t, f->code, g->code, g->nel, 1);
    case AC_OP_ANN:
        /* METECC (34): <f~ b~ g> = (-1)^{f-g+b} <g~ b f> */
        if (f->nel != g->nel - 1)
            return 0.0;
        return phase2(lab_e2(f->lab) - lab_e2(g->lab) + lab_e2(x->lab)) *
               ac_cfp_xcfp(c->t, g->code, f->code, f->nel, 1);
    default:
        break;
    }
    uint64_t key[SME_WORDS];
    sme_entry *e = sme_key(tr, node, l, f, g, key) ? NULL : sme_slot(c->t, key);
    if (e && e->k[0])
        return e->v;
    int err0 = c->err;
    c->err = 0;
    double v = shell_me_sum(c, tr, node, l, f, g);
    if (!c->err) {
        /* the sum below may have grown the table: find the slot again */
        e = e ? sme_slot(c->t, key) : NULL;
        if (e && !e->k[0]) {
            memcpy(e->k, key, sizeof e->k);
            e->v = v;
            sme_len++;
        }
        c->err = err0;
    }
    return v;
}

/* the coupled subtree at `node` (an internal node) */
static double shell_me_sum(ctx *c, const ac_otree *tr, int node, int l,
                           const sstate *f, const sstate *g)
{
    const ac_onode *x = &tr->n[node];
    /* METECC (48): sqrt[h] sum_k (-1)^{h1+h2+f+g} {h1 h2 h; g f k}
                    <f~ h1 k> <k~ h2 g> */
    int h1 = x->left, h2 = x->right;
    ac_lab H1 = tr->n[h1].lab, H2 = tr->n[h2].lab, H = x->lab;
    int kn = g->nel + ac_ot_nops(tr, h2);
    if (kn < 0 || kn > 4 * l + 2)
        return 0.0; /* no such intermediate state: h2 annihilates g */
    const int *codes;
    int nk = ac_cfp_states(c->t, l, kn, &codes);
    if (nk < 0) {
        if (tr->n[h1].kind == AC_OP_ANN && tr->n[h2].kind == AC_OP_CRE)
            return ann_cre_by_anticommutation(c, tr, node, l, f, g);
        c->err = AC_FACTOR_ERR_STATES;
        return 0.0;
    }
    double ph = phase2(lab_e2(H1) + lab_e2(H2) + lab_e2(f->lab) + lab_e2(g->lab));
    double sum = 0.0;
    for (int i = 0; i < nk; i++) {
        ac_shell_qn q;
        if (ac_cfp_decode(c->t, codes[i], &q)) {
            c->err = AC_FACTOR_ERR_CSF;
            return 0.0;
        }
        sstate k = {codes[i], kn, {q.ts, q.tL}};
        double w = lab_sixj(H1, H2, H, g->lab, f->lab, k.lab);
        if (w == 0.0)
            continue;
        double a = shell_me(c, tr, h1, l, f, &k);
        if (a == 0.0)
            continue;
        sum += w * a * shell_me(c, tr, h2, l, &k, g);
    }
    return ph * sqrt(lab_dim(H)) * sum;
}

/* one term in sequential form: the product of METECC (51) over all shells */
static double factor_term(ctx *c, const ac_otree *tr, const ac_csf *bra,
                          const ac_csf *ket)
{
    int M = ket->nsh;
    int hnode[AC_CSF_MAXSH + 1]; /* subtree of shell k, or -1 */
    ac_lab Hk[AC_CSF_MAXSH + 1]; /* label of H_k where shell k has operators */
    for (int k = 0; k <= M; k++)
        hnode[k] = -1;

    /* walk (G_1 (G_2 (... G_m))): a node whose leaves are one shell is a group */
    int x = tr->root;
    for (;;) {
        int lv[AC_OT_MAXNODE];
        int n = ac_ot_leaves(tr, x, lv);
        int sh = tr->n[lv[0]].shell, one = 1;
        for (int i = 1; i < n; i++)
            if (tr->n[lv[i]].shell != sh)
                one = 0;
        if (one) {
            if (sh < 1 || sh > M) {
                c->err = AC_FACTOR_ERR_TREE;
                return 0.0;
            }
            hnode[sh] = x;
            Hk[sh] = tr->n[x].lab;
            break;
        }
        int g = tr->n[x].left;
        int gl[AC_OT_MAXNODE];
        ac_ot_leaves(tr, g, gl);
        int gs = tr->n[gl[0]].shell;
        if (gs < 1 || gs > M) {
            c->err = AC_FACTOR_ERR_TREE;
            return 0.0;
        }
        hnode[gs] = g;
        Hk[gs] = tr->n[x].lab;
        x = tr->n[x].right;
    }

    ac_lab one = {0, 0};
    ac_lab Hprev = one;
    int nprev = 0; /* electron-number change of the operators in shells < k */
    double prod = 1.0;
    for (int k = 1; k <= M; k++) {
        ac_lab h = hnode[k] >= 0 ? tr->n[hnode[k]].lab : one;
        ac_lab Hcur = hnode[k] >= 0 ? Hk[k] : Hprev;
        ac_lab gk = ket->f[k], fk = bra->f[k];
        ac_lab Gp = ket->cpl[k - 1], Gc = ket->cpl[k];
        ac_lab Fp = bra->cpl[k - 1], Fc = bra->cpl[k];

        double w9 = lab_ninej(h, Hprev, Hcur, gk, Gp, Gc, fk, Fp, Fc);
        if (w9 == 0.0)
            return 0.0;
        sstate f = {bra->code[k], bra->nel[k], fk};
        sstate g = {ket->code[k], ket->nel[k], gk};
        double me;
        if (hnode[k] >= 0)
            me = shell_me(c, tr, hnode[k], ket->l[k], &f, &g);
        else
            me = (f.code == g.code && f.nel == g.nel) ? sqrt(lab_dim(fk)) : 0.0;
        if (me == 0.0)
            return 0.0;
        double sign = ((ket->nel[k] * nprev) & 1) ? -1.0 : 1.0;
        prod *= sign * sqrt(lab_dim(Hcur) * lab_dim(Gc) * lab_dim(Fc)) * w9 * me;

        if (hnode[k] >= 0)
            nprev += ac_ot_nops(tr, hnode[k]);
        Hprev = Hcur;
    }
    return prod;
}

/* ---- the shell groups of a sequential term, found once ----------------
   factor_term walks the tree of its term on every call to find which shell
   carries which subtree. For the terms of a cached sequential form that is
   done once (plan_term): the groups in increasing shell order, each with
   its subtree, the label H_k and its electron-number change. factor_plan
   then does the arithmetic of factor_term, in the same order. A term whose
   groups do not fit (more than PLAN_NG, a shell twice or out of range) has
   ng = -1 and goes through factor_term. */
#define PLAN_NG 8

typedef struct {
    signed char ng;
    signed char sh[PLAN_NG], node[PLAN_NG], nops[PLAN_NG];
    ac_lab H[PLAN_NG];
} term_plan;

static void plan_term(const ac_otree *tr, term_plan *p)
{
    int ng = 0, x = tr->root;
    p->ng = -1;
    for (;;) {
        int lv[AC_OT_MAXNODE];
        int n = ac_ot_leaves(tr, x, lv);
        int sh = tr->n[lv[0]].shell, one = 1, node = x;
        for (int i = 1; i < n; i++)
            if (tr->n[lv[i]].shell != sh)
                one = 0;
        if (!one) {
            node = tr->n[x].left;
            int gl[AC_OT_MAXNODE];
            ac_ot_leaves(tr, node, gl);
            sh = tr->n[gl[0]].shell;
        }
        if (ng == PLAN_NG || sh < 1 || sh > AC_CSF_MAXSH)
            return;
        for (int i = 0; i < ng; i++)
            if (p->sh[i] == sh)
                return;
        p->sh[ng] = (signed char)sh;
        p->node[ng] = (signed char)node;
        p->H[ng] = tr->n[x].lab;
        p->nops[ng] = (signed char)ac_ot_nops(tr, node);
        ng++;
        if (one)
            break;
        x = tr->n[x].right;
    }
    for (int i = 1; i < ng; i++) /* increasing shell */
        for (int j = i; j > 0 && p->sh[j - 1] > p->sh[j]; j--) {
            signed char t;
            ac_lab h = p->H[j];
            p->H[j] = p->H[j - 1], p->H[j - 1] = h;
            t = p->sh[j], p->sh[j] = p->sh[j - 1], p->sh[j - 1] = t;
            t = p->node[j], p->node[j] = p->node[j - 1], p->node[j - 1] = t;
            t = p->nops[j], p->nops[j] = p->nops[j - 1], p->nops[j - 1] = t;
        }
    p->ng = (signed char)ng;
}

static double factor_plan(ctx *c, const ac_otree *tr, const term_plan *p, const ac_csf *bra,
                          const ac_csf *ket)
{
    if (p->ng < 0)
        return factor_term(c, tr, bra, ket);
    int M = ket->nsh, gi = 0;
    if (p->sh[p->ng - 1] > M) {
        c->err = AC_FACTOR_ERR_TREE;
        return 0.0;
    }
    ac_lab one = {0, 0};
    ac_lab Hprev = one;
    int nprev = 0; /* electron-number change of the operators in shells < k */
    double prod = 1.0;
    for (int k = 1; k <= M; k++) {
        int grp = gi < p->ng && p->sh[gi] == k;
        ac_lab h = grp ? tr->n[p->node[gi]].lab : one;
        ac_lab Hcur = grp ? p->H[gi] : Hprev;
        ac_lab gk = ket->f[k], fk = bra->f[k];
        ac_lab Gp = ket->cpl[k - 1], Gc = ket->cpl[k];
        ac_lab Fp = bra->cpl[k - 1], Fc = bra->cpl[k];

        /* A shell without operators where every label is (0, 0) (the
           closed core before the first operator, typically): the 9j of
           zeros is 1, [H][G][F] = 1 and the shell element is 1 or 0, and
           the sign is + (a scalar shell holds an even number of
           electrons). The factor is exactly 1, or the term vanishes. */
        if (!grp && !(Hprev.ts | Hprev.tl | gk.ts | gk.tl | fk.ts | fk.tl | Gp.ts | Gp.tl |
                      Gc.ts | Gc.tl | Fp.ts | Fp.tl | Fc.ts | Fc.tl)) {
            if (bra->code[k] != ket->code[k] || bra->nel[k] != ket->nel[k])
                return 0.0;
            continue;
        }

        double w9 = lab_ninej(h, Hprev, Hcur, gk, Gp, Gc, fk, Fp, Fc);
        if (w9 == 0.0)
            return 0.0;
        sstate f = {bra->code[k], bra->nel[k], fk};
        sstate g = {ket->code[k], ket->nel[k], gk};
        double me;
        if (grp)
            me = shell_me(c, tr, p->node[gi], ket->l[k], &f, &g);
        else
            me = (f.code == g.code && f.nel == g.nel) ? sqrt(lab_dim(fk)) : 0.0;
        if (me == 0.0)
            return 0.0;
        double sign = ((ket->nel[k] * nprev) & 1) ? -1.0 : 1.0;
        prod *= sign * sqrt(lab_dim(Hcur) * lab_dim(Gc) * lab_dim(Fc)) * w9 * me;

        if (grp)
            nprev += p->nops[gi++];
        Hprev = Hcur;
    }
    return prod;
}

double ac_factor_invariant(const ac_cfptab *t, const ac_csf *bra,
                           const ac_csf *ket, ac_oexpr *e, int *err)
{
    ctx c = {t, 0};
    *err = 0;
    if (bra->nsh != ket->nsh) {
        *err = AC_FACTOR_ERR_CSF;
        return 0.0;
    }
    if (ac_oexpr_sequential(e)) {
        *err = AC_FACTOR_ERR_TREE;
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < e->len && !c.err; i++)
        sum += e->coef[i] * factor_term(&c, &e->t[i], bra, ket);
    *err = c.err;
    return c.err ? 0.0 : sum;
}

/* ---- cache of sequential forms, keyed by the operator tree ------------- */

typedef struct {
    ac_otree key;
    int normal;
    int used;
    ac_oexpr seq;  /* sequential form of the operator          */
    ac_oexpr cseq; /* of its contraction (len 0 if none)        */
    term_plan *pseq, *pcseq; /* the plans of their terms       */
} seq_entry;

static _Thread_local seq_entry *seq_tab;
static _Thread_local size_t seq_cap, seq_len;

/* The table grows with the variety of operator trees: the two-electron
   gauge terms of a large CI (Li 2P, 690 CSFs, l <= 4) reach 1.5 GB per
   thread. Above AC_SEQ_CACHE_MB per thread (default 128) all entries are
   dropped and recomputed on demand; the values do not change. The default
   was 512: with 8 threads B I 2P n = 5 with so2 ss oo gc2e peaked at 4.8 GB,
   against 1.3 GB and ~10% more time at 128. */
static _Thread_local size_t seq_bytes, seq_limit;

static void seq_flush(void)
{
    for (size_t i = 0; i < seq_cap; i++)
        if (seq_tab[i].used) {
            ac_oexpr_free(&seq_tab[i].seq);
            ac_oexpr_free(&seq_tab[i].cseq);
            free(seq_tab[i].pseq);
            free(seq_tab[i].pcseq);
            seq_tab[i].pseq = seq_tab[i].pcseq = NULL;
            seq_tab[i].used = 0;
        }
    seq_len = 0;
    seq_bytes = 0;
}

static int tree_same(const ac_otree *a, const ac_otree *b)
{
    if (a->nn != b->nn || a->root != b->root)
        return 0;
    for (int i = 0; i < a->nn; i++) {
        const ac_onode *x = &a->n[i], *y = &b->n[i];
        if (x->kind != y->kind || x->shell != y->shell || x->left != y->left ||
            x->right != y->right || x->lab.ts != y->lab.ts || x->lab.tl != y->lab.tl)
            return 0;
    }
    return 1;
}

static uint64_t tree_hash(const ac_otree *a, int normal)
{
    uint64_t h = 1469598103934665603ULL ^ (uint64_t)normal;
    for (int i = 0; i < a->nn; i++) {
        const ac_onode *x = &a->n[i];
        int v[6] = {x->kind, x->shell, x->left, x->right, x->lab.ts, x->lab.tl};
        for (int j = 0; j < 6; j++) {
            h ^= (uint64_t)(v[j] + 7);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

static seq_entry *seq_lookup(const ac_otree *tr, int normal, int *err)
{
    if (!seq_limit) {
        const char *s = getenv("AC_SEQ_CACHE_MB");
        long mb = s ? atol(s) : 0;
        seq_limit = (size_t)(mb > 0 ? mb : 128) << 20;
    }
    if (seq_bytes > seq_limit)
        seq_flush();
    if (2 * (seq_len + 1) > seq_cap) {
        size_t ncap = seq_cap ? 2 * seq_cap : 1024;
        seq_entry *nt = calloc(ncap, sizeof *nt);
        if (!nt) {
            *err = AC_FACTOR_ERR_TREE;
            return NULL;
        }
        for (size_t i = 0; i < seq_cap; i++)
            if (seq_tab[i].used) {
                size_t j = tree_hash(&seq_tab[i].key, seq_tab[i].normal) & (ncap - 1);
                while (nt[j].used)
                    j = (j + 1) & (ncap - 1);
                nt[j] = seq_tab[i];
            }
        free(seq_tab);
        seq_tab = nt;
        seq_cap = ncap;
    }
    size_t j = tree_hash(tr, normal) & (seq_cap - 1);
    while (seq_tab[j].used) {
        if (seq_tab[j].normal == normal && tree_same(&seq_tab[j].key, tr))
            return &seq_tab[j];
        j = (j + 1) & (seq_cap - 1);
    }
    seq_entry *e = &seq_tab[j];
    e->key = *tr;
    e->normal = normal;
    ac_oexpr_init(&e->seq);
    ac_oexpr_init(&e->cseq);
    if (ac_oexpr_push(&e->seq, 1.0, tr) || ac_oexpr_sequential(&e->seq)) {
        *err = AC_FACTOR_ERR_TREE;
        return NULL;
    }
    if (normal) {
        if (ac_oexpr_push(&e->cseq, 1.0, tr) || ac_oexpr_contraction(&e->cseq) ||
            ac_oexpr_sequential(&e->cseq)) {
            *err = AC_FACTOR_ERR_TREE;
            return NULL;
        }
    }
    e->pseq = malloc((size_t)(e->seq.len ? e->seq.len : 1) * sizeof *e->pseq);
    e->pcseq = malloc((size_t)(e->cseq.len ? e->cseq.len : 1) * sizeof *e->pcseq);
    if (!e->pseq || !e->pcseq) {
        free(e->pseq);
        free(e->pcseq);
        e->pseq = e->pcseq = NULL;
        ac_oexpr_free(&e->seq);
        ac_oexpr_free(&e->cseq);
        *err = AC_FACTOR_ERR_TREE;
        return NULL;
    }
    for (int i = 0; i < e->seq.len; i++)
        plan_term(&e->seq.t[i], &e->pseq[i]);
    for (int i = 0; i < e->cseq.len; i++)
        plan_term(&e->cseq.t[i], &e->pcseq[i]);
    e->used = 1;
    seq_len++;
    seq_bytes += (size_t)(e->seq.cap + e->cseq.cap) * (sizeof(ac_otree) + sizeof(double)) +
                 (size_t)(e->seq.len + e->cseq.len) * sizeof(term_plan);
    return e;
}

static double eval_seq(ctx *c, const ac_oexpr *e, const term_plan *p, const ac_csf *bra,
                       const ac_csf *ket)
{
    double sum = 0.0;
    for (int i = 0; i < e->len && !c->err; i++)
        sum += e->coef[i] * factor_plan(c, &e->t[i], &p[i], bra, ket);
    return sum;
}

double ac_factor_op(const ac_cfptab *t, const ac_csf *bra, const ac_csf *ket,
                    const ac_otree *tr, int normal, int *err)
{
    *err = 0;
    if (bra->nsh != ket->nsh) {
        *err = AC_FACTOR_ERR_CSF;
        return 0.0;
    }
    if (normal) {
        /* only a two-body tree whose b~_b and b_c share a shell contracts */
        const ac_onode *r = &tr->n[tr->root];
        normal = r->kind == AC_OP_NODE && tr->nn == 7 &&
                 tr->n[1].shell == tr->n[3].shell;
    }
    seq_entry *e = seq_lookup(tr, normal, err);
    if (!e)
        return 0.0;
    ctx c = {t, 0};
    double v = eval_seq(&c, &e->seq, e->pseq, bra, ket);
    if (normal && !c.err)
        v -= eval_seq(&c, &e->cseq, e->pcseq, bra, ket);
    *err = c.err;
    return c.err ? 0.0 : v;
}

/* ---- all the CSF pairs of two sets at once (ac_factor.h) --------------- */

static _Thread_local const ac_csf *set_base;

/* the first shell in which two CSFs of a set differ, or nsh + 1 */
static int set_diff(const ac_csf *x, const ac_csf *y)
{
    for (int k = 1; k <= x->nsh; k++)
        if (x->code[k] != y->code[k] || x->f[k].ts != y->f[k].ts || x->f[k].tl != y->f[k].tl ||
            x->cpl[k].ts != y->cpl[k].ts || x->cpl[k].tl != y->cpl[k].tl)
            return k;
    return x->nsh + 1;
}

static int set_cmp(const void *a, const void *b)
{
    int ia = *(const int *)a, ib = *(const int *)b;
    const ac_csf *x = &set_base[ia], *y = &set_base[ib];
    int k = set_diff(x, y);
    if (k > x->nsh)
        return (ia > ib) - (ia < ib);
    int vx[5] = {x->code[k], x->f[k].ts, x->f[k].tl, x->cpl[k].ts, x->cpl[k].tl};
    int vy[5] = {y->code[k], y->f[k].ts, y->f[k].tl, y->cpl[k].ts, y->cpl[k].tl};
    for (int i = 0; i < 5; i++)
        if (vx[i] != vy[i])
            return vx[i] < vy[i] ? -1 : 1;
    return 0;
}

static atomic_ulong set_ids;

int ac_csf_set_make(ac_csf_set *s, int n, const ac_csf *c)
{
    s->id = atomic_fetch_add(&set_ids, 1) + 1;
    s->n = n;
    s->nsh = n > 0 ? c[0].nsh : 0;
    s->c = c;
    s->ord = malloc((size_t)(n > 0 ? n : 1) * sizeof *s->ord);
    s->end = malloc((size_t)(n > 0 ? n : 1) * (size_t)(s->nsh > 0 ? s->nsh : 1) * sizeof *s->end);
    s->lid = malloc((size_t)(n > 0 ? n : 1) * (size_t)(s->nsh > 0 ? s->nsh : 1) * sizeof *s->lid);
    s->nl = malloc((size_t)(s->nsh > 0 ? s->nsh : 1) * sizeof *s->nl);
    int *d = malloc((size_t)(n > 0 ? n : 1) * sizeof *d);
    if (!s->ord || !s->end || !s->lid || !s->nl || !d) {
        free(d);
        ac_csf_set_free(s);
        return -1;
    }
    for (int p = 0; p < n; p++)
        s->ord[p] = p;
    set_base = c;
    qsort(s->ord, (size_t)n, sizeof *s->ord, set_cmp);
    /* d[p]: the first shell in which position p differs from position p - 1 */
    for (int p = 1; p < n; p++)
        d[p] = set_diff(&c[s->ord[p - 1]], &c[s->ord[p]]);
    for (int k = 1; k <= s->nsh; k++) {
        int *end = s->end + (size_t)(k - 1) * n, e = n;
        for (int p = n - 1; p >= 0; p--) {
            end[p] = e;
            if (p == 0 || d[p] <= k)
                e = p;
        }
    }
    /* the links of shell k; d[0 .. nl-1] holds a position of each */
    for (int k = 1; k <= s->nsh; k++) {
        int *lid = s->lid + (size_t)(k - 1) * n, nl = 0;
        for (int p = 0; p < n; p++) {
            const ac_csf *x = &c[s->ord[p]];
            int i;
            for (i = 0; i < nl; i++) {
                const ac_csf *y = &c[s->ord[d[i]]];
                if (x->code[k] == y->code[k] && x->f[k].ts == y->f[k].ts &&
                    x->f[k].tl == y->f[k].tl && x->cpl[k].ts == y->cpl[k].ts &&
                    x->cpl[k].tl == y->cpl[k].tl && x->cpl[k - 1].ts == y->cpl[k - 1].ts &&
                    x->cpl[k - 1].tl == y->cpl[k - 1].tl)
                    break;
            }
            if (i == nl)
                d[nl++] = p;
            lid[p] = i;
        }
        s->nl[k - 1] = nl;
    }
    free(d);
    return 0;
}

void ac_csf_set_free(ac_csf_set *s)
{
    free(s->ord);
    free(s->end);
    free(s->lid);
    free(s->nl);
    s->ord = s->end = s->lid = s->nl = NULL;
}

/* ---- the factors of the shells, kept for one pair of sets ---------------
   The factor of shell k of a term depends on the two links there and, from
   the term, only on the subtree the shell carries (none for a shell without
   operators), the labels H_{k-1} and H_k, and the parity that decides its
   sign. Terms of the same operator, and of the other operators evaluated on
   the same two sets, repeat these again and again, so the table of factors
   by (bra link, ket link) is kept under that key while the two sets stay the
   same (fc_bid, fc_kid). A value is NaN until it is computed; it is the
   number factor_plan multiplies into its product, or 0. Above fc_limit()
   doubles per thread everything is dropped and computed again on demand:
   128 MB, or AC_FC_LIMIT_MB megabytes. (With 64 MB the tables of H_SO2 on
   3d6 lists of Fe2+ were dropped 1800 times in a run and filled again, which
   was two thirds of the time.) */
static _Thread_local size_t fc_limit_v;

static size_t fc_limit(void)
{
    if (!fc_limit_v) {
        const char *s = getenv("AC_FC_LIMIT_MB");
        long mb = s ? atol(s) : 0;
        fc_limit_v = (size_t)(mb > 0 ? mb : 128) << 17;
    }
    return fc_limit_v;
}

typedef struct {
    uint64_t w[3];                  /* the subtree, as sme_encode gives it */
    int k, nn, par;                 /* shell; nodes of the subtree (0: no
                                       operators); parity of the sign      */
    int hp_ts, hp_tl, hc_ts, hc_tl; /* H_{k-1}, H_k                        */
    unsigned long uniq;             /* nonzero: a table of its own (a
                                       subtree that has no encoding)       */
} fc_key;

typedef struct {
    fc_key key;
    size_t off;
    unsigned gen;
} fc_entry;

static _Thread_local fc_entry *fc_tab;
static _Thread_local size_t fc_cap, fc_len;
static _Thread_local unsigned fc_gen;
static _Thread_local double *fc_v;
static _Thread_local size_t fc_vcap, fc_vlen;
static _Thread_local unsigned long fc_bid, fc_kid, fc_uniq;

static void fc_reset(void)
{
    fc_len = 0;
    fc_vlen = 0;
    if (++fc_gen == 0) { /* wrapped: no old entry may look valid */
        if (fc_tab)
            memset(fc_tab, 0, fc_cap * sizeof *fc_tab);
        fc_gen = 1;
    }
}

static size_t fc_hash(const fc_key *k)
{
    /* eight bytes at a time (the key is cleared before it is filled, so its
       padding is part of it); the rest, if any, byte by byte */
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)k;
    size_t i = 0;
    for (; i + 8 <= sizeof *k; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
    }
    for (; i < sizeof *k; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return (size_t)(h ^ (h >> 32));
}

/* the place in fc_v of the table of key (size doubles, NaN when new);
   (size_t)-1 when out of memory */
static size_t fc_get(const fc_key *key, size_t size)
{
    if (2 * (fc_len + 1) > fc_cap) {
        size_t ncap = fc_cap ? 2 * fc_cap : 1024;
        fc_entry *nt = calloc(ncap, sizeof *nt);
        if (!nt)
            return (size_t)-1;
        for (size_t i = 0; i < fc_cap; i++)
            if (fc_tab[i].gen == fc_gen) {
                size_t j = fc_hash(&fc_tab[i].key) & (ncap - 1);
                while (nt[j].gen == fc_gen)
                    j = (j + 1) & (ncap - 1);
                nt[j] = fc_tab[i];
            }
        free(fc_tab);
        fc_tab = nt;
        fc_cap = ncap;
    }
    size_t j = fc_hash(key) & (fc_cap - 1);
    while (fc_tab[j].gen == fc_gen) {
        if (!memcmp(&fc_tab[j].key, key, sizeof *key))
            return fc_tab[j].off;
        j = (j + 1) & (fc_cap - 1);
    }
    if (fc_vlen + size > fc_vcap) {
        size_t ncap = fc_vcap ? fc_vcap : 4096;
        while (ncap < fc_vlen + size)
            ncap *= 2;
        double *nv = realloc(fc_v, ncap * sizeof *nv);
        if (!nv)
            return (size_t)-1;
        fc_v = nv;
        fc_vcap = ncap;
    }
    for (size_t i = 0; i < size; i++)
        fc_v[fc_vlen + i] = NAN;
    fc_tab[j].key = *key;
    fc_tab[j].off = fc_vlen;
    fc_tab[j].gen = fc_gen;
    fc_len++;
    fc_vlen += size;
    return fc_tab[j].off;
}

/* one term over the two sets: what depends on the term alone, per shell */
typedef struct {
    ctx *c;
    const ac_otree *tr;
    const term_plan *p;
    const ac_csf_set *B, *K;
    int M;
    double coef, *acc;
    signed char g[AC_CSF_MAXSH + 2]; /* the group of shell k, or -1       */
    int np[AC_CSF_MAXSH + 2];        /* nprev of factor_plan at shell k   */
    ac_lab Hprev[AC_CSF_MAXSH + 2], Hcur[AC_CSF_MAXSH + 2];
    size_t off[AC_CSF_MAXSH + 2];    /* the factors of shell k in fv      */
    double *fv;
} blk;

/* 1 when (a, b, c) has even sums of 2S and of 2L and violates a triangle
   inequality (with odd sums the 9j itself decides) */
static int blk_tri_fails(ac_lab a, ac_lab b, ac_lab c)
{
    if (((a.ts + b.ts + c.ts) | (a.tl + b.tl + c.tl)) & 1)
        return 0;
    return c.ts > a.ts + b.ts || c.ts < (a.ts > b.ts ? a.ts - b.ts : b.ts - a.ts) ||
           c.tl > a.tl + b.tl || c.tl < (a.tl > b.tl ? a.tl - b.tl : b.tl - a.tl);
}

/* the runs [b0, b1) of the bra set and [g0, g1) of the ket set agree in the
   shells below k, and prod is the product of their factors there; the
   arithmetic of shell k is that of factor_plan */
static void blk_rec(blk *b, int k, int b0, int b1, int g0, int g1, double prod)
{
    const ac_csf_set *B = b->B, *K = b->K;
    if (k > b->M) {
        double v = b->coef * prod;
        for (int p = b0; p < b1; p++)
            for (int q = g0; q < g1; q++)
                b->acc[(size_t)B->ord[p] * (size_t)K->n + (size_t)K->ord[q]] += v;
        return;
    }
    int grp = b->g[k];
    ac_lab one = {0, 0};
    ac_lab h = grp >= 0 ? b->tr->n[b->p->node[grp]].lab : one;
    ac_lab Hprev = b->Hprev[k], Hcur = b->Hcur[k];
    const int *be = B->end + (size_t)(k - 1) * (size_t)B->n;
    const int *ke = K->end + (size_t)(k - 1) * (size_t)K->n;
    const int *bl = B->lid + (size_t)(k - 1) * (size_t)B->n;
    const int *kl = K->lid + (size_t)(k - 1) * (size_t)K->n;
    size_t nlk = (size_t)K->nl[k - 1];
    const double *fv = b->fv + b->off[k];
    for (int p = b0; p < b1; p = be[p]) {
        size_t row = (size_t)bl[p] * nlk;
        for (int q = g0; q < g1; q = ke[q]) {
            /* the factor of this pair of links: 0, or what factor_plan
               multiplies into its product (kept: see fc_key) */
            size_t x = row + (size_t)kl[q];
            double fac = fv[x];
            if (isnan(fac)) {
                const ac_csf *bra = &B->c[B->ord[p]], *ket = &K->c[K->ord[q]];
                ac_lab fk = bra->f[k], Fp = bra->cpl[k - 1], Fc = bra->cpl[k];
                ac_lab gk = ket->f[k], Gp = ket->cpl[k - 1], Gc = ket->cpl[k];
                fac = 0.0;
                /* a shell without operators where every label is (0, 0):
                   factor_plan multiplies nothing in, which is the factor 1
                   (the product times 1.0 is the product, bit for bit) */
                if (grp < 0 && !(Hprev.ts | Hprev.tl | gk.ts | gk.tl | fk.ts | fk.tl | Gp.ts |
                                 Gp.tl | Gc.ts | Gc.tl | Fp.ts | Fp.tl | Fc.ts | Fc.tl)) {
                    if (bra->code[k] == ket->code[k] && bra->nel[k] == ket->nel[k])
                        fac = 1.0;
                    b->fv[b->off[k] + x] = fac;
                    if (fac == 0.0)
                        continue;
                    blk_rec(b, k + 1, p, be[p], q, ke[q], prod * fac);
                    continue;
                }
                /* a column of the 9j that is no triangle: the symbol is 0
                   and is not looked up */
                double w9 = blk_tri_fails(h, gk, fk) || blk_tri_fails(Hcur, Gc, Fc)
                                ? 0.0
                                : lab_ninej(h, Hprev, Hcur, gk, Gp, Gc, fk, Fp, Fc);
                if (w9 != 0.0) {
                    sstate f = {bra->code[k], bra->nel[k], fk};
                    sstate g = {ket->code[k], ket->nel[k], gk};
                    double me;
                    if (grp >= 0)
                        me = shell_me(b->c, b->tr, b->p->node[grp], ket->l[k], &f, &g);
                    else
                        me = (f.code == g.code && f.nel == g.nel) ? sqrt(lab_dim(fk)) : 0.0;
                    if (me != 0.0) {
                        double sign = ((ket->nel[k] * b->np[k]) & 1) ? -1.0 : 1.0;
                        fac = sign * sqrt(lab_dim(Hcur) * lab_dim(Gc) * lab_dim(Fc)) * w9 * me;
                    }
                }
                b->fv[b->off[k] + x] = fac;
            }
            if (fac == 0.0)
                continue;
            blk_rec(b, k + 1, p, be[p], q, ke[q], prod * fac);
        }
    }
}

/* the factor of shell k of the term of b for the links of bra and ket: 0,
   or what factor_plan multiplies into its product (as in blk_rec) */
static double blk_factor(const blk *b, int k, const ac_csf *bra, const ac_csf *ket)
{
    int grp = b->g[k];
    ac_lab one = {0, 0};
    ac_lab h = grp >= 0 ? b->tr->n[b->p->node[grp]].lab : one;
    ac_lab Hprev = b->Hprev[k], Hcur = b->Hcur[k];
    ac_lab fk = bra->f[k], Fp = bra->cpl[k - 1], Fc = bra->cpl[k];
    ac_lab gk = ket->f[k], Gp = ket->cpl[k - 1], Gc = ket->cpl[k];
    if (grp < 0 && !(Hprev.ts | Hprev.tl | gk.ts | gk.tl | fk.ts | fk.tl | Gp.ts | Gp.tl | Gc.ts |
                     Gc.tl | Fp.ts | Fp.tl | Fc.ts | Fc.tl))
        return bra->code[k] == ket->code[k] && bra->nel[k] == ket->nel[k] ? 1.0 : 0.0;
    if (blk_tri_fails(h, gk, fk) || blk_tri_fails(Hcur, Gc, Fc))
        return 0.0;
    double w9 = lab_ninej(h, Hprev, Hcur, gk, Gp, Gc, fk, Fp, Fc);
    if (w9 == 0.0)
        return 0.0;
    sstate f = {bra->code[k], bra->nel[k], fk};
    sstate g = {ket->code[k], ket->nel[k], gk};
    double me;
    if (grp >= 0)
        me = shell_me(b->c, b->tr, b->p->node[grp], ket->l[k], &f, &g);
    else
        me = (f.code == g.code && f.nel == g.nel) ? sqrt(lab_dim(fk)) : 0.0;
    if (me == 0.0)
        return 0.0;
    double sign = ((ket->nel[k] * b->np[k]) & 1) ? -1.0 : 1.0;
    return sign * sqrt(lab_dim(Hcur) * lab_dim(Gc) * lab_dim(Fc)) * w9 * me;
}

/* what term i of e needs per shell (blk), with the places of its kept
   factors; 0, or -1 with the error set */
static int blk_setup(blk *b, ctx *c, const ac_oexpr *e, const term_plan *pl, int i,
                     const ac_csf_set *B, const ac_csf_set *K, double *acc)
{
    const term_plan *p = &pl[i];
    b->c = c;
    b->tr = &e->t[i];
    b->p = p;
    b->B = B;
    b->K = K;
    b->M = K->nsh;
    b->coef = e->coef[i];
    b->acc = acc;
    if (p->sh[p->ng - 1] > b->M) {
        c->err = AC_FACTOR_ERR_TREE;
        return -1;
    }
    ac_lab Hp = {0, 0};
    int nprev = 0, gi = 0;
    for (int k = 1; k <= b->M; k++) {
        int grp = gi < p->ng && p->sh[gi] == k;
        b->g[k] = (signed char)(grp ? gi : -1);
        b->np[k] = nprev;
        b->Hprev[k] = Hp;
        b->Hcur[k] = grp ? p->H[gi] : Hp;
        fc_key key;
        memset(&key, 0, sizeof key);
        key.k = k;
        key.par = (K->c[0].nel[k] * nprev) & 1;
        key.hp_ts = Hp.ts, key.hp_tl = Hp.tl;
        key.hc_ts = b->Hcur[k].ts, key.hc_tl = b->Hcur[k].tl;
        if (grp) {
            unsigned char sub[3 * SME_NODES] = {0};
            key.nn = sme_encode(&e->t[i], p->node[gi], sub, 0);
            if (key.nn < 0)
                key.uniq = ++fc_uniq;
            else
                memcpy(key.w, sub, sizeof key.w);
        }
        b->off[k] = fc_get(&key, (size_t)B->nl[k - 1] * (size_t)K->nl[k - 1]);
        if (b->off[k] == (size_t)-1) {
            c->err = AC_FACTOR_ERR_TREE;
            return -1;
        }
        if (grp)
            nprev += p->nops[gi++];
        Hp = b->Hcur[k];
    }
    return 0;
}

/* ---- all the terms of an expression in one descent ----------------------
   blk_rec walks the pairs of runs once per term, and nearly all of its time
   went into the calls of that walk (Ti: 340 per term). The terms of one
   expression walk the same pairs of runs, so they go down together: a node
   carries the terms whose product is not 0 so far, with their products
   (idx, prod, ascending in the term number), and a pair is reached once.
   Each product is formed as in blk_rec, and a pair adds its terms in the
   order of the term numbers, as the term-by-term walk does.
   At a shell many terms have the same table of factors: tid is the number
   of the table of a term among the distinct ones of the shell, toff the
   place of table j and trep a term that has it. For a pair of runs the
   factor of each table in use is read once (facj), and the terms take
   theirs from there. */
typedef struct {
    blk *b;       /* the terms */
    int nt, M;
    const ac_csf_set *B, *K;
    double *acc;
    const double *coef; /* [nt] */
    int *ibuf;    /* [(M + 2) * nt] the lists of the levels        */
    double *pbuf;
    int *tid;     /* [k * nt + t] the table of term t at shell k   */
    int *trep;    /* [k * nt + j] a term that has table j          */
    size_t *toff; /* [k * nt + j] the place of table j in fc_v     */
    int *tal;     /* [k * nt ..] scratch: the tables in use        */
    int *seen;    /* [k * nt + j] scratch: 0 between the uses      */
    double *facj; /* [k * nt + j] scratch: the factor of table j   */
} mterm;

static void mt_rec(const mterm *m, int k, int b0, int b1, int g0, int g1, int na, const int *idx,
                   const double *prod)
{
    const ac_csf_set *B = m->B, *K = m->K;
    if (k > m->M) {
        const double *coef = m->coef;
        for (int p = b0; p < b1; p++)
            for (int q = g0; q < g1; q++) {
                double a = m->acc[(size_t)B->ord[p] * (size_t)K->n + (size_t)K->ord[q]];
                for (int u = 0; u < na; u++)
                    a += coef[idx[u]] * prod[u];
                m->acc[(size_t)B->ord[p] * (size_t)K->n + (size_t)K->ord[q]] = a;
            }
        return;
    }
    size_t nt = (size_t)m->nt;
    const int *be = B->end + (size_t)(k - 1) * (size_t)B->n;
    const int *ke = K->end + (size_t)(k - 1) * (size_t)K->n;
    const int *bl = B->lid + (size_t)(k - 1) * (size_t)B->n;
    const int *kl = K->lid + (size_t)(k - 1) * (size_t)K->n;
    size_t nlk = (size_t)K->nl[k - 1];
    int *oi = m->ibuf + (size_t)k * nt;
    double *op = m->pbuf + (size_t)k * nt;
    const int *tid = m->tid + (size_t)k * nt, *trep = m->trep + (size_t)k * nt;
    const size_t *toff = m->toff + (size_t)k * nt;
    int *tal = m->tal + (size_t)k * nt, *seen = m->seen + (size_t)k * nt, ntal = 0;
    double *fj = m->facj + (size_t)k * nt;
    double *fv = m->b[0].fv;
    /* the tables of the terms that are still alive here */
    for (int u = 0; u < na; u++) {
        int j = tid[idx[u]];
        if (!seen[j])
            seen[j] = 1, tal[ntal++] = j;
    }
    for (int x = 0; x < ntal; x++)
        seen[tal[x]] = 0;
    for (int p = b0; p < b1; p = be[p]) {
        size_t row = (size_t)bl[p] * nlk;
        for (int q = g0; q < g1; q = ke[q]) {
            size_t x = row + (size_t)kl[q];
            int live = 0;
            for (int y = 0; y < ntal; y++) {
                int j = tal[y];
                double *slot = fv + toff[j] + x;
                double fac = *slot;
                if (isnan(fac))
                    fac = *slot = blk_factor(&m->b[trep[j]], k, &B->c[B->ord[p]], &K->c[K->ord[q]]);
                fj[j] = fac;
                live |= fac != 0.0;
            }
            if (!live)
                continue;
            int nout = 0;
            for (int u = 0; u < na; u++) {
                double fac = fj[tid[idx[u]]];
                if (fac != 0.0) {
                    oi[nout] = idx[u];
                    op[nout++] = prod[u] * fac;
                }
            }
            if (nout)
                mt_rec(m, k + 1, p, be[p], q, ke[q], nout, oi, op);
        }
    }
}

static _Thread_local blk *mt_b;
static _Thread_local int *mt_i;
static _Thread_local double *mt_p;
static _Thread_local size_t *mt_o;
static _Thread_local size_t mt_bcap, mt_lcap;

/* acc[i, j] += coef x term for all the pairs, in the order of eval_seq */
static void blk_seq(ctx *c, const ac_oexpr *e, const term_plan *pl, const ac_csf_set *B,
                    const ac_csf_set *K, double *acc)
{
    int nt = e->len, planned = nt > 0, M = K->nsh;
    for (int i = 0; i < nt; i++)
        planned = planned && pl[i].ng >= 0;
    /* the kept factors belong to these two sets */
    if (B->id != fc_bid || K->id != fc_kid || fc_vlen > fc_limit() || !fc_gen) {
        fc_reset();
        fc_bid = B->id;
        fc_kid = K->id;
    }
    size_t lev = ((size_t)M + 2) * (size_t)nt;
    if (planned) {
        if ((size_t)nt > mt_bcap) {
            blk *nb = realloc(mt_b, (size_t)nt * sizeof *nb);
            if (nb)
                mt_b = nb, mt_bcap = (size_t)nt;
        }
        if (lev > mt_lcap) {
            int *ni = realloc(mt_i, 5 * lev * sizeof *ni);
            if (ni)
                mt_i = ni;
            double *np = realloc(mt_p, 3 * lev * sizeof *np);
            if (np)
                mt_p = np;
            size_t *no = realloc(mt_o, lev * sizeof *no);
            if (no)
                mt_o = no;
            if (ni && np && no)
                mt_lcap = lev;
        }
        planned = (size_t)nt <= mt_bcap && lev <= mt_lcap;
    }
    if (planned) {
        for (int i = 0; i < nt; i++)
            if (blk_setup(&mt_b[i], c, e, pl, i, B, K, acc))
                return;
        double *coef = mt_p + 2 * lev;
        mterm m = {mt_b, nt, M, B, K, acc, coef, mt_i, mt_p, mt_i + lev, mt_i + 2 * lev, mt_o,
                   mt_i + 3 * lev, mt_i + 4 * lev, mt_p + lev};
        for (int i = 0; i < nt; i++) {
            mt_b[i].fv = fc_v; /* fc_v does not move once every table is there */
            coef[i] = e->coef[i];
            mt_i[i] = i;
            mt_p[i] = 1.0;
        }
        memset(m.seen, 0, lev * sizeof *m.seen);
        /* the distinct tables of each shell */
        for (int k = 1; k <= M; k++) {
            int *tid = m.tid + (size_t)k * (size_t)nt, *trep = m.trep + (size_t)k * (size_t)nt;
            size_t *toff = m.toff + (size_t)k * (size_t)nt;
            int ntab = 0;
            for (int i = 0; i < nt; i++) {
                int j;
                for (j = 0; j < ntab; j++)
                    if (toff[j] == mt_b[i].off[k])
                        break;
                if (j == ntab) {
                    trep[ntab] = i;
                    toff[ntab++] = mt_b[i].off[k];
                }
                tid[i] = j;
            }
        }
        mt_rec(&m, 1, 0, B->n, 0, K->n, nt, mt_i, mt_p);
        return;
    }
    /* a term without a plan (or no memory for the lists): term by term */
    for (int i = 0; i < nt && !c->err; i++) {
        const term_plan *p = &pl[i];
        if (p->ng < 0) { /* no plan: pair by pair */
            for (int x = 0; x < B->n; x++)
                for (int y = 0; y < K->n; y++)
                    acc[(size_t)x * (size_t)K->n + (size_t)y] +=
                        e->coef[i] * factor_term(c, &e->t[i], &B->c[x], &K->c[y]);
            continue;
        }
        blk b;
        if (blk_setup(&b, c, e, pl, i, B, K, acc))
            return;
        b.fv = fc_v;
        blk_rec(&b, 1, 0, B->n, 0, K->n, 1.0);
    }
}

void ac_factor_op_block(const ac_cfptab *t, const ac_csf_set *bra, const ac_csf_set *ket,
                        const ac_otree *tr, int normal, double *out, double *work, int *err)
{
    size_t nn = (size_t)bra->n * (size_t)ket->n;
    *err = 0;
    for (size_t x = 0; x < nn; x++)
        out[x] = 0.0;
    if (!nn)
        return;
    if (bra->nsh != ket->nsh) {
        *err = AC_FACTOR_ERR_CSF;
        return;
    }
    if (normal) {
        const ac_onode *r = &tr->n[tr->root];
        normal = r->kind == AC_OP_NODE && tr->nn == 7 && tr->n[1].shell == tr->n[3].shell;
    }
    seq_entry *e = seq_lookup(tr, normal, err);
    if (!e)
        return;
    ctx c = {t, 0};
    blk_seq(&c, &e->seq, e->pseq, bra, ket, out);
    if (normal && !c.err) {
        for (size_t x = 0; x < nn; x++)
            work[x] = 0.0;
        blk_seq(&c, &e->cseq, e->pcseq, bra, ket, work);
        for (size_t x = 0; x < nn; x++)
            out[x] -= work[x];
    }
    *err = c.err;
}

double ac_factor_shms(const ac_cfptab *t, int nht, const int32_t *nhlf,
                      const int32_t *nhwn, const int32_t *nhwq,
                      const int32_t *nhwcsl, const int32_t *nhvn,
                      const int32_t *nhvq, const int32_t *nhvcsl, int c1,
                      int a1, int lsi, int lri, int c2, int a2, int lsf,
                      int lrf, int lst, int lrt, int normal_order, int *err)
{
    ac_csf bra, ket;
    *err = 0;
    if (ac_csf_from_shms(&bra, t, nht, nhlf, nhwn, nhwq, nhwcsl) ||
        ac_csf_from_shms(&ket, t, nht, nhlf, nhvn, nhvq, nhvcsl)) {
        *err = AC_FACTOR_ERR_CSF;
        return 0.0;
    }
    if (c1 < 1 || c1 > nht || a1 < 1 || a1 > nht) {
        *err = AC_FACTOR_ERR_CSF;
        return 0.0;
    }
    ac_lab p = {lsi - 1, lri - 1};
    ac_otree tr;
    int two_body = c2 <= nht;
    if (two_body) {
        if (a2 < 1 || a2 > nht) {
            *err = AC_FACTOR_ERR_CSF;
            return 0.0;
        }
        ac_lab q = {lsf - 1, lrf - 1}, tot = {lst - 1, lrt - 1};
        ac_ot_two_body(&tr, c1, ket.l[c1], a1, ket.l[a1], p, c2, ket.l[c2], a2,
                       ket.l[a2], q, tot);
    } else {
        ac_ot_one_body(&tr, c1, ket.l[c1], a1, ket.l[a1], p);
    }

    ac_oexpr e;
    ac_oexpr_init(&e);
    ac_oexpr_push(&e, 1.0, &tr);
    double v = ac_factor_invariant(t, &bra, &ket, &e, err);
    ac_oexpr_free(&e);
    if (*err)
        return 0.0;

    if (two_body && normal_order && a1 == c2) {
        ac_oexpr_init(&e);
        ac_oexpr_push(&e, 1.0, &tr);
        if (ac_oexpr_contraction(&e)) {
            ac_oexpr_free(&e);
            *err = AC_FACTOR_ERR_TREE;
            return 0.0;
        }
        double cterm = ac_factor_invariant(t, &bra, &ket, &e, err);
        ac_oexpr_free(&e);
        if (*err)
            return 0.0;
        v -= cterm;
    }
    return v / sqrt(lab_dim(bra.cpl[nht]));
}
