/*
 * ac_factor.c -- factorized matrix elements (see ac_factor.h).
 */
#include "ac_factor.h"
#include "ac_labmath.h"

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
    e->used = 1;
    seq_len++;
    seq_bytes += (size_t)(e->seq.cap + e->cseq.cap) * (sizeof(ac_otree) + sizeof(double));
    return e;
}

static double eval_seq(ctx *c, const ac_oexpr *e, const ac_csf *bra, const ac_csf *ket)
{
    double sum = 0.0;
    for (int i = 0; i < e->len && !c->err; i++)
        sum += e->coef[i] * factor_term(c, &e->t[i], bra, ket);
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
    double v = eval_seq(&c, &e->seq, bra, ket);
    if (normal && !c.err)
        v -= eval_seq(&c, &e->cseq, bra, ket);
    *err = c.err;
    return c.err ? 0.0 : v;
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
