/*
 * ac_optree.c -- operator trees and their recoupling (see ac_optree.h and
 * the atomci2 technical note, Sec. 3.5).
 */
#include "ac_optree.h"
#include "ac_labmath.h"

#include <stdlib.h>
#include <string.h>

/* ---- trees ---------------------------------------------------------- */

void ac_ot_init(ac_otree *t)
{
    memset(t, 0, sizeof *t);
    t->root = -1;
}

int ac_ot_leaf(ac_otree *t, int kind, int shell, int l)
{
    int i = t->nn++;
    ac_onode *n = &t->n[i];
    n->kind = (signed char)kind;
    n->shell = (signed char)shell;
    n->left = n->right = -1;
    if (kind == AC_OP_ONE) {
        n->lab.ts = 0;
        n->lab.tl = 0;
    } else {
        n->lab.ts = 1;
        n->lab.tl = 2 * l;
    }
    t->root = i;
    return i;
}

int ac_ot_node(ac_otree *t, int left, int right, ac_lab lab)
{
    int i = t->nn++;
    ac_onode *n = &t->n[i];
    n->kind = AC_OP_NODE;
    n->shell = 0;
    n->left = (signed char)left;
    n->right = (signed char)right;
    n->lab = lab;
    t->root = i;
    return i;
}

void ac_ot_two_body(ac_otree *t, int c1, int l_c1, int a1, int l_a1, ac_lab p,
                    int c2, int l_c2, int a2, int l_a2, ac_lab q, ac_lab tot)
{
    ac_ot_init(t);
    int x1 = ac_ot_leaf(t, AC_OP_CRE, c1, l_c1);
    int x2 = ac_ot_leaf(t, AC_OP_ANN, a1, l_a1);
    int np = ac_ot_node(t, x1, x2, p);
    int x3 = ac_ot_leaf(t, AC_OP_CRE, c2, l_c2);
    int x4 = ac_ot_leaf(t, AC_OP_ANN, a2, l_a2);
    int nq = ac_ot_node(t, x3, x4, q);
    ac_ot_node(t, np, nq, tot);
}

void ac_ot_one_body(ac_otree *t, int c, int l_c, int a, int l_a, ac_lab p)
{
    ac_ot_init(t);
    int x1 = ac_ot_leaf(t, AC_OP_CRE, c, l_c);
    int x2 = ac_ot_leaf(t, AC_OP_ANN, a, l_a);
    ac_ot_node(t, x1, x2, p);
}

int ac_ot_nops(const ac_otree *t, int node)
{
    const ac_onode *n = &t->n[node];
    switch (n->kind) {
    case AC_OP_CRE: return 1;
    case AC_OP_ANN: return -1;
    case AC_OP_ONE: return 0;
    default: return ac_ot_nops(t, n->left) + ac_ot_nops(t, n->right);
    }
}

/* number of fermion operators (parity of n in the swap phase) */
static int nferm(const ac_otree *t, int node)
{
    const ac_onode *n = &t->n[node];
    if (n->kind == AC_OP_CRE || n->kind == AC_OP_ANN)
        return 1;
    if (n->kind == AC_OP_ONE)
        return 0;
    return nferm(t, n->left) + nferm(t, n->right);
}

int ac_ot_leaves(const ac_otree *t, int node, int *out)
{
    const ac_onode *n = &t->n[node];
    if (n->kind != AC_OP_NODE) {
        out[0] = node;
        return 1;
    }
    int k = ac_ot_leaves(t, n->left, out);
    return k + ac_ot_leaves(t, n->right, out + k);
}

/* bit set of the shells under a node */
static unsigned long long shellset(const ac_otree *t, int node)
{
    const ac_onode *n = &t->n[node];
    if (n->kind == AC_OP_CRE || n->kind == AC_OP_ANN)
        return 1ULL << n->shell; /* shells run up to AC_CSF_MAXSH = 32 */
    if (n->kind == AC_OP_ONE)
        return 0;
    return shellset(t, n->left) | shellset(t, n->right);
}

static int tree_equal(const ac_otree *a, const ac_otree *b)
{
    if (a->nn != b->nn || a->root != b->root)
        return 0;
    for (int i = 0; i < a->nn; i++) {
        const ac_onode *x = &a->n[i], *y = &b->n[i];
        if (x->kind != y->kind || x->shell != y->shell || x->left != y->left ||
            x->right != y->right || !lab_eq(x->lab, y->lab))
            return 0;
    }
    return 1;
}

/* ---- expressions ------------------------------------------------------ */

void ac_oexpr_init(ac_oexpr *e)
{
    memset(e, 0, sizeof *e);
}

void ac_oexpr_free(ac_oexpr *e)
{
    free(e->coef);
    free(e->t);
    memset(e, 0, sizeof *e);
}

/* add a term, merging it into an equal tree already present */
int ac_oexpr_push(ac_oexpr *e, double coef, const ac_otree *t)
{
    if (coef == 0.0)
        return 0;
    for (int i = 0; i < e->len; i++)
        if (tree_equal(&e->t[i], t)) {
            e->coef[i] += coef;
            return 0;
        }
    if (e->len == e->cap) {
        int ncap = e->cap ? 2 * e->cap : 16;
        double *nc = realloc(e->coef, (size_t)ncap * sizeof *nc);
        if (!nc)
            return -1;
        e->coef = nc;
        ac_otree *nt = realloc(e->t, (size_t)ncap * sizeof *nt);
        if (!nt)
            return -1;
        e->t = nt;
        e->cap = ncap;
    }
    e->coef[e->len] = coef;
    e->t[e->len] = *t;
    e->len++;
    return 0;
}

static void replace(ac_oexpr *e, ac_oexpr *with)
{
    ac_oexpr_free(e);
    *e = *with;
}

int ac_oexpr_swap(ac_oexpr *e, int node)
{
    for (int i = 0; i < e->len; i++) {
        ac_otree *t = &e->t[i];
        ac_onode *x = &t->n[node];
        if (x->kind != AC_OP_NODE)
            return -1;
        int l = x->left, r = x->right;
        if (shellset(t, l) & shellset(t, r))
            return -1;
        double ph = phase2(lab_e2(t->n[l].lab) + lab_e2(t->n[r].lab) - lab_e2(x->lab));
        if ((nferm(t, l) * nferm(t, r)) & 1)
            ph = -ph;
        e->coef[i] *= ph;
        x->left = (signed char)r;
        x->right = (signed char)l;
    }
    return 0;
}

/* sum over every label G with triangle(b, c, G) and triangle(a, G, f) */
#define FOR_LABELS(G, b, c, a, f)                                              \
    for (G.ts = abs(b.ts - c.ts); G.ts <= b.ts + c.ts; G.ts += 2)              \
        for (G.tl = abs(b.tl - c.tl); G.tl <= b.tl + c.tl; G.tl += 2)          \
            if (lab_tri(a, G, f))

int ac_oexpr_rotate_right(ac_oexpr *e, int node)
{
    ac_oexpr out;
    ac_oexpr_init(&out);
    for (int i = 0; i < e->len; i++) {
        const ac_otree *t = &e->t[i];
        const ac_onode *x = &t->n[node];
        if (x->kind != AC_OP_NODE || t->n[x->left].kind != AC_OP_NODE)
            goto fail;
        int y = x->left;
        int na = t->n[y].left, nb = t->n[y].right, nc = x->right;
        ac_lab A = t->n[na].lab, B = t->n[nb].lab, C = t->n[nc].lab;
        ac_lab D = t->n[y].lab, F = x->lab, G;
        double ph = phase2(lab_e2(A) + lab_e2(B) + lab_e2(C) + lab_e2(F));
        FOR_LABELS(G, B, C, A, F)
        {
            double c = ph * sqrt(lab_dim(D) * lab_dim(G)) * lab_sixj(A, B, D, C, F, G);
            if (c == 0.0)
                continue;
            ac_otree nt = *t;
            nt.n[node].left = (signed char)na;
            nt.n[node].right = (signed char)y;
            nt.n[y].left = (signed char)nb;
            nt.n[y].right = (signed char)nc;
            nt.n[y].lab = G;
            if (ac_oexpr_push(&out, e->coef[i] * c, &nt))
                goto fail;
        }
    }
    replace(e, &out);
    return 0;
fail:
    ac_oexpr_free(&out);
    return -1;
}

int ac_oexpr_rotate_left(ac_oexpr *e, int node)
{
    ac_oexpr out;
    ac_oexpr_init(&out);
    for (int i = 0; i < e->len; i++) {
        const ac_otree *t = &e->t[i];
        const ac_onode *x = &t->n[node];
        if (x->kind != AC_OP_NODE || t->n[x->right].kind != AC_OP_NODE)
            goto fail;
        int y = x->right;
        int na = x->left, nb = t->n[y].left, nc = t->n[y].right;
        ac_lab A = t->n[na].lab, B = t->n[nb].lab, C = t->n[nc].lab;
        ac_lab G = t->n[y].lab, F = x->lab, D;
        double ph = phase2(lab_e2(A) + lab_e2(B) + lab_e2(C) + lab_e2(F));
        /* D couples (A, B) and then (D, C) to F */
        FOR_LABELS(D, A, B, C, F)
        {
            double c = ph * sqrt(lab_dim(D) * lab_dim(G)) * lab_sixj(A, B, D, C, F, G);
            if (c == 0.0)
                continue;
            ac_otree nt = *t;
            nt.n[node].left = (signed char)y;
            nt.n[node].right = (signed char)nc;
            nt.n[y].left = (signed char)na;
            nt.n[y].right = (signed char)nb;
            nt.n[y].lab = D;
            if (ac_oexpr_push(&out, e->coef[i] * c, &nt))
                goto fail;
        }
    }
    replace(e, &out);
    return 0;
fail:
    ac_oexpr_free(&out);
    return -1;
}

/* ---- canonical forms -------------------------------------------------- */

/* rotate left until no node under `node` has an internal right child */
static int to_left_comb(ac_oexpr *e, int node)
{
    if (e->len == 0)
        return 0;
    for (;;) {
        const ac_otree *t = &e->t[0];
        const ac_onode *x = &t->n[node];
        if (x->kind != AC_OP_NODE)
            return 0;
        if (t->n[x->right].kind == AC_OP_NODE) {
            if (ac_oexpr_rotate_left(e, node))
                return -1;
            if (e->len == 0)
                return 0;
            continue;
        }
        return to_left_comb(e, x->left);
    }
}

/* parent of a node, or -1 */
static int parent_of(const ac_otree *t, int node)
{
    for (int i = 0; i < t->nn; i++)
        if (t->n[i].kind == AC_OP_NODE && (t->n[i].left == node || t->n[i].right == node))
            return i;
    return -1;
}

/* in a left comb rooted at the tree root, exchange leaves i and i+1 */
static int swap_adjacent(ac_oexpr *e, int i)
{
    int lv[AC_OT_MAXNODE];
    const ac_otree *t = &e->t[0];
    ac_ot_leaves(t, t->root, lv);
    if (i == 0)
        return ac_oexpr_swap(e, parent_of(t, lv[0]));
    int p = parent_of(t, lv[i + 1]); /* p = ((R x_i) x_{i+1}) */
    int q = e->t[0].n[p].left;
    if (ac_oexpr_rotate_right(e, p)) /* (R (x_i x_{i+1})), inner node q */
        return -1;
    if (e->len == 0)
        return 0;
    if (ac_oexpr_swap(e, q))
        return -1;
    return ac_oexpr_rotate_left(e, p);
}

/* subtree at `node` is a left comb; make it (G_1 (G_2 (... G_m))) */
static int group_right_comb(ac_oexpr *e, int node)
{
    for (;;) {
        if (e->len == 0)
            return 0;
        const ac_otree *t = &e->t[0];
        int lv[AC_OT_MAXNODE];
        int n = ac_ot_leaves(t, node, lv);
        int s = 1;
        while (s < n && t->n[lv[s]].shell == t->n[lv[0]].shell)
            s++;
        if (s == n)
            return 0; /* a single group */
        int nl = ac_ot_leaves(t, t->n[node].left, lv);
        if (nl > s) {
            if (ac_oexpr_rotate_right(e, node))
                return -1;
            continue;
        }
        int r = t->n[node].right;
        if (to_left_comb(e, r))
            return -1;
        node = r;
    }
}

int ac_oexpr_sequential(ac_oexpr *e)
{
    if (e->len == 0)
        return 0;
    int root = e->t[0].root;
    if (to_left_comb(e, root))
        return -1;
    /* stable bubble sort by descending shell */
    for (int changed = 1; changed && e->len > 0;) {
        changed = 0;
        int lv[AC_OT_MAXNODE];
        int n = ac_ot_leaves(&e->t[0], root, lv);
        for (int i = 0; i + 1 < n; i++)
            if (e->t[0].n[lv[i]].shell < e->t[0].n[lv[i + 1]].shell) {
                if (swap_adjacent(e, i))
                    return -1;
                changed = 1;
                break;
            }
    }
    return group_right_comb(e, root);
}

/* ---- normal ordering -------------------------------------------------- */

/* replace node y = (L 1)^D or (1 R)^D by its non-identity child */
static int drop_identity(ac_oexpr *e, int y)
{
    ac_oexpr out;
    ac_oexpr_init(&out);
    for (int i = 0; i < e->len; i++) {
        ac_otree nt = e->t[i];
        ac_onode *x = &nt.n[y];
        int keep;
        if (nt.n[x->right].kind == AC_OP_ONE)
            keep = x->left;
        else if (nt.n[x->left].kind == AC_OP_ONE)
            keep = x->right;
        else
            goto fail;
        if (!lab_eq(nt.n[keep].lab, x->lab))
            continue; /* (X 1)^D vanishes unless D = X */
        *x = nt.n[keep];
        if (ac_oexpr_push(&out, e->coef[i], &nt))
            goto fail;
    }
    replace(e, &out);
    return 0;
fail:
    ac_oexpr_free(&out);
    return -1;
}

int ac_oexpr_contraction(ac_oexpr *e)
{
    if (e->len == 0)
        return 0;
    /* ((b1 b~2)^p (b3 b~4)^q)^T  ->  (((b1 b~2)^p b3)^D b~4)^T */
    int root = e->t[0].root;
    if (ac_oexpr_rotate_left(e, root))
        return -1;
    if (e->len == 0)
        return 0;
    int y = e->t[0].n[root].left; /* ((b1 b~2)^p b3)^D */
    int pnode = e->t[0].n[y].left;
    if (ac_oexpr_rotate_right(e, y)) /* (b1 (b~2 b3)^x)^D, inner = pnode */
        return -1;
    if (e->len == 0)
        return 0;

    /* (b~ b)^x = -(-1)^{2b-x} (b b~)^x + delta_{x0} sqrt[b]  (METECC eq. 15):
       the contraction keeps x = 0 with sqrt[b] in place of the pair */
    ac_oexpr out;
    ac_oexpr_init(&out);
    for (int i = 0; i < e->len; i++) {
        ac_otree nt = e->t[i];
        ac_onode *pn = &nt.n[pnode];
        const ac_onode *ann = &nt.n[pn->left], *cre = &nt.n[pn->right];
        if (ann->kind != AC_OP_ANN || cre->kind != AC_OP_CRE || ann->shell != cre->shell) {
            ac_oexpr_free(&out);
            return -1;
        }
        if (pn->lab.ts != 0 || pn->lab.tl != 0)
            continue;
        double c = sqrt(lab_dim(cre->lab));
        pn->kind = AC_OP_ONE;
        pn->shell = 0;
        pn->left = pn->right = -1;
        if (ac_oexpr_push(&out, e->coef[i] * c, &nt)) {
            ac_oexpr_free(&out);
            return -1;
        }
    }
    replace(e, &out);
    if (e->len == 0)
        return 0;
    return drop_identity(e, y);
}
