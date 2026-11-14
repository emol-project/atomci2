/*
 * ac_optree.h -- coupled products of shell operators and their recoupling.
 *
 * An operator is a binary tree: leaves are the creation tensor b or its
 * adjoint tensor b~ (METECC eq. 9) of one shell, or the identity; every
 * internal node couples its two children to a label (2S, 2L). An expression
 * is a linear combination of such trees with real coefficients.
 *
 * All terms of an expression always have the same tree shape (the same node
 * indices); only their labels differ. Every rewrite below acts on the same
 * node index in all terms, so the shape stays uniform.
 *
 * Rewrites (atomci2 technical note, Sec. 3.5):
 *   swap          (X Y)^C = (-1)^{X+Y-C} (-1)^{n(X)n(Y)} (Y X)^C,
 *                 only when X and Y have no shell in common;
 *   rotate_right  ((A B)^D C)^F = sum_G (-1)^{A+B+C+F} sqrt([D][G])
 *                 {A B D; C F G} (A (B C)^G)^F;
 *   rotate_left   its inverse (same coefficients).
 * A label exponent such as A+B-C means the sum of its spin and orbital parts.
 */
#ifndef AC_OPTREE_H
#define AC_OPTREE_H

#include "ac_csf.h"

enum { AC_OP_NODE = 0, AC_OP_CRE = 1, AC_OP_ANN = 2, AC_OP_ONE = 3 };

#define AC_OT_MAXNODE 16

typedef struct {
    signed char kind;  /* AC_OP_NODE or a leaf kind                 */
    signed char shell; /* leaves: 1-based shell in coupling order   */
    signed char left;  /* internal nodes: child indices             */
    signed char right;
    ac_lab lab;        /* leaves: (1, 2l) for b and b~, (0,0) for 1 */
} ac_onode;

typedef struct {
    ac_onode n[AC_OT_MAXNODE];
    int nn;
    int root;
} ac_otree;

typedef struct {
    int len, cap;
    double *coef;
    ac_otree *t;
} ac_oexpr;

/* ---- building trees ---- */
void ac_ot_init(ac_otree *t);
int ac_ot_leaf(ac_otree *t, int kind, int shell, int l);
int ac_ot_node(ac_otree *t, int left, int right, ac_lab lab);

/* the two-body operator ((b_c1 b~_a1)^p (b_c2 b~_a2)^q)^tot */
void ac_ot_two_body(ac_otree *t, int c1, int l_c1, int a1, int l_a1, ac_lab p,
                    int c2, int l_c2, int a2, int l_a2, ac_lab q, ac_lab tot);
/* the one-body operator (b_c b~_a)^p */
void ac_ot_one_body(ac_otree *t, int c, int l_c, int a, int l_a, ac_lab p);

/* ---- expressions ---- */
void ac_oexpr_init(ac_oexpr *e);
void ac_oexpr_free(ac_oexpr *e);
int ac_oexpr_push(ac_oexpr *e, double coef, const ac_otree *t);

/* rewrites at a node index; return 0, or -1 on misuse (wrong node type,
   shells shared in a swap, out of memory) */
int ac_oexpr_swap(ac_oexpr *e, int node);
int ac_oexpr_rotate_right(ac_oexpr *e, int node);
int ac_oexpr_rotate_left(ac_oexpr *e, int node);

/* Bring every term to the sequential form of METECC eq. (49): leaves sorted
   by descending shell (stable, so operators of one shell keep their order),
   coupled as (G_1 (G_2 (... G_m))) with G_i the subtree of one shell. */
int ac_oexpr_sequential(ac_oexpr *e);

/* Normal ordering of the two-body operator built by ac_ot_two_body():
   replaces e by its contraction term, i.e. the part that :O: = O - C drops.
   Requires a1 and c2 to be the same shell; e must hold only that operator. */
int ac_oexpr_contraction(ac_oexpr *e);

/* helpers */
int ac_ot_nops(const ac_otree *t, int node);      /* #b - #b~ under node */
int ac_ot_leaves(const ac_otree *t, int node, int *out); /* in order */

#endif
