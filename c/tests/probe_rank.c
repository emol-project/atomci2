/*
 * probe_rank.c -- does ac_factor handle non-scalar total ranks?
 *
 * The total spin S = sum over shells of the one-body (b b~)^(1,0) (spin
 * rank 1, orbital rank 0) and the total orbital momentum L = sum of
 * (b b~)^(0,1) are generators: between CSFs of one configuration their
 * invariants must vanish unless bra = ket, and on the diagonal be
 *   <F~ S F> = c_S * sqrt(S(S+1)(2S+1)(2L+1)),
 *   <F~ L F> = c_L * sqrt(L(L+1)(2S+1)(2L+1)),
 * with c_S, c_L the same for every CSF. This program prints the ratios over
 * all CSFs (all shell states and intermediate couplings) of a few
 * configurations, and the largest off-diagonal invariant.
 *
 * usage: probe_rank CFPTBL
 */
#include "ac_cfp.h"
#include "ac_csf.h"
#include "ac_factor.h"
#include "ac_optree.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCSF 4096

typedef struct {
    int nsh;
    int l[4], n[4];
} config;

static ac_csf csfs[MAXCSF];
static int ncsf;

/* all CSFs of a configuration: every shell state and coupling */
static void gen(const ac_cfptab *t, const config *c, int k, ac_csf *cur)
{
    if (k > c->nsh) {
        cur->nsh = c->nsh;
        if (ncsf < MAXCSF)
            csfs[ncsf++] = *cur;
        return;
    }
    const int *codes;
    int ns = ac_cfp_states(t, c->l[k - 1], c->n[k - 1], &codes);
    for (int i = 0; i < ns; i++) {
        ac_shell_qn q;
        ac_cfp_decode(t, codes[i], &q);
        ac_lab prev = cur->cpl[k - 1];
        cur->l[k] = c->l[k - 1];
        cur->code[k] = codes[i];
        cur->nel[k] = c->n[k - 1];
        cur->f[k].ts = q.ts;
        cur->f[k].tl = q.tL;
        for (int ts = abs(q.ts - prev.ts); ts <= q.ts + prev.ts; ts += 2)
            for (int tl = abs(q.tL - prev.tl); tl <= q.tL + prev.tl; tl += 2) {
                cur->cpl[k].ts = ts;
                cur->cpl[k].tl = tl;
                gen(t, c, k + 1, cur);
            }
    }
}

/* invariant of sum_k w_k (b_k b~_k)^lab between two CSFs. One shell's
   (b b~)^(1,0) is its spin times a factor 1/sqrt[l] from the orbital
   coupling (l l)^0, and (b b~)^(0,1) its orbital momentum over
   sqrt(l(l+1)[l]); the weights w_k make the sums the total S and L. */
static double one_body_sum(const ac_cfptab *t, const ac_csf *bra, const ac_csf *ket,
                           ac_lab lab, int *err)
{
    double s = 0.0;
    for (int k = 1; k <= ket->nsh; k++) {
        double l = ket->l[k];
        double w = lab.tl == 0 ? sqrt(2 * l + 1) : sqrt(l * (l + 1) * (2 * l + 1));
        if (lab.ts == 0 && lab.tl == 0)
            w = 1.0;
        ac_otree tr;
        ac_ot_one_body(&tr, k, ket->l[k], k, ket->l[k], lab);
        ac_oexpr e;
        ac_oexpr_init(&e);
        ac_oexpr_push(&e, 1.0, &tr);
        s += w * ac_factor_invariant(t, bra, ket, &e, err);
        ac_oexpr_free(&e);
        if (*err)
            return 0.0;
    }
    return s;
}

int main(int argc, char **argv)
{
    ac_cfptab t;
    char msg[256];
    if (argc != 2 || ac_cfp_load(&t, argv[1], msg, sizeof msg)) {
        fprintf(stderr, "usage: %s CFPTBL\n", argv[0]);
        return 2;
    }
    const config cfgs[] = {
        {2, {1, 2}, {2, 1}}, /* p^2 d    */
        {2, {1, 2}, {1, 1}}, /* p d      */
        {3, {0, 1, 2}, {1, 3, 1}}, /* s p^3 d */
        {2, {2, 3}, {3, 2}}, /* d^3 f^2  */
    };
    ac_lab Sop = {2, 0}, Lop = {0, 2};
    if (getenv("PROBE_DEBUG")) { /* a few raw values for the first config */
        ac_csf cur = {0};
        ncsf = 0;
        gen(&t, &cfgs[0], 1, &cur);
        ac_lab one = {0, 0};
        for (int a = 0; a < 4 && a < ncsf; a++) {
            int err = 0;
            double n = one_body_sum(&t, &csfs[a], &csfs[a], one, &err);
            double s = one_body_sum(&t, &csfs[a], &csfs[a], Sop, &err);
            printf("csf %d: total (2S,2L) = (%d,%d)  scalar %.10f  S-op %.10f  err %d\n", a,
                   csfs[a].cpl[2].ts, csfs[a].cpl[2].tl, n, s, err);
        }
    }
    for (size_t c = 0; c < sizeof cfgs / sizeof *cfgs; c++) {
        ac_csf cur = {0};
        ncsf = 0;
        gen(&t, &cfgs[c], 1, &cur);
        double rmin[2] = {1e300, 1e300}, rmax[2] = {-1e300, -1e300}, offmax = 0.0;
        int err = 0, nratio[2] = {0, 0};
        for (int a = 0; a < ncsf && !err; a++)
            for (int b = 0; b < ncsf && !err; b++) {
                const ac_csf *F = &csfs[a], *G = &csfs[b];
                ac_lab FT = F->cpl[F->nsh], GT = G->cpl[G->nsh];
                if (FT.ts != GT.ts || FT.tl != GT.tl)
                    continue;
                for (int w = 0; w < 2; w++) {
                    double v = one_body_sum(&t, F, G, w ? Lop : Sop, &err);
                    if (a != b) {
                        if (fabs(v) > offmax)
                            offmax = fabs(v);
                        continue;
                    }
                    double S = FT.ts / 2.0, L = FT.tl / 2.0, x = w ? L : S;
                    double ref = sqrt(x * (x + 1) * (2 * S + 1) * (2 * L + 1));
                    if (ref == 0.0) {
                        if (fabs(v) > offmax)
                            offmax = fabs(v);
                        continue;
                    }
                    double r = v / ref;
                    if (r < rmin[w]) rmin[w] = r;
                    if (r > rmax[w]) rmax[w] = r;
                    nratio[w]++;
                }
            }
        printf("config %zu: %d CSFs, err %d\n", c + 1, ncsf, err);
        printf("  S: %d diagonal ratios in [%.15f, %.15f]\n", nratio[0], rmin[0], rmax[0]);
        printf("  L: %d diagonal ratios in [%.15f, %.15f]\n", nratio[1], rmin[1], rmax[1]);
        printf("  largest invariant that must vanish: %.3e\n", offmax);
    }
    ac_cfp_free(&t);
    return 0;
}
