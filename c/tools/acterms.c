/*
 * acterms.c -- every LS term of a set of configurations in one run (S2-M2,
 * check V2b).
 *
 * usage: acterms CFPTBL nfte nftri configs [nroots]
 *
 * The tape nfte supplies the orbital list, the default closed shells and the
 * integral addressing (any tape of the same orbitals will do); nftri the
 * transformed integrals. configs has one configuration per line as
 * "seq*occ seq*occ ..." with 1-based sequence numbers of the tape's orbital
 * list; default closed shells are added unless listed.
 *
 * All CSFs of all (S, L) are generated (ac_csfgen), H is built per (S, L)
 * block with the general-form engine and diagonalized. For each block the
 * output is "term 2S+1 L ncsf E1 E2 ...".
 */
#include "ac_cfp.h"
#include "ac_ci.h"
#include "ac_ftnio.h"
#include "ac_ints.h"
#include "ac_tape.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCFG 512

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s CFPTBL nfte nftri configs [nroots]\n", argv[0]);
        return 2;
    }
    int nroots = argc > 5 ? atoi(argv[5]) : 3;
    char err[256];
    ac_cfptab t;
    ac_ints ints;
    ac_nfte_head h;
    ac_ftn_file f;
    if (ac_cfp_load(&t, argv[1], err, sizeof err) ||
        ac_ints_load(&ints, argv[3], err, sizeof err)) {
        fprintf(stderr, "acterms: %s\n", err);
        return 1;
    }
    if (ac_ftn_open(&f, argv[2], "rb") || ac_nfte_read_head(&f, &h, err, sizeof err)) {
        fprintf(stderr, "acterms: %s: %s\n", argv[2], err);
        return 1;
    }
    ac_ftn_close(&f);

    static int seq[MAXCFG][AC_CSF_MAXSH], occ[MAXCFG][AC_CSF_MAXSH];
    int nocc[MAXCFG], ncfg = 0;
    FILE *cf = fopen(argv[4], "r");
    if (!cf) {
        perror(argv[4]);
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
    if (ac_ci_from_configs(&ci, &t, &ints, &h, ncfg, nocc,
                           (const int (*)[AC_CSF_MAXSH])seq,
                           (const int (*)[AC_CSF_MAXSH])occ, err, sizeof err)) {
        fprintf(stderr, "acterms: %s\n", err);
        return 1;
    }
    printf("acterms: %d configurations, %d CSFs\n", ncfg, ci.ncsf);

    /* blocks by final (2S, 2L) */
    int *done = calloc((size_t)ci.ncsf, sizeof *done), *idx = malloc((size_t)ci.ncsf * sizeof *idx);
    for (int a = 0; a < ci.ncsf; a++) {
        if (done[a])
            continue;
        const ac_csf *A = &ci.csf[a].c;
        ac_lab T = A->cpl[A->nsh];
        int n = 0;
        for (int b = a; b < ci.ncsf; b++) {
            const ac_csf *B = &ci.csf[b].c;
            if (!done[b] && B->cpl[B->nsh].ts == T.ts && B->cpl[B->nsh].tl == T.tl) {
                idx[n++] = b;
                done[b] = 1;
            }
        }
        double *H = calloc((size_t)n * n, sizeof *H), *w = malloc((size_t)n * sizeof *w);
        for (int j = 0; j < n; j++)
            for (int i = j; i < n; i++) {
                int e;
                H[(size_t)j * n + i] = ac_ci_hij(&ci, idx[i], idx[j], &e);
                if (e) {
                    fprintf(stderr, "acterms: error %d\n", e);
                    return 1;
                }
            }
        if (ac_ci_eigvals(n, H, w)) {
            fprintf(stderr, "acterms: dsyev failed\n");
            return 1;
        }
        printf("term %d %d %d", T.ts + 1, T.tl / 2, n);
        for (int r = 0; r < nroots && r < n; r++)
            printf(" %.15f", w[r]);
        printf("\n");
        free(H);
        free(w);
    }
    free(done);
    free(idx);
    ac_ci_free(&ci);
    ac_nfte_head_free(&h);
    ac_ints_free(&ints);
    ac_cfp_free(&t);
    return 0;
}
