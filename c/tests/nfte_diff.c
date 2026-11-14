/*
 * nfte_diff.c -- L4: compare two energy-expression tapes by meaning.
 *
 * usage: nfte_diff tapeA tapeB
 *
 * The header records and the CSF image must be byte-identical. The data
 * records are reduced to a map (row CSF, column CSF, integral address) ->
 * sum of coefficients, so record boundaries, record order and the order of
 * pairs inside records do not matter. Two maps agree when they have the
 * same keys and every value differs by at most TOL (relative to
 * max(1, |value|)); a key present on one side only counts as a difference
 * unless its value is below ZERO.
 */
#include "ac_ftnio.h"
#include "ac_tape.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-12
#define ZERO 1e-12

typedef struct {
    int32_t i, j, adr;
    double v;
} entry;

typedef struct {
    entry *e;
    long n, cap;
} emap;

static int cmp(const void *a, const void *b)
{
    const entry *x = a, *y = b;
    if (x->i != y->i) return x->i < y->i ? -1 : 1;
    if (x->j != y->j) return x->j < y->j ? -1 : 1;
    if (x->adr != y->adr) return x->adr < y->adr ? -1 : 1;
    return 0;
}

static int push(emap *m, int32_t i, int32_t j, int32_t adr, double v)
{
    if (m->n == m->cap) {
        long nc = m->cap ? 2 * m->cap : 1 << 16;
        entry *ne = realloc(m->e, (size_t)nc * sizeof *ne);
        if (!ne)
            return -1;
        m->e = ne;
        m->cap = nc;
    }
    m->e[m->n++] = (entry){i, j, adr, v};
    return 0;
}

/* read a tape: header records into h, data into a sorted, merged map */
static int load(const char *path, ac_nfte_head *h, emap *m, long *nrec)
{
    ac_ftn_file f;
    char err[256];
    if (ac_ftn_open(&f, path, "rb")) {
        fprintf(stderr, "cannot open %s\n", path);
        return -1;
    }
    if (ac_nfte_read_head(&f, h, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", path, err);
        return -1;
    }
    ac_nfte_rec r;
    memset(&r, 0, sizeof r);
    int rc;
    *nrec = 0;
    while ((rc = ac_nfte_read_rec(&f, &r)) == AC_FTN_OK) {
        if (r.noe == 0)
            continue;
        ++*nrec;
        for (int p = 0; p < r.nij; p++) {
            int32_t i = r.indexi[p], j = r.nii ? r.indexj[p] : r.indexi[p];
            for (int e = 0; e < r.noe; e++)
                if (push(m, i, j, r.isqn[p * r.noe + e], r.wt[e]))
                    return -1;
        }
    }
    ac_nfte_rec_free(&r);
    ac_ftn_close(&f);
    if (rc != AC_FTN_EOF) {
        fprintf(stderr, "%s: read error %d\n", path, rc);
        return -1;
    }
    qsort(m->e, (size_t)m->n, sizeof *m->e, cmp);
    long k = 0;
    for (long i = 0; i < m->n; i++) {
        if (k > 0 && cmp(&m->e[k - 1], &m->e[i]) == 0)
            m->e[k - 1].v += m->e[i].v;
        else
            m->e[k++] = m->e[i];
    }
    m->n = k;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s tapeA tapeB\n", argv[0]);
        return 2;
    }
    ac_nfte_head ha, hb;
    emap ma = {0}, mb = {0};
    long ra, rb;
    if (load(argv[1], &ha, &ma, &ra) || load(argv[2], &hb, &mb, &rb))
        return 2;

    long bad = 0;
    int same_head = ha.nraw == hb.nraw;
    for (int i = 0; same_head && i < ha.nraw; i++)
        same_head = ha.rawlen[i] == hb.rawlen[i] &&
                    !memcmp(ha.raw[i], hb.raw[i], ha.rawlen[i]);
    printf("header and CSF image (%d records): %s\n", ha.nraw,
           same_head ? "identical" : "DIFFERENT");
    if (!same_head)
        bad++;

    long ia = 0, ib = 0, only_a = 0, only_b = 0, both = 0, differ = 0, shown = 0;
    double maxd = 0.0;
    while (ia < ma.n || ib < mb.n) {
        int c = ia >= ma.n ? 1 : ib >= mb.n ? -1 : cmp(&ma.e[ia], &mb.e[ib]);
        if (c < 0) {
            if (fabs(ma.e[ia].v) >= ZERO) {
                only_a++;
                if (shown++ < 10)
                    fprintf(stderr, "only in A: (%d,%d) adr %d  %.15g\n", ma.e[ia].i,
                            ma.e[ia].j, ma.e[ia].adr, ma.e[ia].v);
            }
            ia++;
        } else if (c > 0) {
            if (fabs(mb.e[ib].v) >= ZERO) {
                only_b++;
                if (shown++ < 10)
                    fprintf(stderr, "only in B: (%d,%d) adr %d  %.15g\n", mb.e[ib].i,
                            mb.e[ib].j, mb.e[ib].adr, mb.e[ib].v);
            }
            ib++;
        } else {
            double d = fabs(ma.e[ia].v - mb.e[ib].v);
            double a = fabs(ma.e[ia].v);
            both++;
            if (d > maxd)
                maxd = d;
            if (d > TOL * (a > 1 ? a : 1)) {
                differ++;
                if (shown++ < 10)
                    fprintf(stderr, "value (%d,%d) adr %d: A %.15g B %.15g\n", ma.e[ia].i,
                            ma.e[ia].j, ma.e[ia].adr, ma.e[ia].v, mb.e[ib].v);
            }
            ia++;
            ib++;
        }
    }
    printf("A: %ld data records, %ld (row, column, address) entries\n", ra, ma.n);
    printf("B: %ld data records, %ld (row, column, address) entries\n", rb, mb.n);
    printf("common %ld, only A %ld, only B %ld, values differing %ld, max |A-B| = %.3e\n",
           both, only_a, only_b, differ, maxd);
    bad += only_a + only_b + differ;
    printf("L4 %s\n", bad ? "FAILED" : "passed");
    free(ma.e);
    free(mb.e);
    ac_nfte_head_free(&ha);
    ac_nfte_head_free(&hb);
    return bad ? 1 : 0;
}
