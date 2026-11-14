/*
 * terms_spaces.h -- the configuration spaces and pseudo-random integrals shared
 * by test_terms.c and ref/make_ref_hij.c. Changing them means regenerating
 * ref_hij.txt from the program the reference comes from (see ref/README.md).
 */
#ifndef TERMS_SPACES_H
#define TERMS_SPACES_H

typedef struct {
    const char *name;
    int nwt;
    int lorb[8];
    int ncfg;
    int occ[8][8];
} space;

static const space spaces[] = {
    {"s2p2+sp3+p4", 2, {0, 1}, 3, {{2, 2}, {1, 3}, {0, 4}}},
    {"1s2_2s2_2p+1s2_2s_2p2", 3, {0, 0, 1}, 2, {{2, 2, 1}, {2, 1, 2}}},
    {"s2d2+sd3+d4", 2, {0, 2}, 3, {{2, 2}, {1, 3}, {0, 4}}},
    {"four_distinct_orbitals", 5, {0, 0, 1, 1, 2}, 4,
     {{1, 1, 1, 1, 0}, {2, 0, 2, 0, 0}, {0, 2, 0, 2, 0}, {0, 0, 1, 1, 2}}},
    {"f3+f2d+f2s", 3, {3, 2, 0}, 3, {{3, 0, 0}, {2, 1, 0}, {2, 0, 1}}},
    {"sp2d_core", 4, {0, 0, 1, 2}, 3, {{2, 1, 2, 0}, {2, 1, 1, 1}, {2, 0, 2, 1}}},
    /* one-electron excitations within the same l: off-diagonal h(a,b) */
    {"s_to_s',p_to_p'", 4, {0, 0, 1, 1}, 4,
     {{2, 1, 1, 0}, {1, 2, 1, 0}, {2, 1, 0, 1}, {1, 2, 0, 1}}},
};

#define NSPACES ((int)(sizeof spaces / sizeof spaces[0]))

/* pseudo-random value of a canonical label, in (-1, 1), never 0 */
static inline double terms_hashval(int k, int a, int b, int c, int d)
{
    unsigned long long x = 1469598103934665603ULL;
    int key[5] = {k + 2, a, b, c, d};
    for (int i = 0; i < 5; i++) {
        x ^= (unsigned long long)key[i];
        x *= 1099511628211ULL;
        x ^= x >> 29;
    }
    x *= 1099511628211ULL;
    return ((double)(x >> 12) + 0.5) / 4503599627370496.0 * 2.0 - 1.0;
}

/* the label an integral equals for real orbitals */
static inline void terms_canon(int *k, int *a, int *b, int *c, int *d)
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

#endif
