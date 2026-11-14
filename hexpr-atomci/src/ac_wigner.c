/*
 * ac_wigner.c -- 6j/9j symbols and the Coulomb angular factors.
 * See ac_wigner.h for conventions (all angular momenta as 2j).
 */
#include "ac_wigner.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* fac[n] = n!, sqfac[n] = sqrt(n!) for n = 0..AC_WIGNER_FACMAX. */
static _Thread_local double fac[AC_WIGNER_FACMAX + 1];
static _Thread_local double sqfac[AC_WIGNER_FACMAX + 1];
static _Thread_local int tables_ready;

static void init_tables(void)
{
    fac[0] = 1.0;
    for (int n = 1; n <= AC_WIGNER_FACMAX; n++)
        fac[n] = fac[n - 1] * n;
    for (int n = 0; n <= AC_WIGNER_FACMAX; n++)
        sqfac[n] = sqrt(fac[n]);
    tables_ready = 1;
}

static int fac_ok(int n)
{
    return n >= 0 && n <= AC_WIGNER_FACMAX;
}

double ac_sqrt_factorial(int n)
{
    if (!tables_ready)
        init_tables();
    return fac_ok(n) ? sqfac[n] : NAN;
}

/* (a,b,c) in 2j units: triangle inequality and integer perimeter. */
static int triangle(int ta, int tb, int tc)
{
    if (ta < 0 || tb < 0 || tc < 0)
        return 0;
    if ((ta + tb + tc) % 2 != 0)
        return 0;
    return tc <= ta + tb && tc >= abs(ta - tb);
}

double ac_tri_delta(int ta, int tb, int tc)
{
    if (!tables_ready)
        init_tables();
    if (!triangle(ta, tb, tc))
        return 0.0;
    int n1 = (ta + tb - tc) / 2, n2 = (ta - tb + tc) / 2, n3 = (-ta + tb + tc) / 2;
    if (!fac_ok(n1) || !fac_ok(n2) || !fac_ok(n3))
        return NAN;
    return sqfac[n1] * sqfac[n2] * sqfac[n3];
}

static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

static double sixj_raw(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6)
{
    if (!tables_ready)
        init_tables();
    /* the four triads of {j1 j2 j3; j4 j5 j6} */
    if (!triangle(tj1, tj2, tj3) || !triangle(tj1, tj5, tj6) ||
        !triangle(tj4, tj2, tj6) || !triangle(tj4, tj5, tj3))
        return 0.0;

    /* Racah's formula (G. Racah, Phys. Rev. 62, 438 (1942); METECC-94
       appendix B; Edmonds (1957), section 6.3), as in the Fortran XWIG6J:
       alpha_i = triad perimeters / 2, beta_j = pair sums / 2 */
    int a1 = (tj1 + tj2 + tj3) / 2;
    int a2 = (tj1 + tj5 + tj6) / 2;
    int a3 = (tj4 + tj2 + tj6) / 2;
    int a4 = (tj4 + tj5 + tj3) / 2;
    int b1 = (tj1 + tj2 + tj4 + tj5) / 2;
    int b2 = (tj2 + tj3 + tj5 + tj6) / 2;
    int b3 = (tj3 + tj1 + tj6 + tj4) / 2;
    int tmin = imax(imax(a1, a2), imax(a3, a4));
    int tmax = imin(b1, imin(b2, b3));
    if (!fac_ok(tmax + 1))
        return NAN;

    double sum = 0.0;
    for (int t = tmin; t <= tmax; t++) {
        double term = fac[t + 1] /
                      (fac[t - a1] * fac[t - a2] * fac[t - a3] * fac[t - a4] *
                       fac[b1 - t] * fac[b2 - t] * fac[b3 - t]);
        sum += (t % 2 == 0) ? term : -term;
    }

    /* Delta(abc) = tri_delta / sqrt((a+b+c+1)!) for each triad */
    double d = ac_tri_delta(tj1, tj2, tj3) * ac_tri_delta(tj1, tj5, tj6) *
               ac_tri_delta(tj4, tj2, tj6) * ac_tri_delta(tj4, tj5, tj3);
    double den = sqfac[a1 + 1] * sqfac[a2 + 1] * sqfac[a3 + 1] * sqfac[a4 + 1];
    return sum * d / den;
}

/* 6j values are cached like the 9j below (all six 2j < 64: a 36-bit key) */
typedef struct {
    uint64_t key; /* key + 1; 0 = empty */
    double v;
} sj_entry;
static _Thread_local sj_entry *sj_tab;
static _Thread_local size_t sj_cap, sj_len;

double ac_sixj(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6)
{
    if (((unsigned)(tj1 | tj2 | tj3 | tj4 | tj5 | tj6)) >= 64)
        return sixj_raw(tj1, tj2, tj3, tj4, tj5, tj6);
    uint64_t key =
        (((((uint64_t)tj1 * 64 + tj2) * 64 + tj3) * 64 + tj4) * 64 + tj5) * 64 + tj6 + 1;
    if (2 * (sj_len + 1) > sj_cap) {
        size_t ncap = sj_cap ? 2 * sj_cap : 1 << 14;
        sj_entry *nt = calloc(ncap, sizeof *nt);
        if (!nt)
            return sixj_raw(tj1, tj2, tj3, tj4, tj5, tj6);
        for (size_t i = 0; i < sj_cap; i++)
            if (sj_tab[i].key) {
                size_t j = (size_t)((sj_tab[i].key * 0x9E3779B97F4A7C15ULL) >> 24) & (ncap - 1);
                while (nt[j].key)
                    j = (j + 1) & (ncap - 1);
                nt[j] = sj_tab[i];
            }
        free(sj_tab);
        sj_tab = nt;
        sj_cap = ncap;
    }
    size_t j = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 24) & (sj_cap - 1);
    while (sj_tab[j].key) {
        if (sj_tab[j].key == key)
            return sj_tab[j].v;
        j = (j + 1) & (sj_cap - 1);
    }
    double v = sixj_raw(tj1, tj2, tj3, tj4, tj5, tj6);
    sj_tab[j].key = key;
    sj_tab[j].v = v;
    sj_len++;
    return v;
}

static double ninej_raw(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6,
                        int tj7, int tj8, int tj9);

/* 9j values are cached: the factorization asks for the same few thousand
   symbols millions of times. Keys are the nine 2j (each < 256). */
typedef struct {
    unsigned char k[9];
    unsigned char used;
    double v;
} nj_entry;
static _Thread_local nj_entry *nj_tab;
static _Thread_local size_t nj_cap, nj_len;

static size_t nj_hash(const unsigned char *k)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < 9; i++) {
        h ^= k[i];
        h *= 1099511628211ULL;
    }
    return (size_t)h;
}

/* fast path: all nine 2j below 64 packed into 54 bits (key + 1, 0 = empty) */
typedef struct {
    uint64_t key;
    double v;
} nj_fast;
static _Thread_local nj_fast *njf_tab;
static _Thread_local size_t njf_cap, njf_len;

static double ninej_fast(uint64_t key, int tj1, int tj2, int tj3, int tj4, int tj5, int tj6,
                         int tj7, int tj8, int tj9)
{
    if (2 * (njf_len + 1) > njf_cap) {
        size_t ncap = njf_cap ? 2 * njf_cap : 1 << 14;
        nj_fast *nt = calloc(ncap, sizeof *nt);
        if (!nt)
            return ninej_raw(tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9);
        for (size_t i = 0; i < njf_cap; i++)
            if (njf_tab[i].key) {
                size_t j = (size_t)((njf_tab[i].key * 0x9E3779B97F4A7C15ULL) >> 24) & (ncap - 1);
                while (nt[j].key)
                    j = (j + 1) & (ncap - 1);
                nt[j] = njf_tab[i];
            }
        free(njf_tab);
        njf_tab = nt;
        njf_cap = ncap;
    }
    size_t j = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 24) & (njf_cap - 1);
    while (njf_tab[j].key) {
        if (njf_tab[j].key == key)
            return njf_tab[j].v;
        j = (j + 1) & (njf_cap - 1);
    }
    double v = ninej_raw(tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9);
    njf_tab[j].key = key;
    njf_tab[j].v = v;
    njf_len++;
    return v;
}

double ac_ninej(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6,
                int tj7, int tj8, int tj9)
{
    if (((unsigned)(tj1 | tj2 | tj3 | tj4 | tj5 | tj6 | tj7 | tj8 | tj9)) < 64) {
        uint64_t key = ((((((((uint64_t)tj1 * 64 + tj2) * 64 + tj3) * 64 + tj4) * 64 + tj5) * 64 +
                          tj6) * 64 + tj7) * 64 + tj8) * 64 + tj9;
        return ninej_fast(key + 1, tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9);
    }
    int a[9] = {tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9};
    unsigned char k[9];
    for (int i = 0; i < 9; i++) {
        if (a[i] < 0 || a[i] > 255)
            return ninej_raw(tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9);
        k[i] = (unsigned char)a[i];
    }
    if (2 * (nj_len + 1) > nj_cap) {
        size_t ncap = nj_cap ? 2 * nj_cap : 1 << 14;
        nj_entry *nt = calloc(ncap, sizeof *nt);
        if (!nt)
            return ninej_raw(tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9);
        for (size_t i = 0; i < nj_cap; i++)
            if (nj_tab[i].used) {
                size_t j = nj_hash(nj_tab[i].k) & (ncap - 1);
                while (nt[j].used)
                    j = (j + 1) & (ncap - 1);
                nt[j] = nj_tab[i];
            }
        free(nj_tab);
        nj_tab = nt;
        nj_cap = ncap;
    }
    size_t j = nj_hash(k) & (nj_cap - 1);
    while (nj_tab[j].used) {
        if (!memcmp(nj_tab[j].k, k, 9))
            return nj_tab[j].v;
        j = (j + 1) & (nj_cap - 1);
    }
    double v = ninej_raw(tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9);
    memcpy(nj_tab[j].k, k, 9);
    nj_tab[j].used = 1;
    nj_tab[j].v = v;
    nj_len++;
    return v;
}

static double ninej_raw(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6,
                        int tj7, int tj8, int tj9)
{
    /* x couples (j1,j9), (j4,j8) and (j2,j6) */
    int lo = imax(abs(tj1 - tj9), imax(abs(tj4 - tj8), abs(tj2 - tj6)));
    int hi = imin(tj1 + tj9, imin(tj4 + tj8, tj2 + tj6));
    if (lo > hi || (hi - lo) % 2 != 0)
        return 0.0;
    if ((lo + tj1 + tj9) % 2 != 0)
        return 0.0;

    double sum = 0.0;
    for (int tx = lo; tx <= hi; tx += 2) {
        double t = (tx + 1) * ac_sixj(tj1, tj2, tj3, tj6, tj9, tx) *
                   ac_sixj(tj4, tj5, tj6, tj2, tx, tj8) *
                   ac_sixj(tj7, tj8, tj9, tx, tj1, tj4);
        sum += (tx % 2 == 0) ? t : -t;
    }
    return sum;
}

double ac_fabc(int a, int b, int c)
{
    if (!tables_ready)
        init_tables();
    if (a < 0 || b < 0 || c < 0 || (a + b + c) % 2 != 0)
        return 0.0;
    if (c > a + b || c < abs(a - b))
        return 0.0;
    int g = (a + b + c) / 2;
    if (!fac_ok(2 * g + 1))
        return NAN;
    return fac[g] * sqfac[2 * g - 2 * a] * sqfac[2 * g - 2 * b] *
           sqfac[2 * g - 2 * c] /
           (sqfac[2 * g + 1] * fac[g - a] * fac[g - b] * fac[g - c]);
}

double ac_hk(int k, int l1, int l2, int l3, int l4)
{
    double f = ac_fabc(k, l1, l2);
    if (f == 0.0)
        return 0.0;
    f *= ac_fabc(k, l3, l4);
    if (f == 0.0)
        return 0.0;
    /* both F's nonzero => l1+l2+l3+l4 is even, so the exponent is exact */
    int e = (l1 - l2 + l3 - l4) / 2;
    double phase = (((e % 2) + 2) % 2 == 0) ? 1.0 : -1.0;
    double dims = (double)(2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) * (2 * l4 + 1);
    return phase * sqrt(dims / (2 * k + 1)) * f;
}
