/*
 * ac_cfp.c -- CFPTBL loading, shell-state codes and lists, XCFP.
 * Ports src/setcfp.fpl, src/cfpset.fpl, src/isp.f, src/ior.f, src/xcfp.f.
 */
#include "ac_cfp.h"
#include "ac_ftnio.h"
#include "ac_wigner.h"

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

/* ---- CFPSET: shell-state lists ------------------------------------------ */

static int lst_index(const ac_cfptab *t, int l, int nel)
{
    return l * (t->maxn + 1) + nel;
}

static int push_code(ac_cfptab *t, int *cap, int code)
{
    if (t->lst_len == *cap) {
        int ncap = *cap ? 2 * *cap : 256;
        int *nc = realloc(t->lst_code, (size_t)ncap * sizeof *nc);
        if (!nc)
            return -1;
        t->lst_code = nc;
        *cap = ncap;
    }
    t->lst_code[t->lst_len++] = code;
    return 0;
}

int ac_cfp_build_lists(ac_cfptab *t)
{
    t->maxn = 4 * t->ibel - 2;
    size_t n = (size_t)t->ibel * (size_t)(t->maxn + 1);
    t->lst_start = malloc(n * sizeof *t->lst_start);
    t->lst_count = malloc(n * sizeof *t->lst_count);
    if (!t->lst_start || !t->lst_count)
        return -1;
    for (size_t i = 0; i < n; i++) {
        t->lst_start[i] = -1;
        t->lst_count[i] = 0;
    }
    int cap = 0;

    for (int nl = 1; nl <= t->ibel; nl++) { /* nl = l + 1 */
        int l = nl - 1;
        int nemax = 2 * nl - t->immint[nl - 1];
        int isq_lo = t->ilstad[nl - 1];
        int isq_hi = nl < t->ibel ? t->ilstad[nl] - 1 : t->icnn;

        /* states up to half filling (all N for truncated shells) */
        for (int ne = 0; ne <= nemax; ne++) {
            int k = lst_index(t, l, ne);
            t->lst_start[k] = t->lst_len;
            for (int isq = isq_lo; isq <= isq_hi; isq++) {
                int iq = t->iconnt[isq - 1][1];
                if (iq + ne >= 2 * nl && (iq + ne) % 2 == 0)
                    for (int lsub = 1; lsub <= t->iconnt[isq - 1][3]; lsub++) {
                        if (push_code(t, &cap, 256 * isq + lsub))
                            return -1;
                        t->lst_count[k]++;
                    }
            }
        }
        /* complete shells: N > half filling reuses the list of 4l+2-N */
        if (t->immint[nl - 1] == 1)
            for (int ne = nemax + 1; ne <= 4 * nl - 2; ne++) {
                int k = lst_index(t, l, ne);
                int m = lst_index(t, l, 4 * nl - 2 - ne);
                t->lst_start[k] = t->lst_start[m];
                t->lst_count[k] = t->lst_count[m];
            }
    }
    return 0;
}

/* ---- loading ------------------------------------------------------------ */

int ac_cfp_load(ac_cfptab *t, const char *path, char *err, int errlen)
{
    if (!strcmp(path, "-")) /* no file: the table generated in C */
        return ac_cfp_generate(t, err, errlen);
    memset(t, 0, sizeof *t);
    ac_ftn_file f;
    if (ac_ftn_open(&f, path, "rb") != AC_FTN_OK)
        return fail(err, errlen, "cannot open %s", path);

    ac_ftn_cursor c;
    int rc = -1;

    /* record 1: INAME(18), IBEL, ICNN, ILFN, ICMTN, ICDX, IMMINT(18), ILSTAD(18) */
    if (ac_ftn_read(&f) != AC_FTN_OK || f.len != 59 * 4) {
        fail(err, errlen, "%s: record 1 missing or not 59 integers", path);
        goto out;
    }
    ac_ftn_cursor_init(&c, &f);
    for (int i = 0; i < 18; i++)
        t->name[i] = ac_ftn_i32(&c);
    t->ibel = ac_ftn_i32(&c);
    t->icnn = ac_ftn_i32(&c);
    t->ilfn = ac_ftn_i32(&c);
    t->icmtn = ac_ftn_i32(&c);
    t->icdx = ac_ftn_i32(&c);
    for (int i = 0; i < 18; i++)
        t->immint[i] = ac_ftn_i32(&c);
    for (int i = 0; i < 18; i++)
        t->ilstad[i] = ac_ftn_i32(&c);
    if (t->ibel < 1 || t->ibel > 18 || t->icnn < 1 || t->ilfn < 1 ||
        t->icmtn < 1 || t->icdx < 1) {
        fail(err, errlen, "%s: implausible sizes ibel=%d icnn=%d ilfn=%d "
             "icmtn=%d icdx=%d", path, t->ibel, t->icnn, t->ilfn, t->icmtn,
             t->icdx);
        goto out;
    }

    t->iconnt = malloc((size_t)t->icnn * sizeof *t->iconnt);
    t->icmtdx = malloc((size_t)t->icmtn * sizeof *t->icmtdx);
    t->ilfunc = malloc((size_t)t->ilfn * sizeof *t->ilfunc);
    t->cmttbl = malloc((size_t)t->icdx * sizeof *t->cmttbl);
    if (!t->iconnt || !t->icmtdx || !t->ilfunc || !t->cmttbl) {
        fail(err, errlen, "out of memory");
        goto out;
    }

    /* record 2: ICONNT(9,ICNN), Fortran column order = row by row here */
    if (ac_ftn_read(&f) != AC_FTN_OK || f.len != (size_t)t->icnn * 9 * 4) {
        fail(err, errlen, "%s: record 2 is not ICONNT(9,%d)", path, t->icnn);
        goto out;
    }
    ac_ftn_cursor_init(&c, &f);
    for (int j = 0; j < t->icnn; j++)
        for (int i = 0; i < 9; i++)
            t->iconnt[j][i] = ac_ftn_i32(&c);

    /* record 3: ICMTDX(5,ICMTN) */
    if (ac_ftn_read(&f) != AC_FTN_OK || f.len != (size_t)t->icmtn * 5 * 4) {
        fail(err, errlen, "%s: record 3 is not ICMTDX(5,%d)", path, t->icmtn);
        goto out;
    }
    ac_ftn_cursor_init(&c, &f);
    for (int j = 0; j < t->icmtn; j++)
        for (int i = 0; i < 5; i++)
            t->icmtdx[j][i] = ac_ftn_i32(&c);

    /* record 4: ILFUNC(ILFN) */
    if (ac_ftn_read(&f) != AC_FTN_OK || f.len != (size_t)t->ilfn * 4) {
        fail(err, errlen, "%s: record 4 is not ILFUNC(%d)", path, t->ilfn);
        goto out;
    }
    ac_ftn_cursor_init(&c, &f);
    for (int i = 0; i < t->ilfn; i++)
        t->ilfunc[i] = ac_ftn_i32(&c);

    /* record 5: CMTTBL(ICDX), real*8 */
    if (ac_ftn_read(&f) != AC_FTN_OK || f.len != (size_t)t->icdx * 8) {
        fail(err, errlen, "%s: record 5 is not CMTTBL(%d)", path, t->icdx);
        goto out;
    }
    ac_ftn_cursor_init(&c, &f);
    for (int i = 0; i < t->icdx; i++)
        t->cmttbl[i] = ac_ftn_f64(&c);

    if (ac_cfp_build_lists(t)) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    rc = 0;
out:
    ac_ftn_close(&f);
    if (rc)
        ac_cfp_free(t);
    return rc;
}

void ac_cfp_free(ac_cfptab *t)
{
    free(t->iconnt);
    free(t->icmtdx);
    free(t->ilfunc);
    free(t->cmttbl);
    free(t->lst_start);
    free(t->lst_count);
    free(t->lst_code);
    memset(t, 0, sizeof *t);
}

/* ---- codes -------------------------------------------------------------- */

/* row index (1-based isq) of a code, or 0 if invalid */
static int code_row(const ac_cfptab *t, int code)
{
    if (code <= 0)
        return 0;
    int isq = code / 256, lsub = code & 255;
    if (isq < 1 || isq > t->icnn || lsub < 1 || lsub > t->iconnt[isq - 1][3])
        return 0;
    return isq;
}

int ac_cfp_decode(const ac_cfptab *t, int code, ac_shell_qn *qn)
{
    int isq = code_row(t, code);
    if (!isq)
        return -1;
    const int *r = t->iconnt[isq - 1];
    int lf = r[4] + (code & 255) - 1; /* 1-based ILFUNC address */
    if (lf < 1 || lf > t->ilfn)
        return -1;
    qn->tl = r[0] - 1;
    qn->tq = r[1] - 1;
    qn->ts = r[2] - 1;
    qn->tL = t->ilfunc[lf - 1] - 1;
    return 0;
}

int ac_cfp_lsqn(const ac_cfptab *t, int tl, int tq, int ts)
{
    int L = tl + 1, IQ = tq + 1, IS = ts + 1; /* multiplicities, as LSQN */
    int m = (L + 1) / 2;
    return (L * (L + 6 - 2 * IQ) + IQ * (IQ - 6) - 2 * IS + 8) / 4 +
           t->ilstad[m - 1];
}

int ac_cfp_states(const ac_cfptab *t, int l, int nel, const int **codes)
{
    *codes = NULL;
    if (l < 0 || l >= t->ibel || nel < 0 || nel > t->maxn)
        return -1;
    int k = lst_index(t, l, nel);
    if (t->lst_start[k] < 0)
        return -1;
    *codes = t->lst_code + t->lst_start[k];
    return t->lst_count[k];
}

/* ---- XCFP --------------------------------------------------------------- */

/* sqrt((n-1)!), the Fortran DEN(n) (n >= 1) */
static double den(int n)
{
    return ac_sqrt_factorial(n - 1);
}

/* the Fortran DELTA(IA,IB,IC) with multiplicity arguments */
static double delta_mult(int ia, int ib, int ic)
{
    return ac_tri_delta(ia - 1, ib - 1, ic - 1);
}

double ac_cfp_xcfp(const ac_cfptab *t, int im, int km, int kmp, int id)
{
    int kshl = code_row(t, km), ishl = code_row(t, im);
    if (!kshl || !ishl)
        return 0.0;
    int k = km & 255, i = im & 255;
    const int *rk = t->iconnt[kshl - 1], *ri = t->iconnt[ishl - 1];
    int lfo = rk[0];
    if (ri[0] != lfo)
        return 0.0;
    int kmu = rk[1], imu = ri[1], isp = ri[2], ksp = rk[2];

    /* ITYP: position of I relative to K in the (2Q+1, 2S+1) plane */
    int ityp = 0, icft = 0;
    if (kmu - imu == 1) {
        if (isp - ksp == 1) { ityp = 1; icft = rk[5]; }
        else if (isp - ksp == -1) { ityp = 2; icft = rk[6]; }
    } else if (kmu - imu == -1) {
        if (isp - ksp == 1) { ityp = 3; icft = rk[7]; }
        else if (isp - ksp == -1) { ityp = 4; icft = rk[8]; }
    }
    if (ityp == 0 || icft <= 0 || icft > t->icmtn)
        return 0.0;

    int nstad = t->icmtdx[icft - 1][0];
    int ncfp;
    if (ityp < 3)
        ncfp = nstad + i + (k - 1) * ri[3] - 1;
    else
        ncfp = nstad + k + (i - 1) * rk[3] - 1;
    if (ncfp < 1 || ncfp > t->icdx)
        return 0.0;
    double cfpv = t->cmttbl[ncfp - 1];

    /* stored orientation is (lower Q) x (higher Q); reverse for ITYP 3/4 */
    if (ityp >= 3) {
        int inad = ri[4] + i - 1, knad = rk[4] + k - 1;
        int ipar = lfo + t->ilfunc[inad - 1] - t->ilfunc[knad - 1] + 2 * ityp + 400;
        if (ipar % 4 != 3)
            cfpv = -cfpv;
    }

    /* quasi-spin Wigner-Eckart factor */
    int iqkz = kmp - lfo; /* 2 q_z of K */
    if (id != 1) {
        int tmp = kmu;
        kmu = imu;
        imu = tmp;
        iqkz -= 1;
    }
    int ikp = (kmu + iqkz + 1) / 2;
    int ikm = ikp - iqkz;
    int iim = (imu - iqkz) / 2;
    int iip = iim + iqkz + 1;
    int isum = (imu + kmu + 3) / 2;
    double v = cfpv * den(ikm) * den(iip) * delta_mult(2, kmu, imu) /
               (den(ikp) * den(iim) * den(isum));
    if (id != 1) {
        if ((kmu * 3 + imu) % 4 == 3)
            v = -v;
    } else if ((kmu + kmu + imu + imu) % 4 == 0) {
        v = -v;
    }
    return v;
}
