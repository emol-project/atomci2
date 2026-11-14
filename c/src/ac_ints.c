/*
 * ac_ints.c -- the transformed integral file NFTRI (see ac_ints.h).
 */
#include "ac_ints.h"
#include "ac_ftnio.h"

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

static int append(ac_ints *in, long *cap, const double *v, long nv)
{
    if (in->n + nv + 1 > *cap) {
        long nc = *cap ? *cap : 1 << 16;
        while (nc < in->n + nv + 1)
            nc *= 2;
        double *nx = realloc(in->x, (size_t)nc * sizeof *nx);
        if (!nx)
            return -1;
        in->x = nx;
        *cap = nc;
    }
    memcpy(in->x + in->n + 1, v, (size_t)nv * sizeof *v);
    in->n += nv;
    return 0;
}

int ac_ints_load(ac_ints *in, const char *path, char *err, int errlen)
{
    memset(in, 0, sizeof *in);
    ac_ftn_file f;
    if (ac_ftn_open(&f, path, "rb"))
        return fail(err, errlen, "cannot open %s", path);
    int rc = -1;
    long cap = 0;
    ac_ftn_cursor c;

    if (ac_ftn_read(&f) != AC_FTN_OK) {
        fail(err, errlen, "%s: no header", path);
        goto out;
    }
    ac_ftn_cursor_init(&c, &f);
    for (int i = 0; i < 33; i++)
        (void)ac_ftn_i32(&c);
    in->nsym = ac_ftn_i32(&c);
    in->nobt = ac_ftn_i32(&c);
    (void)ac_ftn_i32(&c); /* NNUC  */
    (void)ac_ftn_i32(&c); /* NFLGS */
    (void)ac_ftn_i32(&c); /* LTRB1 */
    if (c.bad || in->nsym < 1 || in->nsym > 20) {
        fail(err, errlen, "%s: bad header (NSYM %d)", path, in->nsym);
        goto out;
    }
    for (int l = 0; l < in->nsym; l++)
        in->nob[l] = ac_ftn_i32(&c);

    /* S then H, triangular per l; H overwrites S */
    for (int pass = 0; pass < 2; pass++) {
        in->n = 0;
        for (int l = 0; l < in->nsym; l++) {
            long need = (long)in->nob[l] * (in->nob[l] + 1) / 2;
            while (need > 0) {
                if (ac_ftn_read(&f) != AC_FTN_OK) {
                    fail(err, errlen, "%s: one-electron part truncated", path);
                    goto out;
                }
                ac_ftn_cursor_init(&c, &f);
                (void)ac_ftn_i32(&c);
                (void)ac_ftn_i32(&c);
                int nt = ac_ftn_i32(&c);
                if (nt < 1 || nt > need || ac_ftn_left(&c) != (size_t)nt * 8) {
                    fail(err, errlen, "%s: bad one-electron record (NT %d)", path, nt);
                    goto out;
                }
                double *v = malloc((size_t)nt * sizeof *v);
                for (int i = 0; i < nt; i++)
                    v[i] = ac_ftn_f64(&c);
                int r = append(in, &cap, v, nt);
                free(v);
                if (r)
                    goto oom;
                need -= nt;
            }
        }
    }
    in->n1 = in->n;

    /* two-electron blocks */
    for (;;) {
        int r = ac_ftn_read(&f);
        if (r == AC_FTN_EOF)
            break;
        if (r != AC_FTN_OK || f.len != 14 * 4) {
            fail(err, errlen, "%s: expected a two-electron block header", path);
            goto out;
        }
        int32_t np[15];
        ac_ftn_cursor_init(&c, &f);
        for (int i = 1; i <= 14; i++)
            np[i] = ac_ftn_i32(&c);
        int tri = np[2] == np[4] && np[3] == np[5];
        for (int k = 1; k <= np[11]; k++) {
            long want = tri ? k : np[10];
            if (ac_ftn_read(&f) != AC_FTN_OK || f.len != (size_t)want * 8) {
                fail(err, errlen, "%s: bad two-electron record", path);
                goto out;
            }
            if (append(in, &cap, (const double *)f.buf, want))
                goto oom;
        }
    }
    rc = 0;
    goto out;
oom:
    fail(err, errlen, "out of memory");
out:
    ac_ftn_close(&f);
    if (rc)
        ac_ints_free(in);
    return rc;
}

void ac_ints_free(ac_ints *in)
{
    free(in->x);
    memset(in, 0, sizeof *in);
}
