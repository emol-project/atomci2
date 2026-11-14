/*
 * trace_wrap.c -- record the Fortran angular engine of ATOMCI as an oracle
 * for the C rewrite (stage 1, milestone M0).
 *
 * Linked into a scratch build of atomci with GNU ld --wrap, so the Fortran
 * sources stay untouched:
 *
 *   -Wl,--wrap=xwig6j_,--wrap=xwig9x_,--wrap=xcfp_,--wrap=shms_,
 *       --wrap=smat_,--wrap=isqt_
 *
 * --wrap only redirects references between objects, so the thin wrappers
 * WIG6J/WIG9X/CFP inside wt9x.o are not seen; their bodies XWIG6J/XWIG9X/XCFP
 * live in other objects and are.
 *
 * Output goes to $AC_TRACE_DIR (default "."), one text file per routine:
 *
 *   wig6j.txt  a b c d e f value                (2j+1 arguments)
 *   wig9x.txt  a b c d e f g h i value
 *   xcfp.txt   im km kmp id value
 *   shms.txt   nhlf[12] nhst[12] nhwn[12] nhwq[12] nhwcsl[12]
 *              nhvn[12] nhvq[12] nhvcsl[12]
 *              idummy nl2 nl1 lsi lri nl3 nl4 lsf lrf lst lrt nht  w
 *   smat.txt   one block per SMAT call:
 *                S <seq> nws nwen[nws] nweq[nws] nwesl[nws] nwrps[nws]
 *                        nvs nven[nvs] nveq[nvs] nvesl[nvs] nvrps[nvs]
 *                        nwt nwth[nwt] nwf nwtc[nwf]
 *                T km i1 i2 i3 i4 noe wt          (each ISQT inside the call)
 *                E noe n1oe nhtr nfsr nhdtx nwst nvst
 *              ISQT calls outside any SMAT call are written as "T" lines
 *              preceded by a single "X" line.
 *
 * wig6j/wig9x/xcfp/shms records are deduplicated (a record is written the
 * first time its full text appears); smat.txt keeps every call in order.
 * Doubles are printed with %.17g so they round-trip exactly.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the real Fortran entry points (gfortran: lower case + '_') ---- */
double __real_xwig6j_(int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                      int32_t *);
double __real_xwig9x_(int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                      int32_t *, int32_t *, int32_t *, int32_t *);
double __real_xcfp_(int32_t *, int32_t *, int32_t *, int32_t *);
void __real_shms_(int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                  int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                  int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                  int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                  double *);
void __real_smat_(int32_t *, int16_t *, int16_t *, int16_t *, int16_t *,
                  int32_t *, int16_t *, int16_t *, int16_t *, int16_t *,
                  int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                  int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                  int32_t *);
void __real_isqt_(int32_t *, int32_t *, int32_t *, int32_t *, int32_t *,
                  int32_t *, double *);

/* ---- output files ---- */
enum { F_WIG6J, F_WIG9X, F_XCFP, F_SHMS, F_SMAT, F_COUNT };
static const char *const fname[F_COUNT] = {
    "wig6j.txt", "wig9x.txt", "xcfp.txt", "shms.txt", "smat.txt"};
static FILE *fp[F_COUNT];

static void close_all(void)
{
    for (int i = 0; i < F_COUNT; i++)
        if (fp[i])
            fclose(fp[i]);
}

static FILE *out(int which)
{
    if (!fp[which]) {
        const char *dir = getenv("AC_TRACE_DIR");
        char path[4096];
        snprintf(path, sizeof path, "%s/%s", dir ? dir : ".", fname[which]);
        fp[which] = fopen(path, "w");
        if (!fp[which]) {
            perror(path);
            abort();
        }
        static int registered;
        if (!registered) {
            atexit(close_all);
            registered = 1;
        }
    }
    return fp[which];
}

/* ---- deduplication: set of 64-bit FNV-1a hashes per file ---- */
typedef struct {
    uint64_t *slot;
    size_t cap, n;
} hset;
static hset seen[F_COUNT];

static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h ? h : 1; /* 0 marks an empty slot */
}

/* returns 1 if h was newly inserted */
static int hset_add(hset *s, uint64_t h)
{
    if (2 * (s->n + 1) > s->cap) {
        size_t ncap = s->cap ? 2 * s->cap : 1024;
        uint64_t *ns = calloc(ncap, sizeof *ns);
        if (!ns)
            abort();
        for (size_t i = 0; i < s->cap; i++)
            if (s->slot[i]) {
                size_t j = s->slot[i] & (ncap - 1);
                while (ns[j])
                    j = (j + 1) & (ncap - 1);
                ns[j] = s->slot[i];
            }
        free(s->slot);
        s->slot = ns;
        s->cap = ncap;
    }
    size_t j = h & (s->cap - 1);
    while (s->slot[j]) {
        if (s->slot[j] == h)
            return 0;
        j = (j + 1) & (s->cap - 1);
    }
    s->slot[j] = h;
    s->n++;
    return 1;
}

/* ---- a small growable line buffer ---- */
typedef struct {
    char *p;
    size_t len, cap;
} lbuf;

static void lb_printf(lbuf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void lb_printf(lbuf *b, const char *fmt, ...)
{
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        size_t room = b->cap - b->len;
        int n = vsnprintf(b->p ? b->p + b->len : NULL, room, fmt, ap);
        va_end(ap);
        if (n < 0)
            abort();
        if ((size_t)n < room) {
            b->len += (size_t)n;
            return;
        }
        size_t ncap = b->cap ? 2 * b->cap : 256;
        while (ncap - b->len <= (size_t)n)
            ncap *= 2;
        char *np = realloc(b->p, ncap);
        if (!np)
            abort();
        b->p = np;
        b->cap = ncap;
    }
}

static void lb_ints32(lbuf *b, const int32_t *v, int n)
{
    for (int i = 0; i < n; i++)
        lb_printf(b, " %d", (int)v[i]);
}

static void lb_ints16(lbuf *b, const int16_t *v, int n)
{
    for (int i = 0; i < n; i++)
        lb_printf(b, " %d", (int)v[i]);
}

/* key = everything before the value; written once per distinct key */
static void emit_dedup(int which, lbuf *key, double value)
{
    if (hset_add(&seen[which], fnv1a(key->p)))
        fprintf(out(which), "%s %.17g\n", key->p, value);
    key->len = 0;
    key->p[0] = '\0';
}

/* ---- wrappers ---- */
double __wrap_xwig6j_(int32_t *a, int32_t *b, int32_t *c, int32_t *d,
                      int32_t *e, int32_t *f)
{
    double v = __real_xwig6j_(a, b, c, d, e, f);
    static lbuf k;
    lb_printf(&k, "%d %d %d %d %d %d", *a, *b, *c, *d, *e, *f);
    emit_dedup(F_WIG6J, &k, v);
    return v;
}

double __wrap_xwig9x_(int32_t *a, int32_t *b, int32_t *c, int32_t *d,
                      int32_t *e, int32_t *f, int32_t *g, int32_t *h,
                      int32_t *i)
{
    double v = __real_xwig9x_(a, b, c, d, e, f, g, h, i);
    static lbuf k;
    lb_printf(&k, "%d %d %d %d %d %d %d %d %d", *a, *b, *c, *d, *e, *f, *g,
              *h, *i);
    emit_dedup(F_WIG9X, &k, v);
    return v;
}

double __wrap_xcfp_(int32_t *im, int32_t *km, int32_t *kmp, int32_t *id)
{
    double v = __real_xcfp_(im, km, kmp, id);
    static lbuf k;
    lb_printf(&k, "%d %d %d %d", *im, *km, *kmp, *id);
    emit_dedup(F_XCFP, &k, v);
    return v;
}

void __wrap_shms_(int32_t *nhlf, int32_t *nhst, int32_t *nhwn, int32_t *nhwq,
                  int32_t *nhwcsl, int32_t *nhvn, int32_t *nhvq,
                  int32_t *nhvcsl, int32_t *idummy, int32_t *nl2,
                  int32_t *nl1, int32_t *lsi, int32_t *lri, int32_t *nl3,
                  int32_t *nl4, int32_t *lsf, int32_t *lrf, int32_t *lst,
                  int32_t *lrt, int32_t *nht, double *w)
{
    /* SHMS declares every array with 12 elements; record all 12 so the
       record is the complete input whatever SHMSRC/SHMSD index. */
    static lbuf k;
    const int32_t *arr[8] = {nhlf, nhst, nhwn, nhwq, nhwcsl, nhvn, nhvq, nhvcsl};
    for (int i = 0; i < 8; i++)
        lb_ints32(&k, arr[i], 12);
    lb_printf(&k, " %d %d %d %d %d %d %d %d %d %d %d %d", *idummy, *nl2,
              *nl1, *lsi, *lri, *nl3, *nl4, *lsf, *lrf, *lst, *lrt, *nht);
    __real_shms_(nhlf, nhst, nhwn, nhwq, nhwcsl, nhvn, nhvq, nhvcsl, idummy,
                 nl2, nl1, lsi, lri, nl3, nl4, lsf, lrf, lst, lrt, nht, w);
    emit_dedup(F_SHMS, &k, *w);
}

static long smat_seq;
static int in_smat;
static int outside_marked;

void __wrap_smat_(int32_t *nws, int16_t *nwen, int16_t *nweq, int16_t *nwesl,
                  int16_t *nwrps, int32_t *nvs, int16_t *nven, int16_t *nveq,
                  int16_t *nvesl, int16_t *nvrps, int32_t *noe, int32_t *n1oe,
                  int32_t *nhtr, int32_t *nfsr, int32_t *nhdtx, int32_t *nwst,
                  int32_t *nvst, int32_t *nwt, int32_t *nwf, int32_t *nwth,
                  int32_t *nwtc)
{
    FILE *f = out(F_SMAT);
    static lbuf k;
    lb_printf(&k, "S %ld %d", ++smat_seq, *nws);
    lb_ints16(&k, nwen, *nws);
    lb_ints16(&k, nweq, *nws);
    lb_ints16(&k, nwesl, *nws);
    lb_ints16(&k, nwrps, *nws);
    lb_printf(&k, " %d", *nvs);
    lb_ints16(&k, nven, *nvs);
    lb_ints16(&k, nveq, *nvs);
    lb_ints16(&k, nvesl, *nvs);
    lb_ints16(&k, nvrps, *nvs);
    lb_printf(&k, " %d", *nwt);
    lb_ints32(&k, nwth, *nwt);
    lb_printf(&k, " %d", *nwf);
    lb_ints32(&k, nwtc, *nwf);
    fprintf(f, "%s\n", k.p);
    k.len = 0;
    k.p[0] = '\0';

    in_smat = 1;
    __real_smat_(nws, nwen, nweq, nwesl, nwrps, nvs, nven, nveq, nvesl, nvrps,
                 noe, n1oe, nhtr, nfsr, nhdtx, nwst, nvst, nwt, nwf, nwth,
                 nwtc);
    in_smat = 0;
    outside_marked = 0;
    fprintf(f, "E %d %d %d %d %d %d %d\n", *noe, *n1oe, *nhtr, *nfsr, *nhdtx,
            *nwst, *nvst);
}

void __wrap_isqt_(int32_t *km, int32_t *i1, int32_t *i2, int32_t *i3,
                  int32_t *i4, int32_t *inoe, double *wtx)
{
    FILE *f = out(F_SMAT);
    if (!in_smat && !outside_marked) {
        fputs("X\n", f);
        outside_marked = 1;
    }
    fprintf(f, "T %d %d %d %d %d %d %.17g\n", *km, *i1, *i2, *i3, *i4, *inoe,
            *wtx);
    __real_isqt_(km, i1, i2, i3, i4, inoe, wtx);
}
