/*
 * ac_tape.c -- the NFTE tape (see ac_tape.h).
 *
 * gen_exij() follows src/exij.f statement by statement (labels L<n> are its
 * statement numbers) and store()/storeh() follow the X_STORE/X_STOREH
 * procedures of src/ntisqt.fpl. Arrays are indexed from 1 like the Fortran.
 */
#include "ac_tape.h"

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

/* ---- header ------------------------------------------------------------ */

static int keep_raw(ac_nfte_head *h, const ac_ftn_file *f)
{
    unsigned char **nr = realloc(h->raw, (size_t)(h->nraw + 1) * sizeof *nr);
    if (!nr)
        return -1;
    h->raw = nr;
    size_t *nl = realloc(h->rawlen, (size_t)(h->nraw + 1) * sizeof *nl);
    if (!nl)
        return -1;
    h->rawlen = nl;
    unsigned char *b = malloc(f->len ? f->len : 1);
    if (!b)
        return -1;
    memcpy(b, f->buf, f->len);
    h->raw[h->nraw] = b;
    h->rawlen[h->nraw] = f->len;
    h->nraw++;
    return 0;
}

int ac_nfte_read_head(ac_ftn_file *f, ac_nfte_head *h, char *err, int errlen)
{
    memset(h, 0, sizeof *h);
    ac_ftn_cursor c;

    if (ac_ftn_read(f) != AC_FTN_OK || f->len != 45 * 4)
        return fail(err, errlen, "record 1 missing or not 45 integers");
    if (keep_raw(h, f))
        return fail(err, errlen, "out of memory");
    ac_ftn_cursor_init(&c, f);
    for (int i = 0; i < 30; i++)
        h->name[i] = ac_ftn_i32(&c);
    int32_t *v[15] = {&h->ispin, &h->iorb, &h->ipar, &h->nwt, &h->nwf,
                      &h->itstv, &h->itnel, &h->lijb, &h->idiag, &h->lmax,
                      &h->nint, &h->lwfl, &h->ntype, &h->nref, &h->mxrps};
    for (int i = 0; i < 15; i++)
        *v[i] = ac_ftn_i32(&c);
    if (h->nwt < 1 || h->nwf < 0 || h->lmax < 1 || h->nint < 1 || h->lwfl < 1)
        return fail(err, errlen, "implausible header nwt=%d nwf=%d lmax=%d int=%d",
                    h->nwt, h->nwf, h->lmax, h->nint);

    if (ac_ftn_read(f) != AC_FTN_OK ||
        f->len != (size_t)(h->lmax + 2 * h->nwt + h->nwf) * 4)
        return fail(err, errlen, "record 2 is not ISSH/NWTN/NWTC/NWSQ");
    if (keep_raw(h, f))
        return fail(err, errlen, "out of memory");
    h->issh = malloc((size_t)h->lmax * sizeof(int32_t));
    h->nwtn = malloc((size_t)h->nwt * sizeof(int32_t));
    h->nwtc = malloc((size_t)(h->nwf ? h->nwf : 1) * sizeof(int32_t));
    h->nwsq = malloc((size_t)h->nwt * sizeof(int32_t));
    h->nwfsg = malloc((size_t)h->nint * sizeof(int16_t));
    if (!h->issh || !h->nwtn || !h->nwtc || !h->nwsq || !h->nwfsg)
        return fail(err, errlen, "out of memory");
    ac_ftn_cursor_init(&c, f);
    for (int i = 0; i < h->lmax; i++) h->issh[i] = ac_ftn_i32(&c);
    for (int i = 0; i < h->nwt; i++) h->nwtn[i] = ac_ftn_i32(&c);
    for (int i = 0; i < h->nwf; i++) h->nwtc[i] = ac_ftn_i32(&c);
    for (int i = 0; i < h->nwt; i++) h->nwsq[i] = ac_ftn_i32(&c);

    for (int got = 0; got < h->nint;) {
        if (ac_ftn_read(f) != AC_FTN_OK)
            return fail(err, errlen, "CSF image truncated at word %d of %d", got, h->nint);
        if (keep_raw(h, f))
            return fail(err, errlen, "out of memory");
        ac_ftn_cursor_init(&c, f);
        int kt = ac_ftn_i32(&c);
        if (kt < 1 || got + kt > h->nint || ac_ftn_left(&c) != (size_t)kt * 2)
            return fail(err, errlen, "bad CSF image chunk (KT=%d)", kt);
        for (int i = 0; i < kt; i++)
            h->nwfsg[got + i] = ac_ftn_i16(&c);
        got += kt;
    }
    return 0;
}

void ac_nfte_head_free(ac_nfte_head *h)
{
    free(h->issh);
    free(h->nwtn);
    free(h->nwtc);
    free(h->nwsq);
    free(h->nwfsg);
    for (int i = 0; i < h->nraw; i++)
        free(h->raw[i]);
    free(h->raw);
    free(h->rawlen);
    memset(h, 0, sizeof *h);
}

int ac_nfte_write_head(ac_ftn_file *f, const ac_nfte_head *h)
{
    for (int i = 0; i < h->nraw; i++)
        if (ac_ftn_write(f, h->raw[i], h->rawlen[i]) != AC_FTN_OK)
            return -1;
    return 0;
}

/* ---- data records ------------------------------------------------------ */

static int grow(void **p, int *cap, int n, size_t sz)
{
    if (n <= *cap)
        return 0;
    void *np = realloc(*p, (size_t)n * sz);
    if (!np)
        return -1;
    *p = np;
    *cap = n;
    return 0;
}

int ac_nfte_read_rec(ac_ftn_file *f, ac_nfte_rec *r)
{
    int rc = ac_ftn_read(f);
    if (rc != AC_FTN_OK)
        return rc;
    ac_ftn_cursor c;
    ac_ftn_cursor_init(&c, f);
    int32_t *v[8] = {&r->itw, &r->jtv, &r->noe, &r->n1oe, &r->n2oe, &r->nij, &r->nii, &r->ltri};
    for (int i = 0; i < 8; i++)
        *v[i] = ac_ftn_i32(&c);
    if (r->noe < 0 || r->ltri < 0 || r->nij < 0 || r->nii < 0)
        return AC_FTN_ERR_FMT;
    if (grow((void **)&r->wt, &r->cap_wt, r->noe, sizeof *r->wt) ||
        grow((void **)&r->isqn, &r->cap_isqn, r->ltri, sizeof *r->isqn) ||
        grow((void **)&r->indexi, &r->cap_i, r->nij, sizeof *r->indexi) ||
        grow((void **)&r->indexj, &r->cap_j, r->nii, sizeof *r->indexj))
        return AC_FTN_ERR_IO;
    for (int i = 0; i < r->noe; i++) r->wt[i] = ac_ftn_f64(&c);
    for (int i = 0; i < r->ltri; i++) r->isqn[i] = ac_ftn_i32(&c);
    for (int i = 0; i < r->nij; i++) r->indexi[i] = ac_ftn_i32(&c);
    for (int i = 0; i < r->nii; i++) r->indexj[i] = ac_ftn_i32(&c);
    if (c.bad || ac_ftn_left(&c) != 0)
        return AC_FTN_ERR_FMT;
    return AC_FTN_OK;
}

void ac_nfte_rec_free(ac_nfte_rec *r)
{
    free(r->wt);
    free(r->isqn);
    free(r->indexi);
    free(r->indexj);
    memset(r, 0, sizeof *r);
}

/* ---- generation -------------------------------------------------------- */

typedef struct {
    const ac_cfptab *t;
    const ac_nfte_head *h;
    ac_ftn_file *out;
    ac_addr addr;
    int32_t *nwth;     /* [nwt] empty/closed state codes (EXTAPE SETUP) */
    ac_tape_stats *st;
    int err;

    /* /COMEXP/ */
    int iwad[17], ivad[17], iwvad[131];
    /* the last SMAT's terms (/COMSTR/ KMT, I1T..I4T, WT) */
    ac_terms terms;
    /* STORE's buffer (/COMSTR/ ISQN, INDEXI, INDEXJ) */
    int noex, n1oex, n2oe, nij, ltri;
    int ltrix, nijx;
    int32_t *isqn, *indexi, *indexj;
    int cap_isqn, cap_i, cap_j;
} gen;

/* X_STOREH: write the buffered pairs as one record */
static void storeh(gen *g)
{
    if (g->noex == 0 || g->nij == 0)
        return;
    if (g->ltri != g->nij * g->noex) {
        g->err = -2;
        return;
    }
    int itwx = g->indexi[1], jtvx = g->indexj[1];
    int nii = itwx == jtvx ? 0 : g->nij;
    size_t len = 8 * 4 + (size_t)g->noex * 8 + (size_t)g->ltri * 4 +
                 (size_t)g->nij * 4 + (size_t)nii * 4;
    unsigned char *b = malloc(len), *p = b;
    if (!b) {
        g->err = -1;
        return;
    }
    int32_t hd[8] = {itwx, jtvx, g->noex, g->n1oex, g->n2oe, g->nij, nii, g->ltri};
    memcpy(p, hd, sizeof hd);
    p += sizeof hd;
    for (int i = 1; i <= g->noex; i++, p += 8)
        memcpy(p, &g->terms.v[i - 1].wt, 8);
    memcpy(p, g->isqn + 1, (size_t)g->ltri * 4);
    p += (size_t)g->ltri * 4;
    memcpy(p, g->indexi + 1, (size_t)g->nij * 4);
    p += (size_t)g->nij * 4;
    memcpy(p, g->indexj + 1, (size_t)nii * 4);
    if (ac_ftn_write(g->out, b, len) != AC_FTN_OK)
        g->err = -1;
    free(b);
    g->st->records++;
    g->nij = 0;
    g->ltri = 0;
}

/* X_STORE: the last SMAT's terms for the pair (itw, jtv) */
static void store(gen *g, int noe, int n1oe, int itw, int jtv)
{
    g->noex = noe;
    g->n1oex = n1oe;
    if (noe == 0)
        return;
    g->nij++;
    if (grow((void **)&g->isqn, &g->cap_isqn, 2 * (g->ltri + noe + 1), sizeof *g->isqn) ||
        grow((void **)&g->indexi, &g->cap_i, 2 * (g->nij + 1), sizeof *g->indexi) ||
        grow((void **)&g->indexj, &g->cap_j, 2 * (g->nij + 1), sizeof *g->indexj)) {
        g->err = -1;
        return;
    }
    const ac_term *tm = g->terms.v;
    for (int i = 1; i <= n1oe; i++)
        g->isqn[g->ltri + i] = ac_addr_one(&g->addr, g->iwvad[tm[i - 1].i1],
                                           g->iwvad[tm[i - 1].i2]);
    g->n2oe = noe - n1oe;
    for (int i = n1oe + 1; i <= noe; i++)
        g->isqn[g->ltri + i] =
            ac_addr_two(&g->addr, tm[i - 1].km, g->iwvad[tm[i - 1].i1],
                        g->iwvad[tm[i - 1].i2], g->iwvad[tm[i - 1].i3],
                        g->iwvad[tm[i - 1].i4]);
    if (itw > jtv) {
        g->indexi[g->nij] = itw;
        g->indexj[g->nij] = jtv;
    } else {
        g->indexi[g->nij] = jtv;
        g->indexj[g->nij] = itw;
    }
    g->ltri += noe;
    g->st->stored_pairs++;
    if (g->ltri > g->ltrix - g->h->lijb || g->nij > g->nijx - 1)
        storeh(g);
}

/* one SMAT call through ac_smat; copies the slot maps into /COMEXP/ */
static int call_smat(gen *g, int nsi, const int16_t *ien, const int16_t *ieq,
                     const int16_t *iesl, const int16_t *irps, int nsj,
                     const int16_t *jen, const int16_t *jeq, const int16_t *jesl,
                     const int16_t *jrps, ac_smat_out *o)
{
    g->terms.len = 0;
    int rc = ac_smat(g->t, nsi, ien, ieq, iesl, irps, nsj, jen, jeq, jesl, jrps,
                     g->h->nwt, g->nwth, g->h->nwf, g->h->nwtc, &g->terms, o);
    g->st->smat_calls++;
    if (rc)
        return rc;
    /* SMAT sets these entries of /COMEXP/; the others keep their values */
    for (int k = 1; k <= o->nwst && k <= 16; k++) g->iwad[k] = o->iwad[k];
    for (int k = 1; k <= o->nvst && k <= 16; k++) g->ivad[k] = o->ivad[k];
    for (int k = 1; k <= o->nhtr; k++) g->iwvad[k] = o->iwvad[k];
    for (int k = 101; k <= 100 + o->nfsr; k++) g->iwvad[k] = o->iwvad[k];
    return 0;
}

/* EXIJ (src/exij.f) for column type at JJ (offset JA) and row type at II
   (offset IA); NWFSG(k) = h->nwfsg[k-1] */
static int gen_exij(gen *g, int ja, int jj, int ia, int ii)
{
    const int16_t *W = g->h->nwfsg - 1; /* 1-based */
    static uint32_t ibit[801][26];      /* IBIT(25,800): bits per JT */
    int noex = 0, idiag = 0;
    if (ja == ia && jj == ii)
        idiag = 1;
    int nsj = W[jj], njrps = W[jj + 1];
    int jnen = jj + 2, jneq = jnen + nsj, jnesl = jneq + nsj;
    int nsi = W[ii], nirps = W[ii + 1];
    int inen = ii + 2, ineq = inen + nsi, inesl = ineq + nsi;
    if (njrps > 800 || nirps > 800)
        return -4;
    for (int jt = 1; jt <= njrps; jt++)
        memset(ibit[jt], 0, sizeof ibit[jt]);
    int jtr = 1, itr = 1, irem = 0, jt = 1, it = 1, jadt = 0, iadt = 0;
    int noe = 0, n1oe = 0, nht = 0, nfs = 0, nhdt = 0, nsit = 0, nsjt = 0;
    ac_smat_out o;

L2100:
    irem = 0;
    {
        int ikrps = inesl + nsi * itr, jkrps = jnesl + nsj * jtr;
        if (noex > 0)
            storeh(g);
        int rc = call_smat(g, nsi, &W[inen], &W[ineq], &W[inesl], &W[ikrps], nsj,
                           &W[jnen], &W[jneq], &W[jnesl], &W[jkrps], &o);
        if (rc)
            return -2;
    }
    noe = o.noe;
    n1oe = o.n1oe;
    nht = o.nhtr;
    nfs = o.nfsr;
    nhdt = o.nhdtx;
    nsit = o.nwst;
    nsjt = o.nvst;
    noex = noe;
    jt = jtr;
L2010:
    jadt = jnesl + nsj * jt - 1;
    it = 1;
    if (idiag == 2)
        it = jt;
L2021:
    if (idiag == 1)
        jt = it;
    if (ibit[jt][(it - 1) / 32] & (1u << ((it - 1) % 32)))
        goto L2022;
    iadt = inesl + it * nsi - 1;
    for (int iwt = 1; iwt <= nsit; iwt++)
        g->iwvad[g->iwad[iwt]] = W[iwt + iadt];
    if (idiag == 1)
        goto L2050;
    for (int jwt = 1; jwt <= nsjt; jwt++) {
        int kt = g->ivad[jwt];
        if (g->iwvad[kt] == W[jwt + jadt])
            continue;
        if (kt != 130)
            goto L2023;
    }
    for (int k = 2; k <= nht; k++)
        if (g->iwvad[k - 1] >= g->iwvad[k])
            goto L2023;
    for (int k = 2; k <= nfs; k++)
        if (g->iwvad[k + 99] >= g->iwvad[k + 100])
            goto L2023;
L2050:
    if (noe == 0)
        goto L2060;
    if (nhdt != 0 || ia + it == ja + jt) {
        store(g, noe, n1oe, ia + it, ja + jt);
        if (g->err)
            return g->err;
        g->st->noet += noe;
        g->st->n1oet += n1oe;
        goto L2060;
    }
    return -3; /* L9000: identical CSFs in different places */
L2060:
    ibit[jt][(it - 1) / 32] |= 1u << ((it - 1) % 32);
    goto L2022;
L2023:
    if (irem == 1)
        goto L2022;
    irem = 1;
    jtr = jt;
    itr = it;
L2022:
    it = it + 1;
    if (it <= nirps)
        goto L2021;
    if (idiag != 1 || njrps == 1)
        goto L2029;
    idiag = 2;
    jtr = 1;
    itr = 2;
    goto L2100;
L2029:
    jadt = jadt + nsj;
    for (int jwt = 1; jwt <= nsjt; jwt++)
        g->iwvad[g->ivad[jwt]] = W[jwt + jadt];
    jt = jt + 1;
    if (jt <= njrps)
        goto L2010;
    if (irem == 1)
        goto L2100;
    if (noex > 0)
        storeh(g);
    return g->err;
}

static int write_marker(gen *g, int ja)
{
    int32_t m[8] = {ja, 0, 0, 0, 0, 0, 0, 0};
    g->st->markers++;
    return ac_ftn_write(g->out, m, sizeof m) == AC_FTN_OK ? 0 : -1;
}

void ac_nfte_nwth(const ac_cfptab *t, const ac_nfte_head *h, int32_t *nwth)
{
    for (int i = 1, j = 1; i <= h->nwt; i++) {
        int lf = h->nwtn[i - 1] % 256;
        nwth[i - 1] = ac_cfp_lsqn(t, lf - 1, lf, 0) * 256;
        while (j <= h->nwf) {
            if (h->nwtc[j - 1] > i)
                break;
            if (h->nwtc[j - 1] == i) {
                nwth[i - 1] += 2 * lf;
                break;
            }
            j++;
        }
    }
}

int ac_tape_generate(const ac_cfptab *t, const ac_nfte_head *h,
                     ac_ftn_file *out, ac_tape_stats *st, char *err,
                     int errlen)
{
    gen G;
    memset(&G, 0, sizeof G);
    memset(st, 0, sizeof *st);
    gen *g = &G;
    g->t = t;
    g->h = h;
    g->out = out;
    g->st = st;
    ac_terms_init(&g->terms);
    int rc = -4;

    /* EXTAPE SETUP: NWTH(I) = LSQN(LF, LF+1, 1)*256 (+ 2*LF if closed) */
    g->nwth = malloc((size_t)h->nwt * sizeof *g->nwth);
    if (!g->nwth) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    ac_nfte_nwth(t, h, g->nwth);
    if (ac_addr_setup(&g->addr, t, 0, h->nwt, g->nwth, NULL, h->nwsq)) {
        fail(err, errlen, "ac_addr_setup failed");
        goto out;
    }
    if (g->addr.lmax != h->lmax) {
        fail(err, errlen, "LMAX %d from the orbitals, %d on the tape", g->addr.lmax, h->lmax);
        goto out;
    }
    for (int l = 1; l <= h->lmax; l++)
        if (g->addr.issh[l] != h->issh[l - 1]) {
            fail(err, errlen, "ISSH(%d) %d from the orbitals, %d on the tape", l,
                 g->addr.issh[l], h->issh[l - 1]);
            goto out;
        }

    /* the writer's buffer limits (EXTAP1: NIJX = MXRPS**2; LTRIX as WFMAIN
       sets it, a bound that also keeps records within CALEMS's buffer) */
    g->nijx = h->mxrps * h->mxrps;
    g->ltrix = h->mxrps * h->lijb * 100;

    /* ENERGY_EXPRESSION */
    {
        const int16_t *W = h->nwfsg - 1;
        int js = 1, ja = 0;
        do {
            int nsj = W[js], njrps = W[js + 1];
            int ia = ja, ii = js;
            if (write_marker(g, ja)) {
                rc = -1;
                goto io;
            }
            do {
                int nsi = W[ii], nirps = W[ii + 1];
                int r = gen_exij(g, ja, js, ia, ii);
                if (r) {
                    rc = r;
                    fail(err, errlen, "EXIJ failed (%d) for column %d, row %d", r, ja, ia);
                    goto out;
                }
                ia += nirps;
                ii += nsi * (nirps + 3) + 2;
            } while (ii < h->nint);
            js += nsj * (njrps + 3) + 2;
            ja += njrps;
        } while (js < h->nint);
        if (write_marker(g, ja)) {
            rc = -1;
            goto io;
        }
    }
    rc = 0;
    goto out;
io:
    fail(err, errlen, "write error");
out:
    free(g->nwth);
    free(g->isqn);
    free(g->indexi);
    free(g->indexj);
    ac_terms_free(&g->terms);
    return rc;
}
