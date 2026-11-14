/*
 * ac_smat.c -- port of the Fortran SMAT and CONVT (see ac_smat.h).
 *
 * The control flow follows src/smat.f statement by statement; each label
 * L<n> is the Fortran statement n. Arrays are indexed from 1 like the
 * Fortran (element 0 unused).
 */
#include "ac_smat.h"
#include "ac_factor.h"
#include "ac_wigner.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAXH 16   /* open slots + 1 (Fortran: 12) */
#define MAXF 40   /* closed shells (Fortran NFS: 30) */
#define SLES 1.0e-10
#define BYT 256

void ac_terms_init(ac_terms *t)
{
    memset(t, 0, sizeof *t);
}

void ac_terms_free(ac_terms *t)
{
    free(t->v);
    memset(t, 0, sizeof *t);
}

typedef struct {
    const ac_cfptab *t;
    ac_terms *terms;
    int err;
    /* the pair, as SMAT hands it to SHMS (Fortran 1-based) */
    int nht;
    int32_t nhlf[MAXH + 1], nhst[MAXH + 1];
    int32_t nhwn[MAXH + 1], nhwq[MAXH + 1], nhwcsl[MAXH + 1];
    int32_t nhvn[MAXH + 1], nhvq[MAXH + 1], nhvcsl[MAXH + 1];
} sm;

/* ISQT: append a term (the Fortran also numbers it; NOE is counted by
   the caller exactly as there) */
static void isqt(sm *s, int km, int i1, int i2, int i3, int i4, double wt)
{
    ac_terms *t = s->terms;
    if (t->len == t->cap) {
        int ncap = t->cap ? 2 * t->cap : 64;
        ac_term *nv = realloc(t->v, (size_t)ncap * sizeof *nv);
        if (!nv) {
            s->err = AC_SMAT_ERR_MEM;
            return;
        }
        t->v = nv;
        t->cap = ncap;
    }
    ac_term *x = &t->v[t->len++];
    x->km = km;
    x->i1 = i1;
    x->i2 = i2;
    x->i3 = i3;
    x->i4 = i4;
    x->wt = wt;
}

/* SHMS through the general engine (normal-ordered, as the Fortran) */
static double shms(sm *s, int c1, int a1, int lsi, int lri, int c2, int a2,
                   int lsf, int lrf)
{
    int err;
    double w = ac_factor_shms(s->t, s->nht, s->nhlf + 1, s->nhwn + 1, s->nhwq + 1,
                              s->nhwcsl + 1, s->nhvn + 1, s->nhvq + 1,
                              s->nhvcsl + 1, c1, a1, lsi, lri, c2, a2, lsf, lrf,
                              1, 1, 1, &err);
    if (err)
        s->err = AC_SMAT_ERR_FACTOR;
    return w;
}

/* FSUBK(L,L1,L2,L3,L4) of smat.f, multiplicity arguments: H^k of METECC
   (45). Unlike the Fortran F3L, ac_hk gives 0 for a parity-forbidden k. */
static double fsubk(int L, int L1, int L2, int L3, int L4)
{
    return ac_hk((L - 1) / 2, (L1 - 1) / 2, (L2 - 1) / 2, (L3 - 1) / 2, (L4 - 1) / 2);
}

/* ILFO: 2l+1 of a code (also of the NWTH codes, whose low byte is an
   electron count rather than an L index) */
static int ilfo(const ac_cfptab *t, int code)
{
    return t->iconnt[code / BYT - 1][0];
}

/* WIG6J with multiplicity arguments */
static double wig6j(int a, int b, int c, int d, int e, int f)
{
    return ac_sixj(a - 1, b - 1, c - 1, d - 1, e - 1, f - 1);
}

/* ---- CONVT (src/convt.f): direct and exchange between open shells i, j ---- */
static void convt(sm *s, int i, int j, int *noe)
{
    double F[42];
    int lfi = s->nhlf[i], lfj = s->nhlf[j];
    double t11 = wig6j(2, 2, 1, 2, 2, 1);
    double t13 = wig6j(2, 2, 3, 2, 2, 1) * sqrt(3.0);
    int kmin = abs(lfi - lfj) + 1;
    int kmax = lfi + lfj - 1;
    int kt = (kmax - kmin) / 2 + 1;
    for (int kst = 1; kst <= kt; kst++) {
        int km = 2 * kst - 1;
        double w = shms(s, i, i, 1, km, j, j, 1, km);
        if (kst % 2 != 0 && fabs(w) >= SLES) {
            ++*noe;
            isqt(s, km, i, i, j, j, fsubk(km, lfi, lfi, lfj, lfj) * w * 2.0);
        }
        double wa = shms(s, i, i, 3, km, j, j, 3, km);
        F[kst] = t11 * w + t13 * wa;
    }
    int kta = (kt + 1) / 2;
    int km = kmin;
    for (int kst = 1; kst <= kta; kst++, km += 4) {
        double w = 0.0;
        for (int kmt = 1; kmt <= kt; kmt++) {
            int kmm = kmt * 2 - 1;
            w += F[kmt] * sqrt((double)(km * kmm)) * wig6j(lfi, lfi, kmm, lfj, lfj, km);
        }
        if (fabs(w) < SLES)
            continue;
        ++*noe;
        isqt(s, km, i, j, j, i, fsubk(km, lfi, lfj, lfj, lfi) * w * 2.0);
    }
}

int ac_smat(const ac_cfptab *t, int nws, const int16_t *nwen,
            const int16_t *nweq, const int16_t *nwesl, const int16_t *nwrps,
            int nvs, const int16_t *nven, const int16_t *nveq,
            const int16_t *nvesl, const int16_t *nvrps, int nwt,
            const int32_t *nwth0, int nwf, const int32_t *nwtc0, ac_terms *terms,
            ac_smat_out *out)
{
    sm S;
    memset(&S, 0, sizeof S);
    S.t = t;
    S.terms = terms;
    sm *s = &S;
    memset(out, 0, sizeof *out);
    int start_len = terms->len;

    /* 1-based views of the inputs */
#define NWEN(k) nwen[(k)-1]
#define NWEQ(k) nweq[(k)-1]
#define NWESL(k) nwesl[(k)-1]
#define NWRPS(k) nwrps[(k)-1]
#define NVEN(k) nven[(k)-1]
#define NVEQ(k) nveq[(k)-1]
#define NVESL(k) nvesl[(k)-1]
#define NVRPS(k) nvrps[(k)-1]
#define NWTH(k) nwth0[(k)-1]
#define NWTC(k) nwtc0[(k)-1]
    (void)nwt;

    int32_t *NHLF = s->nhlf, *NHST = s->nhst;
    int32_t *NHWN = s->nhwn, *NHWQ = s->nhwq, *NHWCSL = s->nhwcsl;
    int32_t *NHVN = s->nhvn, *NHVQ = s->nhvq, *NHVCSL = s->nhvcsl;
    int *IWAD = out->iwad, *IVAD = out->ivad, *IWVAD = out->iwvad;
    int NFS[MAXF + 1], NHD[MAXH + 1], NDIF[10], NSND[4][10];
    int NHT, NHDT, KWE, KVE, NFT, NFTI, NOE, N1OE, NFST, KWEN, KVEN, ICASE = 3;
    int NHSTC, ID, KT, IA = 0, NHT1, NL1 = 0, NL2 = 0, NL3 = 0, NL4 = 0, IG = 0;
    int LRL, LRG, NFX, LFFS, LHHS, KM, KMAX, NBM;
    double W, WT, WU = 0.0, AT;
    memset(NSND, 0, sizeof NSND);
    memset(NHD, 0, sizeof NHD);

    /* L112 */
    NHT = 0;
    NHDT = 0;
    for (int k = 1; k <= 9; k++)
        NDIF[k] = 0;
    KWE = 1;
    KVE = 1;
    NFT = 0;
    NFTI = 1;
    NHWCSL[1] = BYT + 1;
    NHVCSL[1] = BYT + 1;
    NOE = 0;
    N1OE = 0;
    NFST = 0;
    KWEN = 256;
    KVEN = 256;
    if (KWE <= nws) KWEN = NWRPS(KWE);
    if (KVE <= nvs) KVEN = NVRPS(KVE);

L134:
    if (KWEN != 256 || KVEN != 256) goto L133;
    if (NHDT == 1 || NHDT == 3) goto L3200;
    goto L311;
L122:
    if (KWE > 1 && IWAD[KWE - 1] == NHT) IWAD[KWE - 1] = NFT + 100;
    if (KVE > 1 && IVAD[KVE - 1] == NHT) IVAD[KVE - 1] = NFT + 100;
L1222:
    NHT = NHT - 1;
    goto L134;
L1224:
    if (KWE > 1 && IWAD[KWE - 1] == NHT) IWAD[KWE - 1] = 130;
    if (KVE <= 1) goto L1222;
    if (IVAD[KVE - 1] == NHT) IVAD[KVE - 1] = 130;
    goto L1222;
L133:
    NHT = NHT + 1;
    if (NHT >= MAXH) return AC_SMAT_ERR_SIZE;
    if (KWEN == KVEN) goto L143;
    if (KWEN > KVEN) goto L163;
    NHVQ[NHT] = NWTH(KWEN) & 255;
    NHVN[NHT] = NWTH(KWEN) - NHVQ[NHT] + 1;
    if (NHT != 1) NHVCSL[NHT] = NHVCSL[NHT - 1];
    ICASE = 2;
    NHWQ[NHT] = NWEQ(KWE);
    NHWN[NHT] = NWEN(KWE);
    NHWCSL[NHT] = NWESL(KWE);
    NHST[NHT] = KWEN;
    IWAD[KWE] = NHT;
    KWE = KWE + 1;
    KWEN = 256;
    if (KWE <= nws) KWEN = NWRPS(KWE);
    goto L141;
L163:
    NHWQ[NHT] = NWTH(KVEN) & 255;
    NHWN[NHT] = NWTH(KVEN) - NHWQ[NHT] + 1;
    if (NHT != 1) NHWCSL[NHT] = NHWCSL[NHT - 1];
    ICASE = 1;
L144:
    NHVQ[NHT] = NVEQ(KVE);
    NHVN[NHT] = NVEN(KVE);
    NHVCSL[NHT] = NVESL(KVE);
    NHST[NHT] = KVEN;
    IVAD[KVE] = NHT;
    KVE = KVE + 1;
    KVEN = 256;
    if (KVE <= nvs) KVEN = NVRPS(KVE);
    goto L141;
L143:
    ICASE = 3;
    NHWQ[NHT] = NWEQ(KWE);
    NHWN[NHT] = NWEN(KWE);
    NHWCSL[NHT] = NWESL(KWE);
    NHST[NHT] = KWEN;
    IWAD[KWE] = NHT;
    KWE = KWE + 1;
    KWEN = 256;
    if (KWE <= nws) KWEN = NWRPS(KWE);
    goto L144;
L141:
    NHSTC = NHST[NHT];
    /* L232/L242: default closed shells before this one become closed */
    while (NFTI <= nwf) {
        int d = NWTC(NFTI) - NHSTC;
        if (d > 0)
            break;
        if (d < 0) {
            if (NFT >= MAXF) return AC_SMAT_ERR_SIZE;
            NFT = NFT + 1;
            NFS[NFT] = NWTC(NFTI);
        }
        NFTI = NFTI + 1;
    }
    /* L252 */
    if (NHWQ[NHT] != NHVQ[NHT]) goto L164;
    if (NHWQ[NHT] == 0) goto L1224;
    if (NHWQ[NHT] != 2 * ilfo(t, NWTH(NHSTC))) goto L164;
    if (NFT >= MAXF) return AC_SMAT_ERR_SIZE;
    NFT = NFT + 1;
    NFS[NFT] = NHSTC;
    goto L122;
L164:
    NHD[NHT] = NHWQ[NHT] - NHVQ[NHT];
    NHDT = NHDT + abs(NHD[NHT]);
    if (abs(NHD[NHT]) > 2) goto L3200;
    if (abs(NHD[NHT]) != 1) goto L1641;
    if (ac_cfp_xcfp(t, NHWN[NHT], NHVN[NHT], NHVQ[NHT], NHD[NHT]) != 0.0) goto L1641;
    NHDT = NHDT + 2;
    NHD[NHT] = 3 * NHD[NHT];
L1641:
    if (NHD[NHT] == 0 && NHWN[NHT] != NHVN[NHT]) goto L185;
    if (NHD[NHT] == 0 && NHWN[NHT] == NHVN[NHT] && NHDT == 0 &&
        NHWCSL[NHT] != NHVCSL[NHT])
        goto L1851;
    goto L186;
L1851:
    NHD[NHT] = 5;
    NHDT = NHDT + 2;
    goto L186;
L185:
    NHD[NHT] = 4;
    NHDT = NHDT + 2;
L186:
    if (NHDT > 4) goto L3200;
    ID = NHD[NHT] + 4;
    NHLF[NHT] = ilfo(t, NHWN[NHT]);
    if (ID == 4) goto L134;
    KT = NDIF[ID] + 1;
    NDIF[ID] = KT;
    if (KT > 3) return AC_SMAT_ERR_SIZE;
    NSND[KT][ID] = NHT;
    goto L134;

    /* CSFs too different: no terms */
L3200:
    if (ICASE == 1) goto L3203;
    if (ICASE == 2) goto L3204;
    goto L3205;
L3203:
    if (KWEN == 256) goto L3205;
    IA = KWEN;
    IWAD[KWE] = NHT + 1;
    KWE = KWE + 1;
    goto L3206;
L3204:
    if (KVEN == 256) goto L3205;
    IA = KVEN;
    IVAD[KVE] = NHT + 1;
    KVE = KVE + 1;
L3206:
    NHT = NHT + 1;
    NHST[NHT] = IA;
L3205:
    for (int k = 1; k <= NHT; k++) IWVAD[k] = NHST[k];
    for (int k = 1; k <= NFT; k++) IWVAD[k + 100] = NFS[k];
    out->nhtr = NHT;
    out->nfsr = NFT;
    out->nhdtx = NHDT;
    out->nwst = KWE - 1;
    out->nvst = KVE - 1;
    out->noe = NOE;
    out->n1oe = N1OE;
    return AC_SMAT_OK;

L311:
    NHT1 = NHT + 1;
    NHLF[NHT1] = 1;
    s->nht = NHT;
    while (NFTI <= nwf) { /* L3114 */
        if (NFT >= MAXF) return AC_SMAT_ERR_SIZE;
        NFT = NFT + 1;
        NFS[NFT] = NWTC(NFTI);
        NFTI = NFTI + 1;
    }
    for (int k = 1; k <= NFT; k++) IWVAD[k + 100] = NFS[k];
    for (int k = 1; k <= NHT; k++) IWVAD[k] = NHST[k];
    out->nhtr = NHT;
    out->nfsr = NFT;
    out->nhdtx = NHDT;
    out->nwst = nws;
    out->nvst = nvs;
    if (NHDT == 0) goto L561;
    if (NHDT != 2) goto L342;
    if (NDIF[3] == 0 && NDIF[8] == 1) goto L413;
    if (NDIF[3] == 0 && NDIF[9] == 1) goto L4131;
    goto L334;
L342:
    if (NDIF[3] == 2 && NDIF[5] == 2) goto L352;
    if (NDIF[8] == 1 && NDIF[3] == 1) goto L363;
    if (NDIF[9] == 1 && NDIF[3] == 1) goto L3631;
    if (NDIF[6] == 1 && NDIF[2] == 1) goto L365;
    if (NDIF[6] == 1 && NDIF[3] == 2) goto L622;
    if (NDIF[2] == 1 && NDIF[5] == 2) goto L623;
    if (NDIF[7] == 1 && NDIF[3] == 1) goto L624;
    if (NDIF[1] == 1 && NDIF[5] == 1) goto L625;
    if (NDIF[8] == 1 && NDIF[9] == 1) goto L3641;
    if (NDIF[8] == 2) goto L364;
    out->noe = NOE;
    out->n1oe = N1OE;
    return AC_SMAT_ERR_NDIF;

L3641:
    NL1 = NSND[1][9];
    for (int na = 1; na <= NL1; na++)
        convt(s, NSND[1][8], na, &NOE);
    goto L383;
L352:
    NL1 = NSND[1][5];
    NL2 = NSND[1][3];
    NL3 = NSND[2][5];
    NL4 = NSND[2][3];
    IG = 1;
L392:
    LRL = (abs(NHLF[NL1] - NHLF[NL2]) > abs(NHLF[NL3] - NHLF[NL4])
               ? abs(NHLF[NL1] - NHLF[NL2])
               : abs(NHLF[NL3] - NHLF[NL4])) + 1;
    LRG = (NHLF[NL1] + NHLF[NL2] < NHLF[NL3] + NHLF[NL4]
               ? NHLF[NL1] + NHLF[NL2]
               : NHLF[NL3] + NHLF[NL4]) - 1;
L372:
    if (LRL > LRG) {
        switch (IG) {
        case 1: goto L382;
        case 2: goto L383;
        case 3: goto L383;
        case 4: goto L421;
        case 5: goto L3632;
        default: goto L3822;
        }
    }
    W = shms(s, NL1, NL2, 1, LRL, NL3, NL4, 1, LRL);
    if (fabs(W) < SLES) goto L373;
    NOE = NOE + 1;
    WT = (W + W) * fsubk(LRL, NHLF[NL1], NHLF[NL2], NHLF[NL3], NHLF[NL4]);
    if (IG == 3) WT = WT * 0.5;
    isqt(s, LRL, NL1, NL2, NL3, NL4, WT);
L373:
    LRL = LRL + 4;
    goto L372;
L382:
    IG = IG + 1;
    {
        int nt = NL2;
        NL2 = NL4;
        NL4 = nt;
    }
    goto L392;
L3632:
    NL1 = NL1 + 1;
    NL2 = NL1;
    if (NL1 <= NSND[1][9]) goto L392;
    IG = 6;
    NL2 = NL4;
    NL1 = 1;
    NL4 = 1;
    goto L392;
L3822:
    NL1 = NL1 + 1;
    NL4 = NL1;
    if (NL1 <= NSND[1][9]) goto L392;
    goto L383;
L3631:
    NL1 = 1;
    NL2 = 1;
    NL3 = NSND[1][5];
    NL4 = NSND[1][3];
    IG = 5;
    goto L392;
L363:
    NL1 = NSND[1][8];
    NL2 = NL1;
    NL3 = NSND[1][5];
    NL4 = NSND[1][3];
    IG = 1;
    goto L392;
L364:
    convt(s, NSND[1][8], NSND[2][8], &NOE);
    goto L383;
L365:
    NL1 = NSND[1][6];
    NL3 = NL1;
    NL2 = NSND[1][2];
    NL4 = NL2;
    IG = 3;
    goto L392;
L622:
    NL1 = NSND[1][6];
    NL2 = NSND[1][3];
    NL4 = NSND[2][3];
    NL3 = NL1;
    IG = 2;
    goto L392;
L623:
    NL1 = NSND[1][5];
    NL2 = NSND[1][2];
    NL3 = NSND[2][5];
    NL4 = NL2;
    IG = 2;
    goto L392;
L624:
    NL1 = NSND[1][7];
    NL2 = NSND[1][3];
    NL3 = NL1;
    NL4 = NL1;
    IG = 2;
    goto L392;
L625:
    NL3 = NSND[1][5];
    NL2 = NSND[1][1];
    NL1 = NL2;
    NL4 = NL2;
    IG = 2;
    goto L392;

    /* single excitation: one-electron term, closed-shell and open-shell
       two-electron terms */
L334:
    NL1 = NSND[1][5];
    NL2 = NSND[1][3];
    W = shms(s, NL1, NL2, 1, 1, NHT1, NHT1, 1, 1);
    if (fabs(W) < SLES) goto L431;
    NOE = NOE + 1;
    WT = -sqrt((double)(2 * NHLF[NL1])) * W;
    isqt(s, 0, NL1, NL2, 0, 0, WT);
    WU = WT;
    N1OE = NOE;
    for (NFX = 1; NFX <= NFT; NFX++) { /* L462 */
        NFST = NFS[NFX];
        LFFS = ilfo(t, NWTH(NFST));
        LHHS = NHLF[NL1];
        KM = 1;
        NOE = NOE + 1;
        WT = -W * fsubk(KM, LHHS, LHHS, LFFS, LFFS) * sqrt((double)(2 * LFFS)) * 2.0;
        isqt(s, KM, NL1, NL2, NFX + 100, NFX + 100, WT);
        KM = abs(LFFS - LHHS) + 1;
        KMAX = LFFS + LHHS - 1;
        AT = -WU / (double)LHHS;
        for (; KM <= KMAX; KM += 4) { /* L472 */
            NOE = NOE + 1;
            WT = AT * sqrt((double)KM) * fsubk(KM, LHHS, LFFS, LFFS, LHHS);
            isqt(s, KM, NL1, NFX + 100, NFX + 100, NL2, WT);
        }
    }
L431:
    for (int nlt = 1; nlt <= NHT; nlt++) {
        if (nlt == NL1 || nlt == NL2)
            continue;
        int lf1 = NHLF[NL1], lf2 = NHLF[NL2], lft = NHLF[nlt];
        KM = abs(lf1 - lf2) + 1;
        KMAX = (lf1 + lf2 < 2 * lft ? lf1 + lf2 : 2 * lft) - 1;
        for (; KM <= KMAX; KM += 4) { /* L434 */
            W = shms(s, NL1, NL2, 1, KM, nlt, nlt, 1, KM);
            if (fabs(W) < SLES)
                continue;
            NOE = NOE + 1;
            WT = W * fsubk(KM, lf1, lf2, lft, lft) * 2.0;
            isqt(s, KM, NL1, NL2, nlt, nlt, WT);
        }
        /* L435 */
        KM = (abs(lf1 - lft) > abs(lf2 - lft) ? abs(lf1 - lft) : abs(lf2 - lft)) + 1;
        KMAX = (lf1 < lf2 ? lf1 : lf2) + lft - 1;
        for (; KM <= KMAX; KM += 4) { /* L437 */
            W = shms(s, nlt, NL2, 1, KM, NL1, nlt, 1, KM);
            if (fabs(W) < SLES)
                continue;
            NOE = NOE + 1;
            WT = W * fsubk(KM, lft, lf2, lf1, lft) * 2.0;
            isqt(s, KM, nlt, NL2, NL1, nlt, WT);
        }
    }
    NL3 = NL1;
    NL4 = NL1;
    IG = 4;
    goto L392;
L421:
    NL3 = NL1;
    NL1 = NL2;
    NL4 = NL2;
    IG = 2;
    goto L392;

L413:
    NL1 = NSND[1][8];
    for (int na = 1; na <= NHT; na++)
        if (na != NL1)
            convt(s, na, NL1, &NOE);
    NL2 = NL1;
    NL3 = NL1;
    NL4 = NL1;
    IG = 3;
    goto L392;
L4131:
    NL1 = NSND[1][9];
    NBM = NL1 + 1;
    if (NBM > NHT) goto L383;
    for (int na = 1; na <= NL1; na++)
        for (int nb = NBM; nb <= NHT; nb++)
            convt(s, na, nb, &NOE);
    goto L383;

    /* diagonal element */
L561:
    for (int k = 1; k <= NFT; k++) { /* closed shells, one-electron */
        NFST = NFS[k];
        NOE = NOE + 1;
        WT = (double)(2 * ilfo(t, NWTH(NFST)));
        isqt(s, 0, k + 100, k + 100, 0, 0, WT);
    }
    for (int k = 1; k <= NHT; k++) { /* L333: open shells, one-electron */
        W = shms(s, k, k, 1, 1, NHT1, NHT1, 1, 1);
        if (fabs(W) < SLES)
            continue;
        NOE = NOE + 1;
        WT = -sqrt((double)(2 * ilfo(t, NHWN[k]))) * W;
        isqt(s, 0, k, k, 0, 0, WT);
    }
    /* L572: closed-closed */
    N1OE = NOE;
    for (int i = 1; i <= NFT; i++) {
        int lfi = ilfo(t, NWTH(NFS[i]));
        for (int j = 1; j < i; j++) {
            int lfj = ilfo(t, NWTH(NFS[j]));
            KM = 1;
            NOE = NOE + 1;
            WT = fsubk(KM, lfi, lfi, lfj, lfj) * sqrt((double)(4 * lfi * lfj)) * 2.0;
            isqt(s, KM, i + 100, i + 100, j + 100, j + 100, WT);
            for (KM = abs(lfi - lfj) + 1; KM <= lfi + lfj - 1; KM += 4) {
                NOE = NOE + 1;
                WT = -fsubk(KM, lfi, lfj, lfj, lfi) * sqrt((double)KM) * 2.0;
                isqt(s, KM, i + 100, j + 100, j + 100, i + 100, WT);
            }
        }
        for (KM = 1; KM <= 2 * lfi - 1; KM += 4) { /* L552 */
            NOE = NOE + 1;
            W = (KM == 1) ? (double)(2 * lfi - 1) : -sqrt((double)KM);
            WT = fsubk(KM, lfi, lfi, lfi, lfi) * W;
            isqt(s, KM, i + 100, i + 100, i + 100, i + 100, WT);
        }
    }
    /* L592: open-closed */
    for (int i = 1; i <= NHT; i++) {
        int lfi = NHLF[i], npi = NHWQ[i];
        for (int j = 1; j <= NFT; j++) {
            int lfj = ilfo(t, NWTH(NFS[j]));
            KM = 1;
            NOE = NOE + 1;
            WT = fsubk(KM, lfi, lfi, lfj, lfj) * sqrt((double)(4 * lfi * lfj)) *
                 (double)npi * 2.0 / (double)(2 * lfi);
            isqt(s, KM, i, i, j + 100, j + 100, WT);
            AT = -2.0 * (double)npi / (double)(2 * lfi);
            for (KM = abs(lfi - lfj) + 1; KM <= lfi + lfj - 1; KM += 4) {
                NOE = NOE + 1;
                WT = fsubk(KM, lfi, lfj, lfj, lfi) * AT * sqrt((double)KM);
                isqt(s, KM, i, j + 100, j + 100, i, WT);
            }
        }
    }
    /* L512: open-open */
    for (int i = 1; i <= NHT; i++) {
        int lfi = NHLF[i], npi = NHWQ[i];
        for (int j = 1; j < i; j++)
            convt(s, i, j, &NOE);
        if (npi < 2)
            continue;
        for (KM = 1; KM <= 2 * lfi - 1; KM += 4) { /* L524 */
            W = shms(s, i, i, 1, KM, i, i, 1, KM);
            if (fabs(W) < SLES)
                continue;
            NOE = NOE + 1;
            WT = W * fsubk(KM, lfi, lfi, lfi, lfi);
            isqt(s, KM, i, i, i, i, WT);
        }
    }

L383:
    out->noe = NOE;
    out->n1oe = N1OE;
    if (s->err)
        return s->err;
    if (terms->len - start_len != NOE) /* every NOE increment has its isqt */
        return AC_SMAT_ERR_MEM;
    return AC_SMAT_OK;
}
