/*
 * ac_addr.c -- port of SETIAD/SETKAD/INT1AD/INT2AD (src/setiad.f).
 * Arrays are indexed from 1 like the Fortran.
 */
#include "ac_addr.h"

#include <stdlib.h>
#include <string.h>

static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

int ac_addr_setup(ac_addr *a, const ac_cfptab *t, int kad, int nwt,
                  const int32_t *nwth, const int32_t *nwlf,
                  const int32_t *nwsq)
{
    memset(a, 0, sizeof *a);
    if (nwt < 1 || nwt > AC_ADDR_MAXORB)
        return -1;
    a->nwt = nwt;
    for (int i = 1; i <= nwt; i++) {
        if (!kad)
            a->nwlf[i] = t->iconnt[nwth[i - 1] / 256 - 1][0] / 2 + 1; /* ILFO/2+1 */
        else
            a->nwlf[i] = nwlf[i - 1];
        a->nwsq[i] = nwsq ? nwsq[i - 1] : 0;
        if (a->nwlf[i] < 1 || a->nwlf[i] > AC_ADDR_MAXL)
            return -1;
    }
    int nsqfg = a->nwsq[1];
    a->lmax = 0;
    for (int i = 1; i <= nwt; i++) {
        int k = a->nwlf[i];
        a->lmax = imax(k, a->lmax);
        a->issh[k]++;
        if (nsqfg == 0)
            a->nwsq[i] = a->issh[k];
    }
    int lmax = a->lmax;

    a->itad[1] = 0;
    a->ioead[1] = 0;
    for (int i = 2; i <= lmax; i++) {
        a->ioead[i] = a->ioead[i - 1] + (a->issh[i - 1] * (a->issh[i - 1] + 1)) / 2;
        a->itad[i] = (lmax - i + 2) * (i - 1) + a->itad[i - 1];
    }
    int kmax = 2 * lmax - 1;
    int igtad = kad ? 0 : a->ioead[lmax] + (a->issh[lmax] * (a->issh[lmax] + 1)) / 2;

    a->lislu = 0;
    for (int km = 1; km <= kmax; km++) {
        int kpt = (km + 1) % 2 + 1;
        int isad = 0;
        for (int ida = kpt; ida <= lmax; ida += 2) {
            if (ida > km)
                break; /* GO TO 191 */
            for (int ita = ida; ita <= kmax; ita += 2) {
                if (ita < km)
                    continue;
                int ln1 = (ida + ita) / 2;
                if (ln1 > lmax)
                    break; /* GO TO 192 */
                int ln2 = (ita - ida) / 2 + 1;
                int kt = (ita - km) / 2;
                int iad = a->itad[ln2] + kt * (lmax - ln2 + 1) + ida;
                a->lislu = imax(iad, a->lislu);
                if (iad > AC_ADDR_LISL)
                    return -1; /* the Fortran clamps, then stops at the end */
                a->iscd[iad] = isad;
                int idima = a->issh[ln1] * a->issh[ln2];
                if (ida == 1)
                    idima = (a->issh[ln1] * (a->issh[ln1] + 1)) / 2;
                isad += idima;
                a->igcd[iad] = igtad;
                for (int idb = kpt; idb <= ida; idb += 2) {
                    int itbg = kmax;
                    if (idb == ida)
                        itbg = ita;
                    for (int itb = idb; itb <= itbg; itb += 2) {
                        if (itb < km)
                            continue;
                        int ln3 = (idb + itb) / 2;
                        if (ln3 > lmax)
                            break; /* GO TO 194 */
                        int ln4 = (itb - idb) / 2 + 1;
                        int idimb = a->issh[ln3] * a->issh[ln4];
                        if (idb == 1)
                            idimb = (a->issh[ln3] * (a->issh[ln3] + 1)) / 2;
                        int idim = idima * idimb;
                        if (ida == idb && ita == itb)
                            idim = (idima * (idima + 1)) / 2;
                        igtad += idim;
                    }
                }
            }
        }
    }
    a->igtad = igtad;
    return 0;
}

int ac_addr_one(const ac_addr *a, int m1, int m2)
{
    int l = a->nwlf[m1];
    int i1 = a->nwsq[m1], i2 = a->nwsq[m2];
    int ia = imin(i1, i2), ib = imax(i1, i2);
    return a->ioead[l] + (ib * (ib - 1)) / 2 + ia;
}

/* a pair's index within its block and the l that addresses the block */
static void pair_index(const ac_addr *a, int i1, int i2, int *idi, int *idim,
                       int *llow)
{
    int l1 = a->nwlf[i1], l2 = a->nwlf[i2];
    *idim = a->issh[l2] * a->issh[l1];
    if (l1 < l2) {
        *idi = a->issh[l2] * (a->nwsq[i1] - 1) + a->nwsq[i2];
        *llow = l1;
    } else if (l1 == l2) {
        int kt = imax(a->nwsq[i1], a->nwsq[i2]);
        *idi = (kt * (kt - 1)) / 2 + imin(a->nwsq[i1], a->nwsq[i2]);
        *idim = (a->issh[l1] * (a->issh[l1] + 1)) / 2;
        *llow = l1;
    } else {
        *idi = a->issh[l1] * (a->nwsq[i2] - 1) + a->nwsq[i1];
        *llow = l2;
    }
}

int ac_addr_two(const ac_addr *a, int kxx, int m1, int m2, int m3, int m4)
{
    int k = kxx / 2 + 1;
    int i1 = m1, i2 = m2, i3 = m3, i4 = m4;
    int ida, idb, ita, itb;
    for (;;) { /* statement 4: order the two pairs */
        int l1 = a->nwlf[i1], l2 = a->nwlf[i2], l3 = a->nwlf[i3], l4 = a->nwlf[i4];
        ida = abs(l1 - l2) + 1;
        idb = abs(l3 - l4) + 1;
        ita = l1 + l2 - 1;
        itb = l3 + l4 - 1;
        if (ida > idb || (ida == idb && ita >= itb))
            break;
        int it = i3;
        i3 = i1;
        i1 = it;
        it = i2;
        i2 = i4;
        i4 = it;
    }
    int idia, idima, l12l, idib, idimb, l34l;
    pair_index(a, i1, i2, &idia, &idima, &l12l);
    pair_index(a, i3, i4, &idib, &idimb, &l34l);
    int idta;
    if (ida == idb && ita == itb) {
        int kt = imax(idia, idib);
        idta = (kt * (kt - 1)) / 2 + imin(idia, idib);
    } else {
        idta = (idia - 1) * idimb + idib;
    }
    int kt = (ita - k) / 2;
    int iada = a->itad[l12l] + kt * (a->lmax - l12l + 1) + ida;
    kt = (itb - k) / 2;
    int iadb = a->itad[l34l] + kt * (a->lmax - l34l + 1) + idb;
    return a->igcd[iada] + a->iscd[iadb] * idima + idta;
}
