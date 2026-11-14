/*
 * ac_tape.h -- the energy-expression tape NFTE: reading, and writing its
 * data records from the C engine (ports of EXTAPE's ENERGY_EXPRESSION, EXIJ,
 * and NTISQT's STORE/STOREH; src/extape.fpl, src/exij.f, src/ntisqt.fpl).
 *
 * File layout (gfortran unformatted sequential):
 *   1  NAME(30), ISPIN, IORB, IPAR, NWT, NWF, ITSTV, ITNEL, LIJB, IDIAG,
 *      LMAX, INT, LWFL, NTYPE, NREF, MXRPS                      (int32)
 *   2  ISSH(LMAX), NWTN(NWT), NWTC(NWF), NWSQ(NWT)                (int32)
 *   3.. the CSF image in chunks: KT (int32), NWFSG(KT) (int16), until
 *      INT words are written
 *   then data records
 *      ITW, JTV, NOE, N1OE, N2OE, NIJ, NII, LTRI                   (int32)
 *      WT(NOE) (real*8), ISQN(LTRI), INDEXI(NIJ), INDEXJ(NII)     (int32)
 *   where one coefficient vector WT serves NIJ CSF pairs (row INDEXI,
 *   column INDEXJ, or INDEXI itself when NII = 0) and ISQN holds NOE
 *   integral addresses per pair; a record with NOE = 0 (ITW, 0, ..., 0)
 *   marks the start of a new column type.
 */
#ifndef AC_TAPE_H
#define AC_TAPE_H

#include "ac_addr.h"
#include "ac_cfp.h"
#include "ac_ftnio.h"
#include "ac_smat.h"

#include <stdint.h>

typedef struct {
    /* record 1 */
    int32_t name[30];
    int32_t ispin, iorb, ipar, nwt, nwf, itstv, itnel, lijb, idiag, lmax,
        nint, lwfl, ntype, nref, mxrps;
    /* record 2 */
    int32_t *issh, *nwtn, *nwtc, *nwsq;
    /* the CSF image */
    int16_t *nwfsg; /* [nint] */
    /* the header records exactly as read, for a byte-identical copy */
    int nraw;
    unsigned char **raw;
    size_t *rawlen;
} ac_nfte_head;

/* Read records 1, 2 and the CSF image; leaves f at the first data record.
   Returns 0 or -1 (message in err). */
int ac_nfte_read_head(ac_ftn_file *f, ac_nfte_head *h, char *err, int errlen);
void ac_nfte_head_free(ac_nfte_head *h);

/* Write the header records exactly as they were read. */
int ac_nfte_write_head(ac_ftn_file *f, const ac_nfte_head *h);

typedef struct {
    int32_t itw, jtv, noe, n1oe, n2oe, nij, nii, ltri;
    double *wt;
    int32_t *isqn, *indexi, *indexj;
    int cap_wt, cap_isqn, cap_i, cap_j;
} ac_nfte_rec;

/* Read the next data record; returns AC_FTN_OK, AC_FTN_EOF or an error. */
int ac_nfte_read_rec(ac_ftn_file *f, ac_nfte_rec *r);
void ac_nfte_rec_free(ac_nfte_rec *r);

/* The empty/closed state code of every sequence orbital (EXTAPE SETUP):
   NWTH(i) = LSQN(LF, LF+1, 1)*256, plus 2*LF if orbital i is a default
   closed shell; nwth must hold h->nwt values. */
void ac_nfte_nwth(const ac_cfptab *t, const ac_nfte_head *h, int32_t *nwth);

/* Statistics of a generation run. */
typedef struct {
    long smat_calls, stored_pairs, records, markers;
    long noet, n1oet;
} ac_tape_stats;

/* Write the data records for the CSFs of h (the EXTAPE ENERGY_EXPRESSION
   loop: every column type J, every row type I >= J, EXIJ), after the
   header has been written. Returns 0, or a negative code: -1 I/O,
   -2 SMAT failure, -3 "identical CSF" (EXIJ's RTNCD 9999), -4 setup. */
int ac_tape_generate(const ac_cfptab *t, const ac_nfte_head *h,
                     ac_ftn_file *out, ac_tape_stats *st, char *err,
                     int errlen);

#endif
