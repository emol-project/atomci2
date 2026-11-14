/*
 * ftn_decl.h -- declarations of the Fortran routines and common blocks the
 * L0/L1 tests call (gfortran: lower case + '_', arguments by reference,
 * hidden CHARACTER lengths as size_t after all other arguments).
 */
#ifndef FTN_DECL_H
#define FTN_DECL_H

#include <stddef.h>
#include <stdint.h>

void ftnload_(const char *path, size_t path_len);
void setwig_(void);

double xwig6j_(const int32_t *, const int32_t *, const int32_t *,
               const int32_t *, const int32_t *, const int32_t *);
double xwig9x_(const int32_t *, const int32_t *, const int32_t *,
               const int32_t *, const int32_t *, const int32_t *,
               const int32_t *, const int32_t *, const int32_t *);
double f3l_(const int32_t *, const int32_t *, const int32_t *);
double delta_(const int32_t *, const int32_t *, const int32_t *);

double xcfp_(const int32_t *im, const int32_t *km, const int32_t *kmp,
             const int32_t *id);
int32_t isp_(const int32_t *jm);
int32_t iqs_(const int32_t *jm);
int32_t ilfo_(const int32_t *jm);
int32_t ior_(const int32_t *jm);
int32_t lsqn_(const int32_t *l, const int32_t *iq, const int32_t *is);

void setiad_(const int32_t *nwt, const int32_t *nwth, int32_t *issh,
             const int32_t *idprnt, const int32_t *nwsq, int32_t *lmax);
void setkad_(const int32_t *nwt, const int32_t *nwlf, int32_t *issh,
             const int32_t *idprnt, const int32_t *nwsq, int32_t *lmax);
void int1ad_(const int32_t *m1, const int32_t *m2, int32_t *adr);
void int2ad_(const int32_t *kxx, const int32_t *m1, const int32_t *m2,
             const int32_t *m3, const int32_t *m4, int32_t *adr);

/* include/COMSTR */
extern struct {
    int32_t lijb, lijbu;
    double wt[2048];
    int32_t kmt[2048], i1t[2048], i2t[2048], i3t[2048], i4t[2048];
    int32_t ltrix, ltrixu, isqn[32768];
    int32_t nijx, nijxu, indexi[16384], indexj[16384];
    int32_t lisl, lislu, igcd[2048], iscd[2048];
} comstr_;

/* include/COMCFP */
extern struct {
    double cmttbl[8000];
    int32_t iname[18];
    int32_t ibel, icnn, ilfn, icmtn, icdx;
    int32_t immint[18], ilstad[18];
    int32_t iconnt[100][9];
    int32_t icmtdx[100][5];
    int32_t ilfunc[600];
    int32_t ibels, icnns, ilfns, icmtns, icdxs;
} comcfp_;

/* include/COMWFA */
extern struct {
    int32_t topadr[18];
    int32_t nshl[400];
    int32_t nel[400];
    int32_t wfadr[400];
    int32_t wf[2400];
    int32_t ws[2400];
    int32_t wr[2400];
    int32_t nsymax, nstmax, nst2max;
} comwfa_;

#endif
