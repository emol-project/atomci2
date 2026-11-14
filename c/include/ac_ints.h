/*
 * ac_ints.h -- the transformed integral file NFTRI written by the Fortran
 * TR stage (src/trnsfm.f, read in the Fortran by CALHMD/RDINTS/PRDINT).
 *
 * Layout (gfortran unformatted sequential):
 *   header  NAME(33), NSYM, NOBT, NNUC, NFLGS, LTRB1, NOB(NSYM), MPOB(NOBT),
 *           MOB(NOBT) (int32), CHARG(NNUC), XNUC(NNUC) (real*8)
 *   S, then H, per l: records  NA, NB, NC (int32), values (real*8), each
 *           l holding NOB(l)(NOB(l)+1)/2 values in triangular order
 *   two-electron blocks: a 14-integer header NPAR, then either NPAR(11)
 *           records of K = 1..NPAR(11) values (when NPAR(2) = NPAR(4) and
 *           NPAR(3) = NPAR(5)) or NPAR(11) records of NPAR(10) values.
 * As in CALHMD, H overwrites S and the two-electron values follow, so the
 * 1-based position of a value is exactly its address from ac_addr
 * (INT1AD for h, INT2AD for R^k).
 */
#ifndef AC_INTS_H
#define AC_INTS_H

typedef struct {
    int nsym, nobt;
    int nob[20];
    long n1;  /* number of one-electron values (H)                 */
    long n;   /* total number of values; x[1..n], x[0] unused      */
    double *x;
} ac_ints;

int ac_ints_load(ac_ints *in, const char *path, char *err, int errlen);
void ac_ints_free(ac_ints *in);

#endif
