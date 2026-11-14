/*
 * ac_basis.h -- the radial basis and the MO coefficients of an ATOMCI run
 * (stage 2, S2-M3).
 *
 * Basis: the header of the AO integral file NFAII (unit 18), written by
 * AIISTF (STO) or AIIGT4 (GTO):
 *   NAME(33), LMAX, NWT, I1, I1, LOEI, ISSH(LMAX), MPOB(NWT), MOB(NWT)  int32
 *   Z, Z0                                                              real*8
 *   NPQN(NWT) int32, DEL(NWT) real*8
 * and for GTO additionally
 *   IFST(LMAX), INXT(NWT), NTERM(NWT) int32, CE(40,150), OE(40,150) real*8
 * (DEL already squared there; CE still the coefficients of normalized
 * primitives, which AIIGT4 scales by OE^((2n+1)/4) only after writing).
 * Radial functions, normalized to integral R^2 r^2 dr = 1:
 *   STO  R(r) = N r^(n-1) exp(-DEL r)
 *   GTO  R(r) = N sum_j CE(j) r^(n-1) exp(-DEL OE(j) r^2)
 * Basis functions of one l are taken in the order the integral routines
 * use: contiguous by l for STO, the IFST/INXT chain for GTO.
 *
 * MO coefficients: the file written by WRTCP (unit NFTRC2 = 21 after TR):
 *   NCOD, LTRB, NSYM, NBF(20), NOB(20), NTAB(21)   then per l records
 *   NT, C(NT) holding NBF(l)*NOB(l) coefficients, basis index fastest.
 */
#ifndef AC_BASIS_H
#define AC_BASIS_H

#define AC_BASIS_MAXL 20
#define AC_BASIS_MAXPRIM 40

typedef struct {
    int l, n;                       /* orbital momentum, power r^(n-1)      */
    int gto;                        /* 0 STO, 1 GTO                          */
    int nprim;
    double alpha[AC_BASIS_MAXPRIM]; /* exponents of r (STO) or r^2 (GTO)     */
    double coef[AC_BASIS_MAXPRIM];  /* contraction coefficients, normalized  */
} ac_bfn;

typedef struct {
    double z;                      /* nuclear charge                         */
    int gto;
    int nbf;
    ac_bfn *bf;                    /* [nbf], grouped by l                    */
    int nl;                        /* number of l blocks (LMAX)              */
    int first[AC_BASIS_MAXL];      /* [l] index of the first function of l   */
    int count[AC_BASIS_MAXL];      /* [l] number of functions of l           */
    /* MO coefficients */
    int nob[AC_BASIS_MAXL];        /* [l] number of MOs                      */
    double *c[AC_BASIS_MAXL];      /* [l] count[l] x nob[l], basis fastest   */
} ac_basis;

int ac_basis_load(ac_basis *b, const char *ao_path, const char *vec_path,
                  char *err, int errlen);
void ac_basis_free(ac_basis *b);

/* R(r) of basis function i */
double ac_bfn_value(const ac_bfn *f, double r);

/* analytic integral of R_i R_j r^(2+p) dr over r >= 0 (p > -2l_i-l_j-3) */
double ac_bfn_moment(const ac_bfn *fi, const ac_bfn *fj, int p);

/* analytic <i| -1/2 nabla^2 - Z/r |j> (same l) */
double ac_bfn_h(const ac_bfn *fi, const ac_bfn *fj, double z);

/* the radial Laplacian (L R)(r) = R'' + 2R'/r - l(l+1)R/r^2 of a basis
   function, and the analytic integral of (L R_i)(L R_j) r^2 dr (same l),
   i.e. <nabla^2 i|nabla^2 j> = <i|p^4|j> */
double ac_bfn_lap(const ac_bfn *f, double r);
double ac_bfn_lap2(const ac_bfn *fi, const ac_bfn *fj);

#endif
