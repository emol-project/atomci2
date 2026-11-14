/*
 * ac_smat.h -- the Coulomb energy-expression terms of one CSF pair: a port of
 * the Fortran SMAT and CONVT (src/smat.f, src/convt.f).
 *
 * SMAT is specific to the non-relativistic Hamiltonian (METECC eqs. 44-46):
 * it classifies how the two CSFs differ and emits, for each case, the
 * one-electron terms h(a,b) and the two-electron terms R^k(a,b;c,d) with
 * their angular coefficients. It is ported nearly line for line (labels
 * L<n> are the Fortran statement numbers), calling ac_factor_shms for SHMS.
 * Stage 2 replaces it by operator-specific term generators.
 *
 * Shell slots in the terms are local: 1..nhtr are the open shells of the
 * pair, 100+j the j-th closed shell, 0 unused (as the Fortran ISQT).
 */
#ifndef AC_SMAT_H
#define AC_SMAT_H

#include "ac_cfp.h"

#include <stdint.h>

typedef struct {
    int km;             /* 2k+1 for two-electron terms, 0 for one-electron */
    int i1, i2, i3, i4; /* local shell slots                              */
    double wt;          /* angular coefficient                            */
} ac_term;

typedef struct {
    ac_term *v;
    int len, cap;
} ac_terms;

void ac_terms_init(ac_terms *t);
void ac_terms_free(ac_terms *t);

/* SMAT's outputs besides the terms, and the slot maps it leaves in
   /COMEXP/ for STORE: iwvad[1..nhtr] = coupling-sequence index of each open
   slot, iwvad[101..100+nfsr] that of each closed shell. */
typedef struct {
    int noe, n1oe;   /* number of terms; number of one-electron terms */
    int nhtr, nfsr;  /* open slots; closed shells                     */
    int nhdtx;       /* measure of difference between the two CSFs    */
    int nwst, nvst;  /* bra/ket shells consumed                       */
    int iwad[17], ivad[17];
    int iwvad[131];
} ac_smat_out;

enum {
    AC_SMAT_OK = 0,
    AC_SMAT_ERR_NDIF = 1,   /* "NDIF INCONSISTENT" in the Fortran         */
    AC_SMAT_ERR_FACTOR = 2, /* ac_factor_shms failed                      */
    AC_SMAT_ERR_SIZE = 3,   /* more open or closed shells than supported  */
    AC_SMAT_ERR_MEM = 4
};

/* Arguments as the Fortran SMAT receives them (Fortran arrays 1-based in
   meaning, 0-based here): bra nws shells nwen/nweq/nwesl (codes, electron
   counts, couplings 256*(2S+1)+(2L+1)) on sequence shells nwrps, ket
   likewise; nwth[nwt] the empty/closed state code of every sequence shell,
   nwtc[nwf] the default closed shells. Appends the terms to `terms`. */
int ac_smat(const ac_cfptab *t, int nws, const int16_t *nwen,
            const int16_t *nweq, const int16_t *nwesl, const int16_t *nwrps,
            int nvs, const int16_t *nven, const int16_t *nveq,
            const int16_t *nvesl, const int16_t *nvrps, int nwt,
            const int32_t *nwth, int nwf, const int32_t *nwtc, ac_terms *terms,
            ac_smat_out *out);

#endif
