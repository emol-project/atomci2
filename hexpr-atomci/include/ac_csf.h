/*
 * ac_csf.h -- one side (bra or ket) of a configuration state function.
 *
 * A CSF couples its open shells one by one in coupling order (METECC eq. 39):
 *   F_0 = vacuum,  F_k = (f_k F_{k-1})^{F_k},  k = 1..nsh,
 * where f_k is the shell state of shell k (a CFPTBL code with an electron
 * count). Angular labels are (2S, 2L) pairs.
 */
#ifndef AC_CSF_H
#define AC_CSF_H

#include "ac_cfp.h"

#include <stdint.h>

#define AC_CSF_MAXSH 32

typedef struct {
    int ts; /* 2S */
    int tl; /* 2L */
} ac_lab;

typedef struct {
    int nsh;                       /* number of open shells             */
    int l[AC_CSF_MAXSH + 1];       /* [1..nsh] orbital l of each shell   */
    int code[AC_CSF_MAXSH + 1];    /* [1..nsh] shell-state code          */
    int nel[AC_CSF_MAXSH + 1];     /* [1..nsh] electrons in the shell    */
    ac_lab f[AC_CSF_MAXSH + 1];    /* [1..nsh] shell-state label (S, L)  */
    ac_lab cpl[AC_CSF_MAXSH + 1];  /* [0..nsh] F_k; cpl[0] = (0,0)       */
} ac_csf;

/* Build one side from the arrays ATOMCI's SHMS receives (multiplicities):
   nhlf[k] = 2l+1, codes[k], nel[k], csl[k] = 256*(2S+1) + (2L+1) of F_k,
   for k = 0..nht-1 (Fortran 1..NHT). Returns 0, or -1 for a bad code or a
   shell whose l does not match its code. */
int ac_csf_from_shms(ac_csf *c, const ac_cfptab *t, int nht,
                     const int32_t *nhlf, const int32_t *codes,
                     const int32_t *nel, const int32_t *csl);

#endif
