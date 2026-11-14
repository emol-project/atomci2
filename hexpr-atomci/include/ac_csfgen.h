/*
 * ac_csfgen.h -- all CSFs of a configuration (stage 2).
 *
 * A configuration is an occupation number for every shell of a fixed shell
 * list. Each shell, empty and closed ones included, is a slot of the CSF.
 * A CSF chooses a shell state (CFPTBL code) for each slot and couples the
 * slots one by one in list order (METECC eq. 39). Every possible chain of
 * intermediate (S, L) is generated, whatever the final (S, L).
 */
#ifndef AC_CSFGEN_H
#define AC_CSFGEN_H

#include "ac_cfp.h"
#include "ac_csf.h"

typedef struct {
    ac_csf *v;
    int len, cap;
} ac_csflist;

void ac_csflist_init(ac_csflist *s);
void ac_csflist_free(ac_csflist *s);

/* Append every CSF of the configuration occ[0..nsh-1] over shells with
   orbital momenta l[0..nsh-1]. Returns the number appended, or -1 if a
   shell has no state list for its occupation (a truncated shell in
   CFPTBL), a size limit is exceeded, or memory runs out. */
int ac_csfgen(const ac_cfptab *t, int nsh, const int *l, const int *occ,
              ac_csflist *out);

#endif
