/*
 * ac_cfp.h -- the CFP table (CFPTBL) and shell states in the LSQ scheme.
 *
 * Stage 1 reads the CFPTBL that the Fortran tools cfpa/cfpb produce
 * (technical note, Sec. 3.3); ac_cfp_generate makes the same
 * table in C (milestone M1b).
 *
 * A shell state is identified, as in ATOMCI, by the code 256*isq + lsub:
 * isq is the 1-based row of ICONNT, i.e. one (l, Q, S) block, and lsub the
 * 1-based index into that block's list of L values (in descending L).
 * The quantum numbers of a code come back in 2j units (ac_shell_qn).
 *
 * The table arrays keep Fortran's contents unchanged, including its 1-based
 * address values (ILSTAD, ICONNT(5,.), ICMTDX(1,.)); only the C array
 * indices are 0-based. So iconnt[isq-1][4] holds ICONNT(5,isq).
 */
#ifndef AC_CFP_H
#define AC_CFP_H

typedef struct {
    int tl; /* 2l of the shell      */
    int tq; /* 2Q, quasi-spin       */
    int ts; /* 2S, spin             */
    int tL; /* 2L, orbital momentum */
} ac_shell_qn;

typedef struct {
    /* record 1 */
    int name[18];
    int ibel, icnn, ilfn, icmtn, icdx;
    int immint[18]; /* 2Q_min+1 per shell                    */
    int ilstad[18]; /* first ICONNT row (1-based) per shell  */
    /* records 2-5 */
    int (*iconnt)[9]; /* [icnn]: LFO, 2Q+1, 2S+1, dim, ILFUNC start, links x4 */
    int (*icmtdx)[5]; /* [icmtn]: CMTTBL start, rows, cols, row block, col block */
    int *ilfunc;      /* [ilfn]: 2L+1 */
    double *cmttbl;   /* [icdx]: <F~_lowQ B F_highQ>, column-major per matrix */

    /* shell-state lists per (l, N), as built by the Fortran CFPSET */
    int maxn;       /* = 4*ibel - 2, the largest electron count stored */
    int *lst_start; /* [ibel*(maxn+1)] offset into lst_code, -1 = not built */
    int *lst_count; /* [ibel*(maxn+1)]                                     */
    int *lst_code;
    int lst_len;
} ac_cfptab;

/* Load CFPTBL and build the shell-state lists. Returns 0 on success; on
   failure writes a message to err (if errlen > 0) and returns -1. The path
   "-" generates the table instead (ac_cfp_generate), so every tool that
   takes a CFPTBL argument also runs without the Fortran file. */
int ac_cfp_load(ac_cfptab *t, const char *path, char *err, int errlen);
void ac_cfp_free(ac_cfptab *t);

/* Generate the table in C (ac_cfpgen.c, milestone M1b): a line-by-line port
   of the Fortran cfpa + cfpb (including the F19.14 rounding of the
   intermediate CFPDATA and cfpb's SYMTLZ), so that the result agrees with
   the CFPTBL they write. Builds the shell-state lists as ac_cfp_load. */
int ac_cfp_generate(ac_cfptab *t, char *err, int errlen);
/* Write records 1-5 as a Fortran unformatted sequential file (CFPTBL). */
int ac_cfp_write(const ac_cfptab *t, const char *path, char *err, int errlen);
/* The shell-state lists from records 1-5 (internal to ac_cfp_load and
   ac_cfp_generate). Returns 0, or -1 when out of memory. */
int ac_cfp_build_lists(ac_cfptab *t);

/* Decode a shell-state code; returns 0, or -1 for an invalid code. */
int ac_cfp_decode(const ac_cfptab *t, int code, ac_shell_qn *qn);

/* ICONNT row (1-based isq) of the (l, Q, S) block: the Fortran LSQN
   closed form. Assumes every parity-allowed block of the shell exists. */
int ac_cfp_lsqn(const ac_cfptab *t, int tl, int tq, int ts);

/* Shell states of shell l with nel electrons, in CFPSET order.
   Returns the number of states (0 if none) and sets *codes; returns -1 if
   the list was not built (shell truncated above its last seniority). */
int ac_cfp_states(const ac_cfptab *t, int l, int nel, const int **codes);

/* The Fortran XCFP(IM,KM,KMP,ID), a reduced matrix element of the shell
   creation operator b between LS shell states:
     id = +1: <f~_{N}(im) b f_{N-1}(km)>, kmp = N-1 electrons in km;
     id = -1: the adjoint orientation used by CFACRC, kmp = electrons in km.
   Returns 0 when the two states are not connected.
   Deliberate differences from the Fortran:
     - the shell row of im is im/256 (the Fortran uses im/255, which agrees
       while (im/256) + (im mod 256) < 255, true for every table cfpa makes);
     - a link to a level that was not generated (ICONNT link = -1) gives 0
       instead of indexing ICMTDX(.,-1). */
double ac_cfp_xcfp(const ac_cfptab *t, int im, int km, int kmp, int id);

#endif
