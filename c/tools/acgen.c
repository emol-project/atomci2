/*
 * acgen.c -- regenerate the data records of an energy-expression tape (NFTE)
 * with the C angular engine.
 *
 * usage: acgen CFPTBL in.nfte out.nfte
 *
 * The header records and the CSF image are copied byte for byte from
 * in.nfte (a tape written by the Fortran EXTAPE or WFETAP); the data
 * records that follow are produced by ac_tape_generate (EXIJ, SMAT, SHMS,
 * STORE in C). out.nfte can replace the Fortran tape for the CI stage.
 */
#include "ac_cfp.h"
#include "ac_ftnio.h"
#include "ac_tape.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s CFPTBL in.nfte out.nfte\n", argv[0]);
        return 2;
    }
    char err[256];
    ac_cfptab t;
    if (ac_cfp_load(&t, argv[1], err, sizeof err)) {
        fprintf(stderr, "acgen: %s\n", err);
        return 1;
    }
    ac_ftn_file in, out;
    if (ac_ftn_open(&in, argv[2], "rb")) {
        fprintf(stderr, "acgen: cannot open %s\n", argv[2]);
        return 1;
    }
    ac_nfte_head h;
    if (ac_nfte_read_head(&in, &h, err, sizeof err)) {
        fprintf(stderr, "acgen: %s: %s\n", argv[2], err);
        return 1;
    }
    ac_ftn_close(&in);
    if (ac_ftn_open(&out, argv[3], "wb") || ac_nfte_write_head(&out, &h)) {
        fprintf(stderr, "acgen: cannot write %s\n", argv[3]);
        return 1;
    }
    ac_tape_stats st;
    int rc = ac_tape_generate(&t, &h, &out, &st, err, sizeof err);
    ac_ftn_close(&out);
    if (rc) {
        fprintf(stderr, "acgen: %s\n", err);
        return 1;
    }
    printf("acgen: %ld SMAT calls, %ld pairs stored, %ld records + %ld markers, "
           "%ld terms (%ld one-electron)\n",
           st.smat_calls, st.stored_pairs, st.records, st.markers, st.noet, st.n1oet);
    ac_nfte_head_free(&h);
    ac_cfp_free(&t);
    return 0;
}
