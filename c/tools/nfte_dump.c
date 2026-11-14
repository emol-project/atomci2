/*
 * nfte_dump.c -- print an energy-expression tape (NFTE) in readable form.
 *
 * usage: nfte_dump [-v] tape
 *
 * Prints the header, the CSF types of the image, and a summary of the data
 * records; with -v every data record as well.
 */
#include "ac_ftnio.h"
#include "ac_tape.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    int verbose = argc == 3 && !strcmp(argv[1], "-v");
    if (argc != 2 && !verbose) {
        fprintf(stderr, "usage: %s [-v] tape\n", argv[0]);
        return 2;
    }
    const char *path = argv[argc - 1];
    ac_ftn_file f;
    if (ac_ftn_open(&f, path, "rb")) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    ac_nfte_head h;
    char err[256];
    if (ac_nfte_read_head(&f, &h, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", path, err);
        return 1;
    }
    char name[121];
    memcpy(name, h.name, 120);
    name[120] = '\0';
    printf("NAME   '%s'\n", name);
    printf("ISPIN %d IORB %d IPAR %d NWT %d NWF %d ITSTV %d ITNEL %d LIJB %d IDIAG %d\n",
           h.ispin, h.iorb, h.ipar, h.nwt, h.nwf, h.itstv, h.itnel, h.lijb, h.idiag);
    printf("LMAX %d INT %d LWFL %d NTYPE %d NREF %d MXRPS %d\n", h.lmax, h.nint,
           h.lwfl, h.ntype, h.nref, h.mxrps);
    printf("ISSH");
    for (int i = 0; i < h.lmax; i++) printf(" %d", h.issh[i]);
    printf("\nNWTN");
    for (int i = 0; i < h.nwt; i++) printf(" %d:%d", h.nwtn[i] / 256, h.nwtn[i] % 256);
    printf("\nNWTC");
    for (int i = 0; i < h.nwf; i++) printf(" %d", h.nwtc[i]);
    printf("\nNWSQ");
    for (int i = 0; i < h.nwt; i++) printf(" %d", h.nwsq[i]);
    printf("\n");

    const int16_t *W = h.nwfsg - 1;
    int js = 1, ntype = 0, ncsf = 0;
    while (js < h.nint && W[js] > 0) {
        int ns = W[js], nrps = W[js + 1];
        if (verbose)
            printf("type %d: %d shells, %d CSFs\n", ntype + 1, ns, nrps);
        ntype++;
        ncsf += nrps;
        js += ns * (nrps + 3) + 2;
    }
    printf("CSF image: %d types, %d CSFs, image words used %d of %d\n", ntype, ncsf,
           js - 1, h.nint);

    ac_nfte_rec r;
    memset(&r, 0, sizeof r);
    long nrec = 0, nmark = 0, npairs = 0, nterms = 0;
    int rc;
    while ((rc = ac_nfte_read_rec(&f, &r)) == AC_FTN_OK) {
        if (r.noe == 0) {
            nmark++;
            if (verbose)
                printf("marker %d\n", r.itw);
            continue;
        }
        nrec++;
        npairs += r.nij;
        nterms += r.ltri;
        if (verbose) {
            printf("record: ITW %d JTV %d NOE %d N1OE %d N2OE %d NIJ %d NII %d LTRI %d\n",
                   r.itw, r.jtv, r.noe, r.n1oe, r.n2oe, r.nij, r.nii, r.ltri);
            for (int p = 0; p < r.nij; p++) {
                printf("  pair (%d,%d):", r.indexi[p], r.nii ? r.indexj[p] : r.indexi[p]);
                for (int e = 0; e < r.noe; e++)
                    printf(" %d*%.10g", r.isqn[p * r.noe + e], r.wt[e]);
                printf("\n");
            }
        }
    }
    if (rc != AC_FTN_EOF)
        fprintf(stderr, "%s: read error %d after %ld records\n", path, rc, nrec);
    printf("data: %ld records, %ld markers, %ld pairs, %ld (pair, term) entries\n", nrec,
           nmark, npairs, nterms);
    ac_nfte_rec_free(&r);
    ac_nfte_head_free(&h);
    ac_ftn_close(&f);
    return rc == AC_FTN_EOF ? 0 : 1;
}
