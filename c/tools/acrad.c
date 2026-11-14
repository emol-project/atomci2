/*
 * acrad -- S2-M3 check of ac_radial: radial integrals of the MOs by
 * numerical quadrature against the Fortran's transformed integrals.
 *
 *   acrad nfaii nftrc nftri [h]
 *
 * nfaii  AO integral file (unit 18): the basis
 * nftrc  MO coefficient file (unit 21)
 * nftri  transformed integral file (unit 16): h and R^k
 * h      grid step in ln r (default 0.005)
 *
 * Compared: the MO overlap with the unit matrix (numeric and analytic);
 * every h(a,b) and every R^k(ab;cd) of NFTRI at its ac_addr address;
 * <a|r^-3|b> numeric against the analytic basis-pair formula; and the
 * hydrogenic 2p value <r^-3> = Z^3/24.
 */
#include "ac_addr.h"
#include "ac_basis.h"
#include "ac_ints.h"
#include "ac_radial.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double analytic_moment(const ac_basis *b, const ac_radial *rad, int a, int c, int p)
{
    int l = rad->l[a], ia = rad->idx[a], ic = rad->idx[c], m = b->count[l];
    double s = 0.0;
    for (int mu = 0; mu < m; mu++)
        for (int nu = 0; nu < m; nu++)
            s += b->c[l][mu + (size_t)ia * m] * b->c[l][nu + (size_t)ic * m] *
                 ac_bfn_moment(&b->bf[b->first[l] + mu], &b->bf[b->first[l] + nu], p);
    return s;
}

static double analytic_h(const ac_basis *b, const ac_radial *rad, int a, int c)
{
    int l = rad->l[a], ia = rad->idx[a], ic = rad->idx[c], m = b->count[l];
    double s = 0.0;
    for (int mu = 0; mu < m; mu++)
        for (int nu = 0; nu < m; nu++)
            s += b->c[l][mu + (size_t)ia * m] * b->c[l][nu + (size_t)ic * m] *
                 ac_bfn_h(&b->bf[b->first[l] + mu], &b->bf[b->first[l] + nu], b->z);
    return s;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: acrad nfaii nftrc nftri [h [rmax [dump]]]\n");
        return 2;
    }
    double h = argc > 4 ? atof(argv[4]) : AC_RGRID_H, rmx = argc > 5 ? atof(argv[5]) : AC_RGRID_RMAX;
    /* dump: every R^k as "k a b c d C F", for an independent check */
    FILE *dump = argc > 6 ? fopen(argv[6], "w") : NULL;
    char err[256];
    ac_basis b;
    ac_ints in;
    if (ac_basis_load(&b, argv[1], argv[2], err, sizeof err) ||
        ac_ints_load(&in, argv[3], err, sizeof err)) {
        fprintf(stderr, "acrad: %s\n", err);
        return 1;
    }
    printf("basis: %s, Z = %g, %d functions, LMAX %d\n", b.gto ? "GTO" : "STO", b.z, b.nbf, b.nl);
    int ok = in.nsym == b.nl;
    for (int l = 0; l < b.nl; l++) {
        printf("  l=%d: %d functions, %d MOs (NFTRI %d)\n", l, b.count[l], b.nob[l],
               l < in.nsym ? in.nob[l] : -1);
        if (l < in.nsym && in.nob[l] != b.nob[l])
            ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "acrad: MO counts of NFTRI and the vector file differ\n");
        return 1;
    }

    ac_rgrid g;
    ac_radial rad;
    if (ac_rgrid_init(&g, AC_RGRID_RMIN, rmx, h) || ac_radial_build(&rad, &b, &g, err, sizeof err)) {
        fprintf(stderr, "acrad: grid: %s\n", err);
        return 1;
    }
    printf("grid: %d points, h = %g, r in [%g, %g]\n", g.n, h, AC_RGRID_RMIN, rmx);

    /* overlap */
    double dsn = 0, dsa = 0;
    for (int a = 0; a < rad.nmo; a++)
        for (int c = 0; c < rad.nmo; c++) {
            if (rad.l[a] != rad.l[c])
                continue;
            double e = a == c;
            double sn = ac_radial_moment(&rad, a, c, 0), sa = analytic_moment(&b, &rad, a, c, 0);
            dsn = fmax(dsn, fabs(sn - e));
            dsa = fmax(dsa, fabs(sa - e));
        }
    printf("overlap: max |S - 1| numeric %.2e, analytic %.2e\n", dsn, dsa);

    /* addresses: MOs in l order, as SETKAD with the one-electron block first */
    int32_t nwlf[AC_ADDR_MAXORB];
    for (int a = 0; a < rad.nmo; a++)
        nwlf[a] = rad.l[a] + 1;
    ac_addr ad;
    if (ac_addr_setup(&ad, NULL, 1, rad.nmo, NULL, nwlf, NULL)) {
        fprintf(stderr, "acrad: address setup failed\n");
        return 1;
    }

    double dh = 0, dha = 0, dfa = 0, hmax = 0;
    long nh = 0;
    for (int a = 0; a < rad.nmo; a++)
        for (int c = 0; c <= a; c++) {
            if (rad.l[a] != rad.l[c])
                continue;
            int p = ac_addr_one(&ad, a + 1, c + 1);
            double v = ac_radial_h(&rad, a, c), va = analytic_h(&b, &rad, a, c);
            dh = fmax(dh, fabs(v - in.x[p]));
            dha = fmax(dha, fabs(v - va));
            dfa = fmax(dfa, fabs(in.x[p] - va));
            hmax = fmax(hmax, fabs(in.x[p]));
            nh++;
        }
    printf("h: %ld values, max |h| %.3g, max |C - F| %.2e, |C - analytic| %.2e,"
           " |F - analytic| %.2e\n", nh, hmax, dh, dha, dfa);

    /* R^k: every triangle- and parity-allowed (k; ab; cd) */
    double *y = malloc((size_t)g.n * sizeof(double));
    double drk = 0, rmax = 0;
    long nrk = 0, nbad = 0, nover = 0;
    int worst[5] = {0};
    double wc = 0, wf = 0;
    for (int a = 0; a < rad.nmo; a++)
        for (int c = 0; c <= a; c++) {
            int la = rad.l[a], lc = rad.l[c];
            for (int k = abs(la - lc); k <= la + lc; k += 2) {
                ac_radial_yk(&rad, k, a, c, y);
                for (int bb = 0; bb < rad.nmo; bb++)
                    for (int d = 0; d <= bb; d++) {
                        int lb = rad.l[bb], ld = rad.l[d];
                        if (k < abs(lb - ld) || k > lb + ld || (lb + ld + k) % 2)
                            continue;
                        long p = in.n1 + ac_addr_two(&ad, 2 * k + 1, a + 1, c + 1, bb + 1, d + 1);
                        if (p <= in.n1 || p > in.n) {
                            nbad++;
                            continue;
                        }
                        double v = ac_radial_dot(&rad, bb, d, y), e = fabs(v - in.x[p]);
                        nover += e > 1e-10;
                        if (dump)
                            fprintf(dump, "%d %d %d %d %d %.17g %.17g\n", k, a, c, bb, d, v, in.x[p]);
                        if (e > drk) {
                            drk = e;
                            wc = v, wf = in.x[p];
                            worst[0] = k, worst[1] = a, worst[2] = c, worst[3] = bb, worst[4] = d;
                        }
                        rmax = fmax(rmax, fabs(in.x[p]));
                        nrk++;
                    }
            }
        }
    printf("R^k: %ld values (%ld out of range), max |R| %.3g, max |C - F| %.2e"
           " at k=%d (%d %d; %d %d): C %.15g, F %.15g; %ld above 1e-10\n",
           nrk, nbad, rmax, drk, worst[0], worst[1], worst[2], worst[3], worst[4], wc, wf, nover);
    free(y);
    if (dump)
        fclose(dump);

    /* r^-3 and zeta */
    double dz = 0, zmax = 0;
    long nz = 0;
    for (int a = 0; a < rad.nmo; a++)
        for (int c = 0; c <= a; c++) {
            if (rad.l[a] != rad.l[c] || rad.l[a] == 0)
                continue;
            double vn = ac_radial_moment(&rad, a, c, -3), va = analytic_moment(&b, &rad, a, c, -3);
            dz = fmax(dz, fabs(vn - va) / fmax(1.0, fabs(va)));
            zmax = fmax(zmax, fabs(va));
            nz++;
        }
    printf("<r^-3>: %ld values, max %.4g, max |numeric - analytic| (relative above 1) %.2e\n",
           nz, zmax, dz);
    /* the first two MOs of each l > 0; 1 hartree = 219474.6313632 cm-1 (CODATA 2018) */
    for (int a = 0; a < rad.nmo; a++)
        if (rad.l[a] > 0 && rad.idx[a] < 2)
            printf("  zeta(%c MO %d) = %.12e hartree = %.6f cm-1\n", "spdfghi"[rad.l[a]],
                   rad.idx[a] + 1, ac_radial_zeta(&rad, a, a),
                   ac_radial_zeta(&rad, a, a) * 219474.6313632);

    /* hydrogenic 2p of the same Z on the same grid */
    ac_bfn hp = {.l = 1, .n = 2, .gto = 0, .nprim = 1, .alpha = {b.z / 2}, .coef = {1}};
    double nrm = ac_bfn_moment(&hp, &hp, 0);
    hp.coef[0] = 1 / sqrt(nrm);
    double sum = 0;
    for (int i = 0; i < g.n; i++) {
        double v = ac_bfn_value(&hp, g.r[i]);
        sum += g.w[i] * v * v / g.r[i];
    }
    double exact = b.z * b.z * b.z / 24;
    printf("hydrogenic 2p <r^-3>: numeric %.15g, analytic formula %.15g, Z^3/24 %.15g\n", sum,
           ac_bfn_moment(&hp, &hp, -3), exact);

    /* mass velocity <nabla^2 a|nabla^2 b>: numeric against the basis-pair
       formula (relative to the largest), and the Darwin values R(0) */
    double dmv = 0, mvmax = 0;
    for (int a = 0; a < rad.nmo; a++)
        for (int c = 0; c <= a; c++) {
            if (rad.l[a] != rad.l[c])
                continue;
            int l = rad.l[a], m = b.count[l];
            double va = 0;
            for (int mu = 0; mu < m; mu++)
                for (int nu = 0; nu < m; nu++)
                    va += b.c[l][mu + (size_t)rad.idx[a] * m] * b.c[l][nu + (size_t)rad.idx[c] * m] *
                          ac_bfn_lap2(&b.bf[b.first[l] + mu], &b.bf[b.first[l] + nu]);
            va *= -AC_ALPHA * AC_ALPHA / 8;
            dmv = fmax(dmv, fabs(ac_radial_mv(&rad, a, c) - va));
            mvmax = fmax(mvmax, fabs(va));
        }
    printf("mass velocity: max |MV| %.4g hartree, max |numeric - analytic| %.2e (relative %.1e)\n",
           mvmax, dmv, dmv / mvmax);
    printf("  Darwin(s MO 1) = %.12e, MV(s MO 1) = %.12e hartree\n",
           ac_radial_darwin(&rad, rad.first[0], rad.first[0]),
           ac_radial_mv(&rad, rad.first[0], rad.first[0]));

    /* spin two-electron M1 radial integrals int R_a R_b r^2 u dr, u from
       ac_radial_m1w(k, lam, c, d), over the first two MOs of l <= 2: printed
       as "m1w k lam a b c d value" for a comparison across grids */
    if (getenv("ACRAD_M1W")) {
        double *u = malloc((size_t)g.n * sizeof(double));
        int sel[AC_ADDR_MAXORB], ns = 0;
        for (int a = 0; a < rad.nmo; a++)
            if (rad.l[a] <= 2 && rad.idx[a] < 2)
                sel[ns++] = a;
        for (int ic = 0; ic < ns; ic++)
            for (int id = 0; id < ns; id++) {
                int c = sel[ic], d = sel[id], lc = rad.l[c], ld = rad.l[d];
                for (int k = abs(lc - ld); k <= lc + ld; k += 2)
                    for (int lam = k - 2; lam <= k + 2; lam += 2) {
                        if (lam < 0)
                            continue;
                        ac_radial_m1w(&rad, k, lam, c, d, u);
                        for (int ia = 0; ia < ns; ia++)
                            for (int ib = 0; ib < ns; ib++) {
                                int a = sel[ia], bb = sel[ib], la = rad.l[a], lb = rad.l[bb];
                                if (lam < abs(la - lb) || lam > la + lb || (la + lb + lam) % 2)
                                    continue;
                                const double *va = rad.val + (size_t)a * g.n;
                                const double *vb = rad.val + (size_t)bb * g.n;
                                double s = 0.0;
                                for (int i = 0; i < g.n; i++)
                                    s += g.w[i] * va[i] * vb[i] * g.r[i] * g.r[i] * u[i];
                                printf("m1w %d %d %d %d %d %d %.15e\n", k, lam, a, bb, c, d, s);
                            }
                    }
            }
        free(u);
    }

    /* Tolerances. Against the Fortran, R^k only to 1e-8: its AO two-electron
       integrals are the less accurate side there (technical note,
       Sec. 6). */
    int pass = dsn < 1e-10 && dsa < 1e-10 && dha < 1e-10 && dh < 1e-10 && nbad == 0 &&
               drk < 1e-8 && dz < 1e-12 && fabs(sum - exact) < 1e-13 * exact &&
               dmv < 1e-11 * mvmax;
    printf("acrad: %s\n", pass ? "PASS" : "FAIL");

    ac_radial_free(&rad);
    ac_rgrid_free(&g);
    ac_ints_free(&in);
    ac_basis_free(&b);
    return pass ? 0 : 3;
}
