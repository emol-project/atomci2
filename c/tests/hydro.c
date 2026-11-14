/*
 * hydro.c -- S2-M6a: the one-electron Breit-Pauli terms of ac_radial
 * against the Dirac energy of hydrogen-like ions.
 *
 * The exact 1s, 2s and 2p orbitals are built from STOs (1s = phi_1(Z),
 * 2s = phi_1(Z/2) - sqrt(3) phi_2(Z/2), 2p = phi_2(Z/2), phi_n normalized
 * r^(n-1) e^(-a r)). Mass velocity, Darwin and spin-orbit (zeta, with
 * E_SO = (zeta/2)[j(j+1) - l(l+1) - 3/4]) are computed on the ac_radial grid
 * and must sum to the alpha^2 term of the Dirac energy,
 *   E2(n, j) = -(alpha^2 Z^4 / (2 n^4)) (n/(j+1/2) - 3/4).
 * The single terms are checked against their hydrogenic closed forms
 * <p^4> = (Z^4/n^4)(8n/(2l+1) - 3) and |psi(0)|^2 = Z^3/(pi n^3), and MV
 * also against the analytic basis-pair integral ac_bfn_lap2.
 *
 * usage: hydro   (exit 0 when every relative error is below 1e-10)
 */
#include "ac_basis.h"
#include "ac_radial.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ac_bfn sto(int l, int n, double a)
{
    ac_bfn f;
    memset(&f, 0, sizeof f);
    f.l = l;
    f.n = n;
    f.nprim = 1;
    f.alpha[0] = a;
    f.coef[0] = 1.0;
    f.coef[0] = 1.0 / sqrt(ac_bfn_moment(&f, &f, 0));
    return f;
}

static double worst = 0.0;

static void cmp(const char *what, double z, double got, double want)
{
    double e = fabs(got - want) / fabs(want);
    if (e > worst)
        worst = e;
    printf("  Z=%-3g %-26s %22.15e %22.15e %8.1e\n", z, what, got, want, e);
}

int main(void)
{
    const double zs[] = {1, 10, 50};
    const double a2 = AC_ALPHA * AC_ALPHA;
    printf("  %-31s %22s %22s %8s\n", "quantity", "ac_radial", "closed form", "rel");
    for (int iz = 0; iz < 3; iz++) {
        double Z = zs[iz];
        ac_bfn bf[4] = {sto(0, 1, Z), sto(0, 1, Z / 2), sto(0, 2, Z / 2), sto(1, 2, Z / 2)};
        double c0[6] = {1, 0, 0, 0, 1, -sqrt(3.0)}, c1[1] = {1};
        ac_basis b;
        memset(&b, 0, sizeof b);
        b.z = Z;
        b.nbf = 4;
        b.bf = bf;
        b.nl = 2;
        b.first[0] = 0, b.count[0] = 3, b.nob[0] = 2, b.c[0] = c0;
        b.first[1] = 3, b.count[1] = 1, b.nob[1] = 1, b.c[1] = c1;
        /* 2s: normalize the combination */
        double s = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                s += c0[3 + i] * c0[3 + j] * ac_bfn_moment(&bf[i], &bf[j], 0);
        for (int i = 3; i < 6; i++)
            c0[i] /= sqrt(s);

        ac_rgrid g;
        ac_radial rad;
        char err[128];
        if (ac_rgrid_init(&g, AC_RGRID_RMIN, 200.0 / Z, AC_RGRID_H) || ac_radial_build(&rad, &b, &g, err, sizeof err)) {
            fprintf(stderr, "hydro: %s\n", err);
            return 1;
        }
        int s1 = 0, s2 = 1, p2 = 2;
        double Z4 = Z * Z * Z * Z;
        double mv1 = ac_radial_mv(&rad, s1, s1), mv2 = ac_radial_mv(&rad, s2, s2),
               mvp = ac_radial_mv(&rad, p2, p2);
        double d1 = ac_radial_darwin(&rad, s1, s1), d2 = ac_radial_darwin(&rad, s2, s2);
        double zt = ac_radial_zeta(&rad, p2, p2);
        cmp("MV 1s", Z, mv1, -a2 / 8 * Z4 * 5);
        cmp("MV 2s", Z, mv2, -a2 / 8 * Z4 / 16 * 13);
        cmp("MV 2p", Z, mvp, -a2 / 8 * Z4 / 16 * (16.0 / 3 - 3));
        cmp("MV 1s, basis-pair formula", Z, mv1, -a2 / 8 * ac_bfn_lap2(&bf[0], &bf[0]));
        cmp("MV 2p, basis-pair formula", Z, mvp, -a2 / 8 * ac_bfn_lap2(&bf[3], &bf[3]));
        /* Darwin = (pi alpha^2 Z / 2) |psi(0)|^2, |psi(0)|^2 = Z^3/(pi n^3) */
        cmp("Darwin 1s", Z, d1, a2 * Z4 / 2);
        cmp("Darwin 2s", Z, d2, a2 * Z4 / 16);
        cmp("zeta 2p = alpha^2 Z^4/48", Z, zt, a2 * Z4 / 48);
        cmp("1s1/2: MV + D = E2(1,1/2)", Z, mv1 + d1, -a2 * Z4 / 2 * (1 - 0.75));
        cmp("2s1/2: MV + D = E2(2,1/2)", Z, mv2 + d2, -a2 * Z4 / 32 * (2 - 0.75));
        cmp("2p1/2: MV - zeta = E2(2,1/2)", Z, mvp - zt, -a2 * Z4 / 32 * (2 - 0.75));
        cmp("2p3/2: MV + zeta/2 = E2(2,3/2)", Z, mvp + zt / 2, -a2 * Z4 / 32 * (1 - 0.75));
        ac_radial_free(&rad);
        ac_rgrid_free(&g);
    }
    printf("hydrogen-like Breit-Pauli terms %s (max relative error %.1e)\n",
           worst < 1e-10 ? "passed" : "FAILED", worst);
    return worst < 1e-10 ? 0 : 1;
}
