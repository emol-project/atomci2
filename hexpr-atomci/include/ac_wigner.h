/*
 * ac_wigner.h -- 6j and 9j symbols and the Coulomb angular factors.
 *
 * Angular momenta are passed as twice their value (tj = 2j), so half-integer
 * spins are exact integers. The orbital-only factors ac_fabc() and ac_hk()
 * take the integer orbital momenta themselves.
 *
 * Every function checks its own triangle and parity conditions and returns
 * 0 when they fail; callers need not pre-screen arguments. This differs from
 * the Fortran F3L, which leaves the parity of a+b+c to its callers.
 *
 * Factorials are tabulated in double precision up to AC_WIGNER_FACMAX!; an
 * argument set that would need a larger factorial returns NaN.
 */
#ifndef AC_WIGNER_H
#define AC_WIGNER_H

#define AC_WIGNER_FACMAX 170

/* sqrt(n!) for 0 <= n <= AC_WIGNER_FACMAX, NaN otherwise. */
double ac_sqrt_factorial(int n);

/* sqrt((a+b-c)! (a-b+c)! (-a+b+c)!); 0 unless (a,b,c) is a triangle with
   integer perimeter. Matches the Fortran DELTA (no (a+b+c+1)! denominator). */
double ac_tri_delta(int ta, int tb, int tc);

/* Wigner 6j symbol {j1 j2 j3; j4 j5 j6}. */
double ac_sixj(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6);

/* Wigner 9j symbol {j1 j2 j3; j4 j5 j6; j7 j8 j9}, as a sum of 6j symbols
   (METECC eq. A40): sum_x (-1)^{2x} [x] {j1 j2 j3; j6 j9 x}
                     {j4 j5 j6; j2 x j8} {j7 j8 j9; x j1 j4}. */
double ac_ninej(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6,
                int tj7, int tj8, int tj9);

/* F(a,b,c) of METECC eq. (45) (Condon-Shortley normalization):
   0 if a+b+c is odd or (a,b,c) is not a triangle, otherwise with 2g=a+b+c
   g! sqrt((2g-2a)!(2g-2b)!(2g-2c)!) / (sqrt((2g+1)!) (g-a)!(g-b)!(g-c)!). */
double ac_fabc(int a, int b, int c);

/* H^k(l1,l2,l3,l4) of METECC eq. (45), as coded in the Fortran FSUBK:
   (-1)^{(l1-l2+l3-l4)/2} sqrt([l1][l2][l3][l4]/[k]) F(k,l1,l2) F(k,l3,l4),
   with [x] = 2x+1. */
double ac_hk(int k, int l1, int l2, int l3, int l4);

#endif
