/*
 * ac_lsj.h -- the CI in the LSJ basis: Coulomb H plus the Breit-Pauli
 * spin-orbit and spin-spin operators (stage 2, S2-M4 and S2-M5).
 *
 * The basis of a given J is every CSF of the ac_ci whose total (S, L)
 * couples to J (spin first, |S L J M> = sum <S M_S L M_L|J M> |S L M_S M_L>).
 * Its matrix is
 *   <i J|H|j J> = delta(S L, S' L') <i|H|j>
 *               + (-1/sqrt 3) (-1)^(S'+L+J) {J L S; 1 S' L'} <i||W||j>
 *               + (1/sqrt 5) (-1)^(S'+L+J) {J L S; 2 S' L'} <i||W_SS||j>
 * with <i||W||j> from ac_ci_so1 + ac_ci_so2 and <i||W_SS||j> from ac_ci_ss
 * (A. R. Edmonds, Angular Momentum in Quantum Mechanics (1957), Eq. (7.1.6);
 * technical note, Secs. 3.1, 3.6 and 4.2). All parts depend on J only through
 * the 6j, so they are computed once for all CSF pairs.
 */
#ifndef AC_LSJ_H
#define AC_LSJ_H

#include "ac_ci.h"

#include <stdio.h>

typedef struct {
    int n;     /* number of CSFs                                        */
    double *h; /* [n*n] <i|H|j> (+ H_OO), 0 unless (S, L) agree         */
    double *w; /* [n*n] <i||W||j>, NULL without H_SO1 and H_SO2         */
    double *w2; /* [n*n] <i||W_SS||j> (spin 2, orbital 2), NULL without SS */
} ac_lsj;

int ac_lsj_prepare(ac_lsj *m, ac_ci *ci, char *err, int errlen);
void ac_lsj_free(ac_lsj *m);

/* The basis of 2J = tJ: its CSF indices in idx[] (room for ci->ncsf) and
   the dim x dim matrix in H (column-major, room for ncsf^2). Returns dim.
   *asym receives max |H - H^T| before symmetrization (a check of the
   phases: H_SO1 is Hermitian). */
int ac_lsj_matrix(const ac_lsj *m, const ac_ci *ci, int tJ, int *idx, double *H,
                  double *asym);

/* <S L J || [W^(ks,kl)]^(K) || S' L' J'> from w = <S L || W || S' L'>, with
   [W]^(K)_Q = sum <ks nu kl q|K Q> W_(nu q) (spin first; Edmonds 7.1.5):
     sqrt([J][J'][K]) {S S' ks; L L' kl; J J' K} w.
   All momenta as 2x integers. */
double ac_lsj_reduce(ac_lab a, int tJ, ac_lab b, int tJp, int ks, int kl, int tK, double w);

/* First-order Zeeman factors of a level of 2J = tJ > 0, the eigenvector
   c[dim] over the CSFs idx[dim] of its ac_lsj_matrix basis:
     gL, gS  <L_z>/J and <S_z>/J at M = J (g_J = gL + g_s gS);
     rmc     Delta g of the relativistic mass correction,
             <-2 alpha^2 sum_i T_i s_z,i>/J;
     ormc    its orbital counterpart <-alpha^2 sum_i T_i l_z,i>/J (absent in
             molecular g tensors, where <L> = 0 to first order; for
             hydrogen-like ions rmc + ormc + gc is the alpha^2 term of the
             Dirac g factor);
     gc      Delta g of the one-electron gauge correction (gauge origin at
             the nucleus), <(alpha^2/2) Z sum_i r_i^-1 [s_z,i - (s_i.r^_i) z^_i]>/J.
   For S = 1/2 these are Neese's (JCP 115, 11080 (2001)) RMC and GC terms
   without his factor g_e/2, as in libhexpr2. tkin(ctx, a, b) gives the
   kinetic integral and zr(ctx, a, b) = Z <a|1/r|b> (sequence orbitals);
   either may be NULL (that term is then 0). */
/* gc2e: Delta g of the two-electron gauge correction (ac_ci_gc2e, its parts
   as ac_gc2e_parts selects), when ci->in.rgc is set; 0 otherwise.
   oz2e: Delta g of the orbit-orbit Zeeman term (ac_ci_oz2e), when
   ci->in.roz is set.
   rec2: <O_z>/M_J of the two-body recoil operator (ac_ci_rec2), when
   ci->in.rrc is set: the recoil correction is -(m/M) (gL + rec2)
   h6: the rank-2 part of Pachucki's (PRA 69, 052502 (2004), Eq. (32)) H6 =
   -kappa alpha^2 sum_i (p_i.s_i) p_z,i per unit kappa = (g_s - 2)/2 (its
   Delta g is kappa h6; the scalar part is (kappa/3) rmc):
   <-alpha^2 sum_i s_i^j (p_j p_z - delta_jz p^2/3)_i>/J
   = <alpha^2 sqrt(15)/3 [s x [p x p]^(2)]^(1)_z>/J, when ac_lsj_h6_rk is set */
typedef struct {
    double gL, gS, rmc, ormc, gc, gc2e, oz2e, rec2, h6;
} ac_lsj_g;
/* the radial integrals of h6: ac_radial_dd as kinds 6-9 of ac_lsj_trans_rk
   (ctx as for tkin); NULL: h6 = 0 and not printed */
extern double (*ac_lsj_h6_rk)(void *, int, int, int);

int ac_lsj_gfactor(ac_ci *ci, int dim, const int *idx, int tJ, const double *c,
                   double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                   void *ctx, ac_lsj_g *g, int *err);

/* A level: the eigenvector c[dim] over the CSFs idx[dim] of the
   ac_lsj_matrix basis of 2J = tJ */
typedef struct {
    int tJ, dim;
    const int *idx;
    const double *c;
} ac_lsj_level;

/* Reduced matrix elements <A||mu^(1)||B> (Edmonds) of each first-order
   Zeeman term between two levels, in the fields of ac_lsj_g: gL = <A||L||B>,
   gS = <A||S||B>, and rmc, ormc, gc, gc2e, oz2e, rec2 the reduced elements of
   the operators whose <...>/J are the Delta g of ac_lsj_gfactor. All zero
   unless |J_A - J_B| <= 1 <= J_A + J_B. For A = B, multiplying by
   1/sqrt(J(J+1)(2J+1)) gives ac_lsj_gfactor; for A != B they give the
   Zeeman coupling <A J_A M|mu_z|B J_B M> = (-1)^(J_A-M) (J_A 1 J_B; -M 0 M)
   x <A||mu||B> with mu = gL + g_s gS + (the alpha^2 terms) in mu_B, the
   matrix elements that mix levels in a field (intermediate field,
   Paschen-Back). Eigenvector phases are those of the CI. */
int ac_lsj_zeeman_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                          double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                          void *ctx, ac_lsj_g *r, int *err);

/* Hyperfine structure: reduced matrix elements <A||T^(K)||B> (Edmonds,
   atomic units) of the electronic operators
     T^(1) = sum_i [ l_i r_i^-3                                       (orb)
                     - sqrt(10) (g_s/2) [s_i x C^2(i)]^(1) r_i^-3      (sd)
                     + (g_s/2) (8 pi/3) delta(r_i) s_i ]               (con)
     T^(2) = sum_i C^2(i) r_i^-3                                       (quad)
   (I. Lindgren and A. Rosen, Case Stud. At. Phys. 4, 93 (1974); the same
   operators as ATSP's hfs, Joensson et al., CPC 74, 399 (1993)).
   rm3(ctx, a, b) = <a|r^-3|b> and r00(ctx, a, b) = R_a(0) R_b(0)
   (sequence orbitals). The constants follow from the diagonal elements:
     A_J = A0 (mu_I/I) <J||T^(1)||J> / sqrt(J(J+1)(2J+1)),
     B_J = -2 B0 Q <J J|T^(2)_0|J J>,
          <J J|T^(2)_0|J J> = sqrt(J(2J-1)/((J+1)(2J+1)(2J+3))) <J||T^(2)||J>,
   A0 = (mu0/4pi) 2 mu_B mu_N/(h a0^3) and B0 = e^2/(4 pi eps0 a0^3 h) x 1 b
   (AC_HFS_A0, AC_HFS_B0; mu_I in nuclear magnetons, Q in barns). Zero
   unless the ranks couple J_A to J_B. */
typedef struct {
    double orb, sd, con, quad;
} ac_lsj_hfs;

int ac_lsj_hfs_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                       double (*rm3)(void *, int, int), double (*r00)(void *, int, int),
                       void *ctx, ac_lsj_hfs *h, int *err);

/* Radiative transitions: reduced matrix elements <A||T^(K)||B> (Edmonds,
   atomic units) between two levels of one CI on one orbital set:
     e1l  E1 length,   T^(1) = sum_i r_i C^1(i)
     e1v  E1 velocity, T^(1) = sum_i grad_i
     m1   M1, T^(1) = L + g_s S (in mu_B; no relativistic correction)
     e2   E2, T^(2) = sum_i r_i^2 C^2(i)
     e3   E3, T^(3) = sum_i r_i^3 C^3(i)
     m2   M2, T^(2) = sum_i grad_i(r_i^2 C^2(i)).((2/3) l_i + g_s s_i) (in mu_B)
          = sqrt6 sum_i r_i [C^1(i) x ((2/3) l_i + g_s s_i)]^(2)
     m3   M3, T^(3) = sum_i grad_i(r_i^3 C^3(i)).((1/2) l_i + g_s s_i) (in mu_B)
          = sqrt15 sum_i r_i^2 [C^2(i) x ((1/2) l_i + g_s s_i)]^(3)
          (m1, m2 and m3 take g_s = 2 instead of AC_GS when the environment
          variable AC_TRANS_GS2 is set, for comparisons with ATSP's trans)
     m1r2, m1p2, m1zr   T^(1) = sum_i s_i f_i with f = r^2, p^2, Z/r: the
          spin-scalar one-electron part of the relativistic M1 operator of
          K. Pachucki, Phys. Rev. A 69, 052502 (2004), Eqs. (59), (60)
          (kappa = 0): with mu = -2 j/(ik x) in mu_B,
            delta mu = -4 alpha^2 [w^2 m1r2/12 + m1p2/3 - m1zr/6]
          (ac_lsj_m1_spin_scalar).
     m1lr2, m1lp2   T^(1) = sum_i l_i f_i, f = r^2, p^2: the orbital part,
            -2 T_vec_a -> -alpha^2 [w^2 m1lr2/10 + m1lp2/2]
     m1tr2, m1tzr, m1tpp   T^(1) = sum_i [s_i x X_i]^(1) with X = r^2 C^2,
          (Z/r) C^2, [p x p]^(2): the tensor part (the delta_ij terms of
          Eq. (62) cancel), -2 T^_a.sigma_a -> alpha^2 [-(sqrt10/15) w^2 m1tr2
          + (sqrt10/3) m1tzr - (sqrt15/6) m1tpp]
          (ac_lsj_m1_rel1e: all the one-electron terms).
     m1s0, m1s2, m1oz   the two-electron (r_ab) terms, in mu_B with alpha^2:
          the spin ones (ac_ci_m1s2, K = 0 and 2; only when ci->in.rm1 is
          set) and the orbital one, which is the orbit-orbit Zeeman operator
          (ac_ci_oz2e; when ci->in.rm1 and ci->in.roz are set);
          ac_lsj_m1_2e_parts selects 1 spin, 2 orbital, 3 both (default).
          ac_lsj_m1_rel sums everything. The orbital, tensor and two-electron
          parts vanish between S states (He 2 3S1 - 1 1S0).
   (E1, M2 and E3 need the two levels of opposite parity, M1, E2 and M3 of
   the same; the one-electron elements vanish otherwise.) rk(ctx, kind, a, b)
   gives the radial factors: kind 0 <a|r|b>, 3 <a|r^2|b>, 11 <a|r^3|b>,
   4 <a|p^2|b> (twice the kinetic energy), 5 Z <a|1/r|b>, and the gradient
   factors of
   ac_ci_rec2, kind 1 int R_a (R_b' - l_b R_b/r) r^2 dr (l_a = l_b + 1) and
   kind 2 int R_a (R_b' + (l_b+1) R_b/r) r^2 dr (l_a = l_b - 1). With
   S = |<A||T||B>|^2, the lower level A' of 2J+1 = g' and the upper B' of g,
   and w = E_B' - E_A' (hartree):
     f(E1, length)   = (2/3) w S_len / g',  f(E1, velocity) = (2/3) S_vel/(w g'),
     A(E1) = (4/3) alpha^3 w^3 S / g,  A(M1) = (1/3) alpha^5 w^3 S / g
     (mu_B = alpha/2 in Gaussian atomic units; ATSP's trp: alpha^(2 lambda + 3)
     for M lambda), A(E2) = (1/15) alpha^5 w^5 S / g,
     A(E3) = (8/4725) alpha^7 w^7 S / g, A(M2) = (1/60) alpha^7 w^5 S / g,
     A(M3) = (2/4725) alpha^9 w^7 S / g
     (A(E lambda) = 2(2 lambda+1)(lambda+1)/(lambda ((2 lambda+1)!!)^2)
     alpha^(2 lambda+1) w^(2 lambda+1) S/g; M lambda the same with
     (alpha/2)^2 S and w^(2 lambda+1)),
   in units of 1/tau0, tau0 = hbar/E_h = 2.4188843265857e-17 s. */
typedef struct {
    double e1l, e1v, m1, e2, e3, m2, m3;
    double m1r2, m1p2, m1zr;
    double m1lr2, m1lp2, m1tr2, m1tzr, m1tpp;
    double m1s0, m1s2, m1oz;
} ac_lsj_trans;

int ac_lsj_trans_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                         double (*rk)(void *, int, int, int), void *ctx, ac_lsj_trans *t,
                         int *err);

/* <A||delta mu||B> (mu_B) of the spin-scalar relativistic M1 part above at
   the transition energy w (hartree) */
double ac_lsj_m1_spin_scalar(const ac_lsj_trans *t, double w);
/* delta mu of all the one-electron terms (spin-scalar, orbital, tensor) */
double ac_lsj_m1_rel1e(const ac_lsj_trans *t, double w);
/* delta mu of everything computed: ac_lsj_m1_rel1e + m1s0 + m1s2 + m1oz */
double ac_lsj_m1_rel(const ac_lsj_trans *t, double w);
extern int ac_lsj_m1_2e_parts;

/* When ac_lsj_trans_rk is set (ctx as for tkin), ac_lsj_print_levels also
   prints the E1, M1, E2, M2, E3 and M3 lines between every pair of printed levels, and
   "M1r": M1 with the one-electron relativistic part (S, <||delta mu||>, A).
   Its kinds: 0 <r>, 1 and 2 the gradient factors, 3 <r^2>, 4 <p^2>, 5 Z <1/r>,
   6-9 ac_radial_dd with (sa, sb) = (-,-), (-,+), (+,-), (+,+). */
extern double (*ac_lsj_trans_rk)(void *, int, int, int);
/* nonzero: the line strengths with 17 digits (tests) */
extern int ac_lsj_digits;
/* J blocks of at least this dimension, with more than 4 nlev rows, are
   solved by Davidson for the nlev lowest levels only (ac_ci_davidson,
   residual 1e-11); smaller ones by dsyev. 0: always dsyev. Default 500. */
extern int ac_lsj_davidson;

/* Isotope shift: expectation values (J-scalars) in one level of
     nms   K_NMS = <sum_i p_i^2/2>          (hartree)
     sms   K_SMS = <sum_{i<j} p_i.p_j>      (hartree; ac_ci_sms)
     rho0  rho(0) = <sum_i delta(r_i)>      (a0^-3)
   The mass shift of a level is (m/M)(nms + sms) to first order in m/M
   (from P^2/2M, P = -sum p_i), the field shift (2 pi/3) Z rho0 delta<r^2>
   (point nucleus density, a.u.; for a hydrogen-like ns state the
   non-relativistic limit of V. M. Shabaev, Phys. Rep. 356, 119 (2002),
   Eqs. (322), (324)). rk as ac_lsj_trans_rk, with kind 10
   R_a(0) R_b(0). */
typedef struct {
    double nms, sms, rho0;
} ac_lsj_is;

int ac_lsj_is_level(ac_ci *ci, const ac_lsj_level *A, double (*rk)(void *, int, int, int),
                    void *ctx, ac_lsj_is *r, int *err);
/* When set (ctx as for tkin), ac_lsj_print_levels also prints the isotope
   shift table of every printed level */
extern double (*ac_lsj_is_rk)(void *, int, int, int);

/* MHz, CODATA 2018 */
#define AC_HFS_A0 95.41053
#define AC_HFS_B0 234.9647

/* g of the electron spin, CODATA 2018 (|g_e|), and 1 hartree in cm-1 */
#define AC_GS 2.00231930436256
#define AC_HARTREE_CM 219474.6313632

/* Print the lowest nlev levels of every 2J (energy, leading LS term and
   weight, g_J, Delta g_RMC, Delta g_GC), then all printed levels relative to
   the lowest in cm-1. tkin, zr, ctx as for ac_lsj_gfactor. When ac_lsj_zoff
   is nonzero, it then prints ac_lsj_zeeman_reduced for every pair of
   printed levels with |J_A - J_B| <= 1 (A listed before B). */
extern int ac_lsj_zoff;
/* When ac_lsj_hfs_rm3 is set (with ac_lsj_hfs_r00; ctx as for tkin), it also
   prints ac_lsj_hfs_reduced for every pair of printed levels, A_J and B_J
   for the diagonal ones. */
extern double (*ac_lsj_hfs_rm3)(void *, int, int);
extern double (*ac_lsj_hfs_r00)(void *, int, int);
int ac_lsj_print_levels(FILE *out, ac_ci *ci, int nlev, double (*tkin)(void *, int, int),
                        double (*zr)(void *, int, int), void *ctx, char *err, int errlen);

/* A printed level: its eigenvector, number within its J (from 1) and
   energy (hartree). */
typedef struct {
    ac_lsj_level v;
    int num;
    double e;
} ac_lsj_kept;

void ac_lsj_kept_free(ac_lsj_kept *k, int n);

/* ac_lsj_print_levels that also hands the printed levels to the caller
   (*keep, *nkeep; free with ac_lsj_kept_free) when keep is not NULL */
int ac_lsj_print_levels_keep(FILE *out, ac_ci *ci, int nlev, double (*tkin)(void *, int, int),
                             double (*zr)(void *, int, int), void *ctx, ac_lsj_kept **keep,
                             int *nkeep, char *err, int errlen);

/* The tables that ac_lsj_zoff, ac_lsj_hfs_rm3 and ac_lsj_trans_rk select,
   between every level of ka (the bra) and every level of kb, both over the
   CSFs of ci (S2-M9: two states on biorthonormal orbitals, ka on A and kb
   on B, the integrals of ctx taking the left orbital from A and the right
   from B). The transitions add a column: the list (1 = ka, 2 = kb) of the
   lower level. */
int ac_lsj_print_cross(FILE *out, ac_ci *ci, const ac_lsj_kept *ka, int na,
                       const ac_lsj_kept *kb, int nb, double (*tkin)(void *, int, int),
                       double (*zr)(void *, int, int), void *ctx, char *err, int errlen);

#endif
