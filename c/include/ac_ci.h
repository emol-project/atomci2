/*
 * ac_ci.h -- the non-relativistic CI of stage 2 (S2-M2).
 *
 * CSFs list their occupied orbitals ("slots", in coupling-sequence order)
 * with shell states and intermediate couplings; closed shells are explicit
 * slots (technical note, Sec. 4.1). A matrix element is a sum
 * over the operators of METECC eqs. (44) and (45) that can connect the pair,
 * each evaluated by the general-form engine (ac_factor_op) on the union of
 * the two slot lists, empty shells inserted where one side has none.
 */
#ifndef AC_CI_H
#define AC_CI_H

#include "ac_addr.h"
#include "ac_cfp.h"
#include "ac_csf.h"
#include "ac_ints.h"
#include "ac_tape.h"
#include "ac_terms.h"

typedef ac_orb_csf ac_ci_csf;

/* Radial integrals by sequence orbital (1-based): h(a,b), R^k(ab;cd) with
   a, b on electron 1, and zeta(a,b) of H_SO1 (NULL: no spin-orbit). The
   constructors from a Fortran tape set h and rk to read NFTRI. h and rk
   share ctx: replacing one of them with a wrapper means wrapping the other
   too (tools/acfs.c, "mvd"). */
typedef struct {
    double (*h)(void *ctx, int a, int b);
    double (*rk)(void *ctx, int k, int a, int b, int c, int d);
    double (*zeta)(void *ctx, int a, int b);
    /* the radial integrals of H_SO2 (ac_radial_rkg): R^k with the gradient
       radial functions G_(a, l_a + sa) G_(b, l_b + sb) on electron 1 and
       R_c R_d on electron 2, sa, sb = +-1; context zctx; NULL: no H_SO2 */
    double (*rkg)(void *ctx, int k, int a, int sa, int b, int sb, int c, int d);
    /* the radial integrals of H_SS (ac_radial_ssw): int R_a R_b r^2 w_lam dr
       with w_lam of the pair density cd and multipole k; context zctx;
       NULL: no spin-spin */
    double (*rss)(void *ctx, int k, int lam, int a, int b, int c, int d);
    /* the radial integrals of H_OO (ac_radial_oo): g >= 0 the current
       term of rank (k, g), g = -1 the divergence term of multipole k;
       context zctx; NULL: no orbit-orbit */
    double (*roo)(void *ctx, int k, int g, int a, int b, int c, int d);
    /* the radial integrals of the two-electron gauge correction of g
       (ac_radial_gc): kp = 0, 1, 2 the [R x grad]^(kp) term of electron-1
       rank g, kp = -1 the density term (plain R^k); context zctx; NULL: none */
    double (*rgc)(void *ctx, int k, int kp, int g, int a, int b, int c, int d);
    /* the radial integrals of the orbit-orbit Zeeman term (ac_radial_oz);
       context zctx; NULL: none */
    double (*roz)(void *ctx, int k, int g, int type, int a, int b, int c, int d);
    /* the one-electron radial factors of the two-body recoil term: kind 0
       int R_a R_b r^3 dr, kind 1 int R_a (R_b' - l_b R_b/r) r^2 dr, kind 2
       int R_a (R_b' + (l_b+1) R_b/r) r^2 dr; context zctx; NULL: none */
    double (*rrc)(void *ctx, int kind, int a, int b);
    /* the radial integrals of the spin two-electron part of the relativistic
       M1 operator (ac_radial_m1w): int R_a R_b r^2 u_lam dr with u_lam of the
       r12 potential of the pair density cd and multipole k; lam = -1: the
       plain R^k(ab;cd) (without any relativistic contact term); context
       zctx; NULL: none (ac_ci_m1s2 then 0) */
    double (*rm1)(void *ctx, int k, int lam, int a, int b, int c, int d);
    void *ctx, *zctx;
} ac_ci_integrals;

typedef struct {
    const ac_cfptab *t;
    const ac_ints *ints;
    ac_ci_integrals in;
    ac_addr addr;
    int nwt;
    int lorb[AC_ADDR_MAXORB + 1]; /* [1..nwt] l of each sequence orbital */
    ac_ci_csf *csf;
    int ncsf;
    long ops_evaluated;           /* statistics */
    void *thc; /* Theta of the Breit-Pauli terms by orbitals, k, g (internal) */
} ac_ci;

/* The CSFs of a Fortran energy-expression tape (its CSF image and default
   closed shells), with the integral addressing of its header. */
int ac_ci_from_nfte(ac_ci *ci, const ac_cfptab *t, const ac_ints *ints,
                    const ac_nfte_head *h, char *err, int errlen);
/* All CSFs, of every (S, L), of a list of configurations, with the
   orbitals, default closed shells and integral addressing of a Fortran
   tape header h. Configuration c lists nocc[c] orbitals: seq[c][i] (1-based
   sequence numbers) holding occ[c][i] electrons; default closed shells not
   listed are added with 4l+2 electrons. Slots are coupled in sequence
   order. */
int ac_ci_from_configs(ac_ci *ci, const ac_cfptab *t, const ac_ints *ints,
                       const ac_nfte_head *h, int ncfg, const int *nocc,
                       const int (*seq)[AC_CSF_MAXSH], const int (*occ)[AC_CSF_MAXSH],
                       char *err, int errlen);
/* All CSFs of configurations over a bare orbital list (no tape, no NFTRI):
   orbital s = 1..nwt has momentum lorb[s-1]; configuration c gives the
   occupation occ[c*nwt + s-1] of every orbital (0 allowed). Every orbital
   is a slot. The caller fills ci->in. */
int ac_ci_from_orbitals(ac_ci *ci, const ac_cfptab *t, int nwt, const int *lorb,
                        int ncfg, const int *occ, char *err, int errlen);
/* The same CSFs with only the occupied orbitals of each configuration as
   slots (in sequence order; empty shells couple trivially), so that nwt may
   reach AC_ADDR_MAXORB and the engine works on a few slots per CSF. */
int ac_ci_from_orbitals_occ(ac_ci *ci, const ac_cfptab *t, int nwt, const int *lorb,
                            int ncfg, const int *occ, char *err, int errlen);
void ac_ci_free(ac_ci *ci);
/* Free only the Theta cache (ci->thc). A shallow copy of an ac_ci with
   thc = NULL may evaluate elements in another thread, with its own cache;
   free it with this, not ac_ci_free. */
void ac_ci_free_cache(ac_ci *ci);
/* the bytes of the Theta cache of ci, and the number of its entries in *n
   (may be NULL) */
size_t ac_ci_cache_bytes(const ac_ci *ci, size_t *n);

/* <i|H|j> between CSFs i and j (same S and L). */
double ac_ci_hij(ac_ci *ci, int i, int j, int *err);

/* The reduced element <i||W||j> (Edmonds) of a one-electron double tensor
     W = sum_ab f(a,b) (b_a b~_b)^(ks/2, kl/2)        (ks, kl as 2x ranks)
   over orbital pairs of equal l. f(ctx, a, b, l) gets sequence orbitals;
   pairs with f = 0 are skipped. Edmonds = (-1)^(ks/2+kl/2) x the engine's
   invariant (technical note, Sec. 3.1).
   A per-electron operator sum_i g(r_i) t(i) u(i), t of spin rank ks/2 and
   u of orbital rank kl/2, equals W with
     f(a,b) = <a|g|b> <1/2||t||1/2> <l||u||l> / (-sqrt([ks/2][kl/2])),
   see ac_ci_one_coef. */
double ac_ci_one(ac_ci *ci, int i, int j, int ks, int kl,
                 double (*f)(void *ctx, int a, int b, int l), void *ctx, int *err);
/* The same for pairs of any l (l_a, l_b within the triangle of kl/2); f gets
   both l, and the orbital factor of f is <l_a||u||l_b>. */
double ac_ci_one_ll(ac_ci *ci, int i, int j, int ks, int kl,
                    double (*f)(void *ctx, int a, int b, int la, int lb), void *ctx, int *err);
/* ac_ci_one_ll for nf (at most AC_CI_ONE_N) operators of the same ranks at
   once: out[x] is the number ac_ci_one_ll gives with ctx[x], and the
   coupling of each orbital pair is evaluated once for all of them.
   Returns 0, or -1 with *err set. */
#define AC_CI_ONE_N 8
int ac_ci_one_ll_n(ac_ci *ci, int i, int j, int ks, int kl,
                   double (*f)(void *ctx, int a, int b, int la, int lb), int nf,
                   void *const *ctx, double *out, int *err);
double ac_ci_one_coef(double spin_reduced, double orbital_reduced, int ks, int kl);

/* The reduced element <i||W||j> of the double tensor
     W = sum_ab zeta(a,b) c_l (b_a b~_b)^(1,1),  c_l = sqrt(l(l+1)(2l+1)/2),
   whose J-scalar is H_SO1 (technical note, Sec. 4.2),
   with zeta from ci->in. */
double ac_ci_so1(ac_ci *ci, int i, int j, int *err);

/* The reduced element <i||W2||j> of the double tensor whose J-scalar is the
   two-electron spin-orbit operator (spin-same-orbit + spin-other-orbit)
     H_SO2 = (alpha^2/2) sum_{p != q} O_pq.(s_p + 2 s_q),
     O_12 = grad_1(1/r12) x p_1 = -r12^-3 (r12 x p_1).
   Integrating by parts, <ac|O|bd> = i int int (1/r12) (grad phi_a* x
   grad phi_b)(1) phi_c* phi_d(2); with grad(R Y_l) = sum_(lam = l+-1)
   G_lam(r) [Y_lam x e]^(l) (gradient formula) and i(A* x B) = -A+ S B (S the
   spin-1 operator on the vector index), O = -sum_k f_k (-1)^k sum_g
   sqrt([g]/3) [[C^k x S]^(g)(1) x C^k(2)]^(1). Each ordered (a, b, c, d),
   k and g = k-1, k, k+1 enters as
     Theta = (alpha^2/2) (-1)^k [k]^(-1/2) (1 for s_p, 2 for s_q)
             <l_c||C^k||l_d> sum_(lam_a, lam_b) sqrt(6 [l_a][l_b][g])
             {lam_a lam_b k; 1 1 1; l_a l_b g} <lam_a||C^k||lam_b>
             R^k(G_a G_b; R_c R_d)
     times the J-scalar of :((b_a b~_b)^(s1 g) (b_c b~_d)^(s2 k))^(1 1):,
   (s1, s2) = (1, 0) or (0, 1) (technical note, Sec. 4.2). Checked
   against PySCF's int2e_p1vxp1 through python/so2ref.py. 0 when ci->in.rkg
   is NULL. */
double ac_ci_so2(ac_ci *ci, int i, int j, int *err);

/* which parts of H_SO2 ac_ci_so2 includes: 1 spin-same-orbit (s_p),
   2 spin-other-orbit (2 s_q), 3 both (default; for comparisons) */
extern int ac_so2_parts;

/* nonzero: ac_ci_so2 and ac_ci_ss return 0 between CSFs that differ by a
   double excitation ab -> cd over four distinct orbitals, as ATSP's bpci
   does on the Si e1.bp-5.c example, whose CSFs carry the closed core
   explicitly (its hzeta/hspin lists have no such elements;
   technical note, Sec. 7.6). It is not a general model of bpci: on
   lists with the core in the header (O, Li) bpci keeps these elements and
   agrees with C without this flag to 1e-11 in g_J, and bpci's own levels
   change with the way closed shells are written (0.5 of the same note). For
   that one comparison only: the elements are not zero (checked against
   PySCF, tests/v5). */
extern int ac_atsp_compat;

/* The reduced element <i||W_SS||j> of the double tensor (spin 2, orbital 2)
   whose J-scalar is the spin-spin operator
     H_SS = alpha^2 sum_{p < q} [s_p.s_q - 3 (s_p.r^)(s_q.r^)] / r_pq^3
   (dipolar part; the contact term is a spin scalar and left out). Its
   spatial part is minus the traceless second derivative of the potential
   V_cd of the pair density cd at electron 1 (libhexpr2's d_ab), and
     s_1.T.s_2 = -[s1 x s2]^(2) . [grad x grad]^(2) V,
   which gives for each ordered (a, b, c, d), multipole k and lam = k-2,
   k, k+2
     Theta = -(alpha^2/4) (-1)^k / (2k+1) <l_a||C^lam||l_b> <l_c||C^k||l_d>
             int R_a R_b r^2 w_lam dr      (w_lam: ac_radial_ssw)
   times the J-scalar of :((b_a b~_b)^(1 lam) (b_c b~_d)^(1 k))^(2 2):.
   0 when ci->in.rss is NULL. */
double ac_ci_ss(ac_ci *ci, int i, int j, int *err);

/* ac_ci_so2 and ac_ci_ss for all the pairs (i0 + i, j0 + j), i < ni, j < nj,
   where the CSFs i0 .. i0+ni-1 have the same orbitals and occupations, and
   so have j0 .. j0+nj-1 (two configurations):
     so2[i * nj + j] = ac_ci_so2(ci, i0 + i, j0 + j),  ss likewise,
   bit for bit, but with the operators set up once and each evaluated for
   all the pairs together. so2 or ss may be NULL (not computed). upper = 1
   (for i0 == j0): only j >= i is wanted, the other elements are left 0.
   tsec, when not NULL, has the wall times of the two parts added to
   tsec[0] and tsec[1]. Returns 0, or -1 with the error in *err (as
   ac_ci_so2; 1 also when out of memory). */
int ac_ci_bp2_block(ac_ci *ci, int i0, int ni, int j0, int nj, int upper, double *so2, double *ss,
                    double *tsec, int *err);

/* The same for H_OO: out[ni * nj] = ac_ci_oo of every pair (0 for j < i when
   upper is set, and for the pairs whose total (S, L) differ). */
int ac_ci_oo_block(ac_ci *ci, int i0, int ni, int j0, int nj, int upper, double *out, int *err);

/* The same for the Coulomb Hamiltonian:
     out[i * nj + j] = ac_ci_hij(ci, i0 + i, j0 + j),
   bit for bit (0 between different total (S, L), and for j < i with
   upper = 1). Returns 0, or -1 with the error in *err (as ac_ci_hij). */
int ac_ci_hij_block(ac_ci *ci, int i0, int ni, int j0, int nj, int upper, double *out, int *err);

/* The same for lists of CSFs: fi[0..nf-1] of one configuration and
   gj[0..ng-1] of one configuration,
     out[p * ng + q] = ac_ci_hij(ci, fi[p], gj[q]). */
int ac_ci_hij_list(ac_ci *ci, int nf, const int *fi, int ng, const int *gj, double *out, int *err);

/* The same for two-electron property operators: kp = 0, 1, 2 gives
     out[i * nj + j] = ac_ci_gc2e(ci, i0 + i, j0 + j, kp),
   kp = -1 gives ac_ci_oz2e(ci, i0 + i, j0 + j), bit for bit, for all the
   pairs of the two configurations. Returns 0; -1 with the error in *err;
   -2 for another kp (nothing computed). */
int ac_ci_prop2_block(ac_ci *ci, int kp, int i0, int ni, int j0, int nj, double *out, int *err);

/* <i|H_OO|j> (same S and L) of the orbit-orbit operator
     H_OO = -(alpha^2/2) sum_{p<q} [p_p.p_q + r_pq.(r_pq.p_p) p_q / r_pq^2] / r_pq.
   From ac_radial_oo's form, with C^k(1).C^k(2) (J(1).J(2)) recoupled to
   [C^k x J]^(g)(1).[C^k x J]^(g)(2), each ordered (a, b, c, d) enters as
     current:    Theta = alpha^2 (-1)^k sqrt[k] (-sqrt3) {k k 0; 1 1 0; g g 0}
                         roo(k, g),  g = k-1, k, k+1,
     divergence: Theta = (alpha^2/2) (-1)^k <l_a||C^k||l_b><l_c||C^k||l_d>
                         roo(k, -1) / sqrt[k]
   times :((b_a b~_b)^(0 g)(b_c b~_d)^(0 g))^(0 0): (g = k for the second),
   divided by sqrt([S][L]) as the Coulomb part of ac_ci_hij (whose H^k is
   (-1)^k <C^k><C^k>/sqrt[k] in this rule). 0 when ci->in.roo is NULL. */
double ac_ci_oo(ac_ci *ci, int i, int j, int *err);

/* The reduced element <i||W^(1 K')||j> (Edmonds, spin rank 1, orbital rank
   K' = kp = 0, 1, 2) of the two-electron gauge correction of g
   (technical note, Sec. 5.1): the magnetic-moment operator (in units
   of g, O_z = 2 dH/dB_z)
     O = (alpha^2/2) sum_{i != j} [grad_i(1/r_ij) x (b x r_i)].(s_i + 2 s_j)
       = sum_K' c'_K' [W^(1 K')]^(1),
   c'_0 = 2/sqrt3 on [R x grad]^(0) and -2 on the density, c'_1 = 1,
   c'_2 = -sqrt(5/3) (spin first; the Cartesian split checked numerically).
   Each ordered (a, b, c, d), multipole k and electron-1 rank g enters as
     Theta = (alpha^2/2) (1 for s_i, 2 for s_j) c'_K' (-1)^k <l_c||C^k||l_d>
             rgc(k, K', g) / sqrt([K'][k])
   (density term: g = k, rgc(k, -1) <l_a||C^k||l_b>) times
   :((b_a b~_b)^(s1 g)(b_c b~_d)^(s2 k))^(1 K'):. ac_gc2e_parts selects 1
   the spin-same-orbit (Coulomb) part, 2 the spin-other-orbit (Breit) part,
   3 both. 0 when ci->in.rgc is NULL. */
double ac_ci_gc2e(ac_ci *ci, int i, int j, int kp, int *err);
extern int ac_gc2e_parts;

/* The reduced element <i||W^(0 1)||j> (Edmonds) of the orbit-orbit Zeeman
   term of g (technical note, Sec. 5.1):
     O_b = -(alpha^2/2) sum_{i != j} [r_i x (K_ij p_j)].b
         = (i alpha^2/2) int int X.K.J,  X = f (b x r1), J the current of j,
   and with K = 2 delta/r - grad grad r
     O^(1) = sqrt2 alpha^2 int int f [R(1) x J(2)]^(1) / r12
             + (alpha^2/sqrt2) int int ([R x grad]^(1) f)(1) Q(2) r12.
   Each ordered (a, b, c, d) and multipole k enters as
     term 1: Theta = 2 sqrt6 alpha^2 (-1)^k sqrt[k] {k k 0; 1 1 1; g1 g2 1}
             <l_a||[C^k x C^1]^(g1)||l_b> roz(k, g2, 0),
             tree :((b_a b~_b)^(0 g1)(b_c b~_d)^(0 g2))^(0 1):;
     term 2: Theta = alpha^2 sqrt(2/3) (-1)^k <l_c||C^k||l_d> roz(k, g, 1) / sqrt[k],
             tree :((b_a b~_b)^(0 g)(b_c b~_d)^(0 k))^(0 1):.
   0 when ci->in.roz is NULL. */
double ac_ci_oz2e(ac_ci *ci, int i, int j, int *err);

/* The reduced element <i||W^(0 1)||j> (Edmonds) of the two-body part of the
   nuclear-recoil correction to the orbital moment (M. Phillips, Phys. Rev.
   76, 1803 (1949), T_H1; R. A. Hegstrom, Phys. Rev. A 7, 451 (1973),
   Eq. (23)):
     O = sum_{i != j} r_i x p_j = -sqrt2 sum_{i != j} [r_i x grad_j]^(1),
   so that g_L = 1 - (m/M) (<L_z> + <O_z>)/<L_z> on the orbital part. With
   <l'||grad||l> = <l'||C^1||l> (d/dr - l/r) for l' = l+1 and
   <l'||C^1||l> (d/dr + (l+1)/r) for l' = l-1, each ordered (a, b, c, d) enters as
     Theta = -sqrt2 (2/3) <l_a||C^1||l_b> rrc(0, a, b) <l_c||grad||l_d>
   times :((b_a b~_b)^(0 1)(b_c b~_d)^(0 1))^(0 1): (each one-electron unit
   operator with the factor sqrt2 <l||u||l'> / (-sqrt3) of ac_ci_one).
   0 when ci->in.rrc is NULL. */
double ac_ci_rec2(ac_ci *ci, int i, int j, int *err);

/* The reduced element <i||W^(0 0)||j> (Edmonds; sqrt([S][L]) times the
   element) of the specific-mass-shift operator
     O = sum_{i<j} p_i.p_j = (sqrt3/2) sum_{i != j} [grad_i x grad_j]^(0),
   whose expectation value times m/M is the specific mass shift (hartree).
   Each ordered (a, b, c, d) enters as
     Theta = <l_a||grad||l_b> <l_c||grad||l_d> / sqrt3
   times :((b_a b~_b)^(0 1)(b_c b~_d)^(0 1))^(0 0): (the gradient units as in
   ac_ci_rec2). grad(ctx, kind, a, b) gives the radial factors, kind 1
   int R_a (R_b' - l_b R_b/r) r^2 dr (l_a = l_b + 1) and kind 2
   int R_a (R_b' + (l_b+1) R_b/r) r^2 dr (l_a = l_b - 1), as ci->in.rrc. */
double ac_ci_sms(ac_ci *ci, int i, int j, double (*grad)(void *, int, int, int), void *ctx,
                 int *err);

/* The reduced element <i||W^(1 K)||j> (Edmonds, spin rank 1, orbital rank
   K = 0 or 2) of the spin two-electron part of the relativistic M1 operator
   (Pachucki, PRA 69, 052502 (2004), Eqs. (60), (62): the 1/(6 r_ab) of T_a
   and the r_ab terms of T^_a), in mu_B with mu = -2 sum (sigma T + ...):
     delta mu = -alpha^2 sum_{a != b} (grad_a grad_a r_ab).s_a
              = W^(1 0) + [W^(1 2)]^(1),
     W^(1 0) = -(2/3) alpha^2 sum_{a != b} s_a / r_ab,
     [W^(1 2)]^(1) = alpha^2 sqrt(5/3) sum_{a != b} [s_a x [grad_a x grad_a]^(2) r_ab]^(1).
   Each ordered (a, b, c, d) (spin on a, b; electron 2 on c, d) enters as
     K = 0:  Theta = -(2/3) alpha^2 (-1)^k <l_a||C^k||l_b><l_c||C^k||l_d>
                     R^k(ab;cd) / sqrt[k] (rm1 with lam = -1),
             tree :((b_a b~_b)^(1 k)(b_c b~_d)^(0 k))^(1 0):;
     K = 2:  Theta = alpha^2 (-1)^k <l_a||C^lam||l_b><l_c||C^k||l_d>
                     rm1(k, lam) / (sqrt3 [k]),  lam = k-2, k, k+2,
             tree :((b_a b~_b)^(1 lam)(b_c b~_d)^(0 k))^(1 2):
   (the one-body units with <s||s>, <l||C||l'>/(-sqrt([s][kappa])) as
   ac_ci_one). 0 when ci->in.rm1 is NULL. */
double ac_ci_m1s2(ac_ci *ci, int i, int j, int K, int *err);

/* All eigenvalues w and eigenvectors (columns of H, overwritten) of the
   symmetric n x n matrix H (column-major) via dsyev. Returns 0 or info. */
int ac_ci_eigvecs(int n, double *H, double *w);

/* All eigenvalues (ascending) of the symmetric n x n matrix H (column-major,
   lower triangle used; destroyed) via LAPACK dsyev. Returns 0 or info. */
int ac_ci_eigvals(int n, double *H, double *w);

/* The number of eigenvalues below sigma of the symmetric n x n matrix H
   (column-major, lower triangle used; destroyed), in *count: H - sigma is
   factorized as L D L^T (LAPACK dsytrf, D of 1 x 1 and 2 x 2 blocks), and
   D has as many negative eigenvalues as H - sigma (Sylvester's law of
   inertia). About a quarter of the work of ac_ci_eigvals. Returns 0, or
   nonzero (memory, LAPACK). */
int ac_ci_count_below(int n, double *H, double sigma, int *count);

/* The k lowest eigenvalues w[0..k-1] (ascending) and eigenvectors (the k
   columns of x, n x k) of the symmetric n x n matrix H (column-major, both
   triangles; not changed), by Davidson's method (E. R. Davidson, J. Comput.
   Phys. 17, 87 (1975)) in Liu's block form (B. Liu, "The simultaneous
   expansion method ...", in Numerical Algorithms in Chemistry: Algebraic
   Methods, C. Moler and I. Shavitt (eds.), LBL-8158 (1978), p. 49) with
   the diagonal preconditioner. Starts from the lowest eigenvectors of the
   block of the (at least 200) lowest diagonal elements; the subspace holds
   up to 8k + 48 vectors and restarts from its 3k lowest Ritz vectors;
   converged when
   every residual |H x_j - w_j x_j| < tol. Returns 0, 1 (not converged; w and
   x hold the last Ritz pairs), or nonzero otherwise (memory, LAPACK). The
   number of iterations in *niter (may be NULL). */
int ac_ci_davidson(int n, const double *H, int k, double tol, double *w, double *x, int *niter);

/* The same for a symmetric matrix that is not stored: mul(ctx, m, V, AV)
   gives AV = H V for m columns of length n (column-major), elem(ctx, a, b)
   the element (a, b); elem is asked for the diagonal and for the block of
   the start vectors only. The same method step by step. */
int ac_ci_davidson_op(int n, void (*mul)(void *ctx, int m, const double *V, double *AV),
                      double (*elem)(void *ctx, int a, int b), void *ctx, int k, double tol,
                      double *w, double *x, int *niter);

/* The same two for a matrix that falls into np parts which H does not
   couple at all (part[i] in 0..np-1 for each row; the two parities of a
   list that has both, the total (S, L) of a J block without spin-dependent
   terms). Davidson stays in what its start vectors span and what H couples
   to it, so on the whole matrix it can miss a level of a part its start
   left out. Here each part is solved alone (its min(k, size) lowest levels;
   by dsyev when it has at most 400 rows) and the k lowest of all are
   returned. np <= 1: as ac_ci_davidson. With parts, mul is asked for
   vectors that are 0 outside one part and must give 0 there; sel(ctx, part,
   cur), when not NULL, tells the operator that part before (part NULL:
   all rows again), so that it can skip the other rows. *niter: the sum over
   the parts. */
int ac_ci_davidson_parts(int n, const double *H, int np, const int *part, int k, double tol,
                         double *w, double *x, int *niter);
int ac_ci_davidson_op_parts(int n, void (*mul)(void *ctx, int m, const double *V, double *AV),
                            double (*elem)(void *ctx, int a, int b),
                            void (*sel)(void *ctx, const int *part, int cur), void *ctx, int np,
                            const int *part, int k, double tol, double *w, double *x, int *niter);

#endif
