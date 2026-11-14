/*
 * ac_radial.h -- radial integrals of the MOs by numerical quadrature
 * (stage 2, S2-M3).
 *
 * The grid is uniform in x = ln r, r_i = exp(x0 + i h), so that the
 * trapezoid rule in x (weights h r_i) integrates the smooth, rapidly
 * decaying radial integrands to near machine precision. The Slater
 * integrals go through the potentials
 *   Y^k_ab(r) = r^-(k+1) int_0^r rho_ab s^k ds + r^k int_r^inf rho_ab s^-(k+1) ds,
 *   rho_ab(s) = R_a(s) R_b(s) s^2,
 * whose two cumulative integrals use a 6-point (sixth-order) rule per
 * interval. Then
 *   R^k(ab; cd) = int R_c R_d Y^k_ab r^2 dr,
 * i.e. a, b belong to electron 1 and c, d to electron 2, as in INT2AD.
 *
 * MOs are numbered 0..nmo-1, grouped by l in the order of ac_basis.
 */
#ifndef AC_RADIAL_H
#define AC_RADIAL_H

#include "ac_basis.h"

/* fine-structure constant, CODATA 2018 */
#define AC_ALPHA (1.0 / 137.035999084)

/* the value every relativistic term of ac_radial and ac_lsj uses; AC_ALPHA
   unless a caller changes it (ATSP, for example, uses 1/137.036) */
extern double ac_alpha;

/* default grid: r in [AC_RGRID_RMIN, AC_RGRID_RMAX], step AC_RGRID_H in ln r.
   The integrand of <nabla^2 s|nabla^2 s> is finite at r = 0, so the part
   below rmin is ~ rmin x 16 Z^5; rmin = 1e-16 keeps it below 1e-14 relative
   up to Z ~ 50. */
#define AC_RGRID_RMIN 1e-16
#define AC_RGRID_RMAX 100.0
#define AC_RGRID_H 0.005

typedef struct {
    int n;
    double x0, h;
    double *r, *w; /* [n] points and trapezoid weights h r_i */
} ac_rgrid;

int ac_rgrid_init(ac_rgrid *g, double rmin, double rmax, double h);
void ac_rgrid_free(ac_rgrid *g);

typedef struct {
    const ac_rgrid *g;
    double z;
    int nmo;
    int *l, *idx;     /* [nmo] l and index within l              */
    int first[AC_BASIS_MAXL];
    double *val, *der; /* [nmo][n] R(r_i) and dR/dr(r_i)          */
    double *lap;       /* [nmo][n] R'' + 2R'/r - l(l+1)R/r^2      */
    double *r0;        /* [nmo] R(0) (nonzero only for s MOs)     */
} ac_radial;

int ac_radial_build(ac_radial *rad, const ac_basis *b, const ac_rgrid *g,
                    char *err, int errlen);
void ac_radial_free(ac_radial *rad);

/* int R_a R_b r^(2+p) dr */
double ac_radial_moment(const ac_radial *rad, int a, int b, int p);
/* <a| -1/2 nabla^2 - Z/r |b> for MOs of the same l */
double ac_radial_h(const ac_radial *rad, int a, int b);
/* Y^k_ab on the grid, y[n] */
void ac_radial_yk(const ac_radial *rad, int k, int a, int b, double *y);
/* int R_c R_d y r^2 dr, with y from ac_radial_yk */
double ac_radial_dot(const ac_radial *rad, int c, int d, const double *y);
/* R^k(ab; cd) */
double ac_radial_rk(const ac_radial *rad, int k, int a, int b, int c, int d);
/* zeta(ab) = (alpha^2 / 2) Z <a| r^-3 |b>, the radial factor of H_SO1 */
double ac_radial_zeta(const ac_radial *rad, int a, int b);

/* One-electron scalar Breit-Pauli terms (same l):
   mass velocity  <a| -(alpha^2/8) p^4 |b> = -(alpha^2/8) <nabla^2 a|nabla^2 b>,
   Darwin         <a| (pi alpha^2/2) Z delta(r) |b> = (alpha^2 Z/8) R_a(0) R_b(0)
                  (s only; |psi(0)|^2 = R(0)^2/(4 pi)). */
double ac_radial_mv(const ac_radial *rad, int a, int b);

/* Two-electron Breit-Pauli radial integrals (Glass and Hibbert 1978;
   Gaigalas et al., Phys. Scr. 57, 207 (1998), eqs. (29), (30)), electron 1
   on i, i' and electron 2 on j, j':
   N^k(ij, i'j') = (alpha^2/4) int int P_i(1) P_j(2) r2^k/r1^(k+3) e(r1 - r2)
                   P_i'(1) P_j'(2) dr1 dr2,   e = step function;
   V^k(ij, i'j') = (alpha^2/4) int int P_i(1) P_j(2) r<^k/r>^(k+3) r2
                   (d/dr1 - 1/r1) P_i'(1) P_j'(2) dr1 dr2,   P = r R.
   This V^k is the one for which the identity of their eq. (32),
     V^(k-1)(ij,i'j') + V^(k-1)(i'j',ij) = k N^k(ij,i'j') - (k+1) N^(k-2)(ji,j'i'),
   holds (integration by parts; checked numerically in acatsp q=). */
double ac_radial_nk(const ac_radial *rad, int k, int i, int j, int ip, int jp);

/* The radial integral of H_SO2 (ac_ci_so2):
     int dr1 r1^2 G_(a,sa)(r1) G_(b,sb)(r1) Y^k_cd(r1),
   with the radial functions of the gradient formula
     grad(R Y_lm) = sum_(lam = l+-1) G_lam(r) [Y_lam x e]^(l)_m,
     G_(l+1) = -sqrt((l+1)/(2l+1)) (R' - l R/r),
     G_(l-1) =  sqrt(l/(2l+1)) (R' + (l+1) R/r),
   sa, sb = +1 for lam = l+1, -1 for lam = l-1; y = Y^k_cd from ac_radial_yk. */
double ac_radial_gdot(const ac_radial *rad, int a, int sa, int b, int sb, const double *y);

/* The radial functions of the rank-2 second derivative of the potential of
   the pair density cd (spin-spin, ac_ci_ss): with
     Y^k_cd(r) = r^-(k+1) A(r) + r^k B(r),  rho = R_c R_d r^2,
   and the radial gradient operators
     D+_k f = <k+1||grad||k> f = -sqrt(k+1) (f' - k f/r),
     D-_k f = <k-1||grad||k> f =  sqrt(k) (f' + (k+1) f/r),
   (correction 2026-10-27: with this code's Wigner-Eckart convention the
   reduced elements are minus these, e.g. <1||grad||0> R = +R'. The overall
   sign of D cancels here and in ac_ci_oo, where grad enters twice; the
   gauge correction, with one gradient, uses the correct sign, checked
   against python/gc2eref.py),
   <lam||[grad x grad]^(2)||k> Y^k = (-1)^(lam+k) sqrt5 sum_(j = k+-1)
   {1 1 2; k lam j} <lam||grad||j><j||grad||k> Y^k for lam = k+2, k, k-2:
     lam = k+2:  -sqrt((k+1)(k+2)) (2k+1) [rho/r^2 - (2k+3) A/r^(k+3)] x c
     lam = k:    (2k+1) [(k+1) {1 1 2; k k k+1} + k {1 1 2; k k k-1}] rho/r^2 x c'
     lam = k-2:  sqrt(k(k-1)) (2k+1) [(2k-1) r^(k-2) B - rho/r^2] x c''
   (c's: the 6j and phase). w[n] receives that function of lam on the grid. */
void ac_radial_ssw(const ac_radial *rad, int k, int lam, int c, int d, double *w);

/* As ac_radial_ssw, for the potential of the kernel r12 instead of 1/r12
   (the spin two-electron part of the relativistic M1 operator, ac_ci_m1s2):
     Z^k_cd(r) = int rho(s) z_k(r, s) ds,  rho = R_c R_d s^2,
     z_k = r<^(k+2)/((2k+3) r>^(k+1)) - r<^k/((2k-1) r>^(k-1))
   (r12 = sum_k z_k C^k(1).C^k(2)); Z' from the same cumulative integrals,
   Z'' from grad^2 (Z^k C^k) = 2 Y^k C^k. u[n] receives
   <lam||[grad x grad]^(2)||k> Z^k (lam = k-2, k, k+2) on the grid, with the
   D+- of ac_radial_ssw. */
void ac_radial_m1w(const ac_radial *rad, int k, int lam, int c, int d, double *u);
double ac_radial_vk(const ac_radial *rad, int k, int i, int j, int ip, int jp);
double ac_radial_darwin(const ac_radial *rad, int a, int b);
/* The one-electron radial factors of the two-body recoil term (ac_ci_rec2):
   kind 0 int R_a R_b r^3 dr, kind 1 int R_a (R_b' - l_b R_b/r) r^2 dr,
   kind 2 int R_a (R_b' + (l_b+1) R_b/r) r^2 dr. */
double ac_radial_rrc(const ac_radial *rad, int kind, int a, int b);
/* int (D_sa R_a)(D_sb R_b) r^2 dr with the radial operators of the gradient
   formula, D_+ = d/dr - l/r (l -> l+1) and D_- = d/dr + (l+1)/r (l -> l-1);
   the radial part of <a|p_i p_j|b> = int (d_i psi_a)(d_j psi_b) through an
   intermediate l'' = l_a + sa = l_b + sb (the relativistic M1 tensor term). */
double ac_radial_dd(const ac_radial *rad, int a, int sa, int b, int sb);

/* The two-electron contact terms. The two-electron Darwin term
   -pi alpha^2 sum_{p<q} delta(r_pq) and the spin-spin contact term
   -(8 pi/3) alpha^2 sum_{p<q} s_p.s_q delta(r_pq) act only on pairs that
   meet, i.e. on spatially symmetric pair states, which are spin singlets
   (s_p.s_q = -3/4): together +pi alpha^2 delta(r_pq), and on antisymmetric
   states this spin-free operator gives the same matrix elements (its
   triplet part vanishes). With
     delta(r1 - r2) = delta(r1 - r2)/(4 pi r^2) sum_k (2k+1) C^k(1).C^k(2),
   it adds to every Slater integral R^k(ab; cd)
     (2k+1) (alpha^2/4) int R_a R_b R_c R_d r^2 dr,
   which is this function; ATSP's bpci adds the same in relativistic mode
   (librad/ykf.f). Checked against a determinant CI with the spin-dependent
   form on PySCF grid integrals of delta (tests/v5). */
double ac_radial_contact(const ac_radial *rad, int k, int a, int b, int c, int d);

/* z(r) = r^-q int_0^r u s^(2+p) ds + r^p int_r^inf u s^(2-q) ds, i.e. the
   potential of the density u(s) s^2 for the kernel r<^p / r>^q */
void ac_radial_kern(const ac_radial *rad, int p, int q, const double *u, double *z);

/* Orbit-orbit (ac_ci_oo). With the transverse kernel
   K_ij = delta_ij/r + r_i r_j/r^3 (div K = 0) only the currents
     J_xy = (phi_x* grad phi_y - phi_y grad phi_x*)/2
   of each electron contribute, and K = 2 delta/r - d_i d_j r gives
     <ab|H_OO|cd> = (alpha^2/2) [2 int int J_ab.J_cd / r12
                                 + int int Q_ab Q_cd r12],  Q = div J
   (electron 1 on a, b; electron 2 on c, d).
   ac_radial_ooj: the radial function <l_x||[C^k x J]^(g)||l_y>(r),
     (1/2)[T_xy - (-1)^(k+1-g+l_y-l_x) T_yx],
     T_xy = (-1)^(l_x+g+l_y) sqrt[g] sum_lam {k 1 g; l_y l_x lam}
            <l_x||C^k||lam> R_x <lam||grad||l_y> R_y.
   ac_radial_ooq: Q_xy = (R_x L R_y - R_y L R_x)/2, L R = R'' + 2R'/r - l(l+1)R/r^2
     (same angular structure as the density R_x R_y).
   ac_radial_oo: g >= 0: int int j_ab j_cd r<^k/r>^(k+1) r1^2 r2^2;
     g = -1: int int Q_ab Q_cd W_k r1^2 r2^2, W_k the multipole-k part of r12,
     r<^(k+2)/((2k+3) r>^(k+1)) - r<^k/((2k-1) r>^(k-1)). */
/* Two-electron gauge correction of g (ac_ci_gc2e, technical note, Sec. 5.1).
   After integrating by parts in electron 1 the operator density of
   electron 1 is -(2f + r d_r f) delta_km + r_m d_k f, f = phi_x* phi_y,
   whose orbital tensors are f itself and [R x grad]^(K') f, K' = 0, 1, 2
   (R = r C^1, grad on both orbitals). With the multipole C^k of 1/r12:
   ac_radial_gcj: the radial function <l_x||[C^k x [C^1 x grad]^(K')]^(g)||l_y>(r),
     q_xy + (-1)^(k-g+l_y-l_x) q_yx,
     q_xy = r R_x (-1)^(l_x+g+l_y) sqrt[g] sum_lam {k K' g; l_y l_x lam} <l_x||C^k||lam>
            (-1)^(lam+K'+l_y) sqrt[K'] sum_mu {1 1 K'; l_y lam mu} <lam||C^1||mu>
            <mu||grad||l_y> R_y   (Edmonds 7.1.1 twice; the second term is the
            gradient on the bra, as in ac_radial_ooj).
   ac_radial_gc: int int u_ab(r1) R_c R_d(r2) r<^k/r>^(k+1) r1^2 r2^2 with
     u_ab = that function (kp >= 0) or R_a R_b (kp = -1: the density term,
     i.e. the plain R^k(ab; cd)). */
void ac_radial_gcj(const ac_radial *rad, int k, int kp, int g, int x, int y, double *e);
double ac_radial_gc(const ac_radial *rad, int k, int kp, int g, int a, int b, int c, int d);
/* Orbit-orbit Zeeman term of g (ac_ci_oz2e): electron 1 (a, b) carries the
   position density, electron 2 (c, d) the current.
   type 0: int int r R_a R_b(r1) (-j^(k g)_cd)(r2) r<^k/r>^(k+1) r1^2 r2^2
           (j from ac_radial_ooj, sign corrected: one current);
   type 1: int int e_ab(r1) Q_cd(r2) W_k(r1, r2) r1^2 r2^2, e_ab the
           [C^k x [C^1 x grad]^(1)]^(g) density function of ac_radial_gcj
           (kp = 1) and W_k the r12 multipole of ac_radial_oo. */
double ac_radial_oz(const ac_radial *rad, int k, int g, int type, int a, int b, int c, int d);
void ac_radial_ooj(const ac_radial *rad, int k, int g, int x, int y, double *j);
void ac_radial_ooq(const ac_radial *rad, int x, int y, double *q);
double ac_radial_oo(const ac_radial *rad, int k, int g, int a, int b, int c, int d);

/* the same functions of single basis functions, for checks */
double ac_bfn_deriv(const ac_bfn *f, double r);

#endif
