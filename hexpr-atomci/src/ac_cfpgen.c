/*
 * ac_cfpgen.c -- the CFP table generated in C (milestone M1b): a line-by-line
 * port of the Fortran programs cfpa (src/cfpa_1.f, cfpa_2.f) and cfpb
 * (src/cfpb.f), which write CFPDATA and CFPTBL.
 *
 * cfpa builds the shell states of l = 0..9 in the LSQ scheme by coupling the
 * shell creation operator to the states of the next higher quasi-spin,
 * orthonormalizing the candidates with the metric of NORMFP (a sequential
 * inverse Cholesky in candidate order; atomci2 technical note,
 * Sec. 3.3), and stores the CFP matrices. cfpb's SYMTLZ then rotates
 * the degenerate (Q, S, L) blocks of s, p, d, f onto the spin <-> quasi-spin
 * mirror blocks by linear least squares (DLLSQ). Everything that decides a
 * value is kept as in the Fortran, so that the table agrees with CFPTBL:
 *   - the arithmetic and its order (WIG6J with its own factorial tables,
 *     NORMFP, the Cholesky loop, DLLSQ, DGMPRD);
 *   - CMTTBL passes through the formatted file CFPDATA (F19.14) between the
 *     two programs: every value is rounded to 14 decimals before SYMTLZ;
 *   - SYMTLZ's EPS is single precision (REAL EPS = 1.0D-10).
 * Arrays are 1-based like the Fortran (index 0 unused).
 */
#include "ac_cfp.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, int errlen, const char *fmt, ...)
{
    if (err && errlen > 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, (size_t)errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ---- cfpa_2.f: WIG6J and DELTA (arguments are 2j+1) ------------------- */

static double FAC[58], DEN[58], RT[58]; /* [1..57] */
static double FASQ[155];                /* [1..154] */

static void wigsfp(void)
{
    FAC[1] = 1.0;
    for (int k = 1; k <= 57; k++) {
        if (k != 1)
            FAC[k] = FAC[k - 1] * (double)(k - 1);
        DEN[k] = sqrt(FAC[k]);
        RT[k] = sqrt((double)(k - 1));
    }
    for (int k = 1; k <= 154; k++)
        FASQ[k] = 0.0;
    double fac = 1.0;
    for (int k = 1; k <= 57; k++) {
        if (k != 1)
            fac = fac * (double)(k - 1);
        FASQ[2 * k + 39] = sqrt(fac);
    }
}

static double fasq(int i)
{
    return i >= 1 && i <= 154 ? FASQ[i] : 0.0;
}

static double delta(int ia, int ib, int ic)
{
    return fasq(ia + ib - ic + 40) * fasq(ia - ib + ic + 40) * fasq(-ia + ib + ic + 40);
}

static double ziz(int i)
{
    return (i / 2 + 1) % 2 ? -1.0 : 1.0;
}

static double pm1(int i) /* DBLE((-1)**I) */
{
    return i % 2 ? -1.0 : 1.0;
}

static double wig6j(int ia, int ib, int ic, int id, int ie, int iv)
{
    if (ia == 1) {
        if (ib != ic || ie != iv || delta(id, ie, ic) == 0.0)
            return 0.0;
        return ziz(id + ie + ic) / (RT[ib + 1] * RT[ie + 1]);
    }
    if (ib == 1) {
        if (ia != ic || id != iv || delta(id, ie, ic) == 0.0)
            return 0.0;
        return ziz(id + ie + ic) / (RT[ia + 1] * RT[id + 1]);
    }
    if (ic == 1) {
        if (ia != ib || id != ie || delta(ia, ie, iv) == 0.0)
            return 0.0;
        return ziz(ia + ie + iv) / (RT[ia + 1] * RT[id + 1]);
    }
    if (id == 1) {
        if (iv != ib || ic != ie || delta(ia, ib, ic) == 0.0)
            return 0.0;
        return ziz(ia + ib + ic) / (RT[ib + 1] * RT[ic + 1]);
    }
    if (ie == 1) {
        if (iv != ia || ic != id || delta(ia, ib, ic) == 0.0)
            return 0.0;
        return ziz(ia + ib + ic) / (RT[ia + 1] * RT[ic + 1]);
    }
    if (iv == 1) {
        if (ie != ia || ib != id || delta(ia, ib, ic) == 0.0)
            return 0.0;
        return ziz(ia + ib + ic) / (RT[ia + 1] * RT[ib + 1]);
    }
    double a = delta(ia, ib, ic) * delta(id, ie, ic) * delta(id, ib, iv) * delta(ia, ie, iv);
    if (a <= 0.0)
        return 0.0;
    int ik = (ia + ib + ic) / 2, il = (id + ie + ic) / 2, im = (id + ib + iv) / 2,
        in = (ia + ie + iv) / 2;
    int kk = (ia + ib + id + ie - 2) / 2, kl = (ia + ic + id + iv - 2) / 2,
        km = (ib + ic + ie + iv - 2) / 2;
    int iq = ik;
    if (il > iq) iq = il;
    if (im > iq) iq = im;
    if (in > iq) iq = in;
    int ip = kk;
    if (kl < ip) ip = kl;
    if (km < ip) ip = km;
    double g = 0.0;
    if (ip > 56)
        return 0.0; /* the Fortran prints a warning and returns G = 0 */
    for (int ir = iq; ir <= ip; ir++) {
        int jk = ir - ik + 1, jl = ir - il + 1, jm = ir - im + 1, jn = ir - in + 1;
        int lk = kk - ir + 1, ll = kl - ir + 1, lm = km - ir + 1;
        g = g - pm1(ir) * FAC[ir + 1] /
                    (FAC[jk] * FAC[jl] * FAC[jm] * FAC[jn] * FAC[lk] * FAC[ll] * FAC[lm]);
    }
    return g * a / (DEN[ik + 1] * DEN[il + 1] * DEN[im + 1] * DEN[in + 1]);
}

/* ---- cfpa_1.f ---------------------------------------------------------- */

#define ND 76 /* rows and columns of FP and CFP used */
#define NDA 88

static double CFP[NDA + 1][NDA + 1];
static int ISP[NDA + 1][NDA + 1], IORB[NDA + 1][NDA + 1], INUM[NDA + 1];
static int IMUMAX, ISPSHL, LSHELL;

static double zf(int i) /* NORMFP's Z(I) = DBLE((-1)**(I/2)) */
{
    return (i / 2) % 2 ? -1.0 : 1.0;
}

static int lsqnfp(const int *lad, int l, int is, int iq)
{
    int m = (l + 1) / 2;
    return lad[m] + (l * (l + 6 - 2 * iq) + iq * (iq - 6) - 2 * is + 8) / 4;
}

static double normfp(int imu, int i, int j, int is, int ir)
{
    int isi = ISP[imu][i], isj = ISP[imu][j], iri = IORB[imu][i], irj = IORB[imu][j];
    double w2 = wig6j(imu, 2, imu - 1, imu, 2, imu + 1) * (double)(imu - 1);
    double w1 = wig6j(imu, 2, imu + 1, imu, 2, imu + 1) / (double)(is * ir);
    double w = 0.0;
    if (i == j)
        w = zf(2 * imu);
    if (imu == IMUMAX)
        return w / w2;
    double wsp = 0.0, wg = 0.0;
    int kmax = INUM[imu + 1];
    for (int k = 1; k <= kmax; k++) {
        double a = CFP[i][k] * CFP[j][k];
        if (a == 0.0)
            continue;
        int isk = ISP[imu + 1][k], irk = IORB[imu + 1][k];
        if (isk == is && irk == ir)
            wsp = wsp + a;
        double h = wig6j(isi, ISPSHL, is, isj, ISPSHL, isk);
        if (h == 0.0)
            continue;
        h = h * wig6j(iri, LSHELL, ir, irj, LSHELL, irk);
        if (h == 0.0)
            continue;
        wg = wg + h * a;
    }
    return (w + zf(isi + iri - isj - irj + 1002) * (w1 * wsp + wg * zf(2 * (imu + is + ir)) /
                                                               (double)(imu + 1))) /
           w2;
}

static int findfp(int i, int j, int k)
{
    if ((i + j + k) % 2 == 0)
        return 0;
    if (i + j - k < 1 || abs(i - j) > k - 1)
        return 0;
    return 1;
}

/* the raw table of cfpa (sizes as declared in cfpa_1.f) */
typedef struct {
    int ibel, icnn, ilfn, icmtn, icdx;
    int iconnt[100 + 1][9 + 1]; /* [j][i] = ICONNT(i, j) */
    int icmtdx[120 + 1][5 + 1];
    int ilfunc[700 + 1];
    double cmttbl[9000 + 1];
    int ilstad[21 + 1], immint[21 + 1];
} rawtab;

static int cfpa(rawtab *R, char *err, int errlen)
{
    static double FP[NDA + 1][NDA + 1], S[NDA + 1][NDA + 1], CTE[NDA + 1][NDA + 1], D[NDA + 1];
    static int ICAN[NDA + 1], ICANC[NDA + 1], ICANM[NDA + 1];
    static const int mumaxa[21] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static const int mumina[21] = {0, 0, 0, 0, 0, 8, 10, 12, 14, 16, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int lad[22] = {0};
    memset(R, 0, sizeof *R);
    memset(CFP, 0, sizeof CFP);
    memset(ISP, 0, sizeof ISP);
    memset(IORB, 0, sizeof IORB);
    memset(INUM, 0, sizeof INUM);
    wigsfp();
    const double sles = 1.0e-5;
    int ibel = 0, icnn = 0, ilfn = 0, icmtn = 0, icdx = 0, ncount = 0;
    double wmaxd = 0.0, wmaxof = 0.0;
    (void)wmaxd;
    (void)wmaxof;
    for (;;) { /* 1000 */
        for (int j = 1; j <= ND; j++)
            for (int k = 1; k <= ND; k++)
                FP[j][k] = 0.0;
        ncount++;
        if (ncount > 20)
            break;
        LSHELL = 0;
        ISPSHL = 0;
        IMUMAX = mumaxa[ncount];
        int imumin = mumina[ncount];
        if (imumin == 0)
            imumin = 1;
        LSHELL = ibel * 2 + 1;
        if (IMUMAX == 0)
            break; /* 220 */
        ibel++;
        IMUMAX = LSHELL + 1;
        ISPSHL = 2;
        int imu = IMUMAX;
        ISP[imu][1] = 1;
        IORB[imu][1] = 1;
        INUM[imu] = 1;
        int ix = 1;
        R->ilstad[ibel] = icnn + 1;
        lad[ibel] = icnn + 1;
        R->immint[ibel] = imumin;
        int next_shell = 0;
        for (int imc = 1; imc <= IMUMAX && !next_shell; imc++) {
            imu = IMUMAX - imc + 1;
            for (int j = 1; j <= ND; j++)
                for (int k = 1; k <= ND; k++) {
                    CFP[j][k] = FP[j][k];
                    FP[j][k] = 0.0;
                }
            if (ix == 0)
                break; /* GO TO 1000 */
            /* the (LFO, Q, S) blocks and their L lists */
            int ispt = 0, icn4 = 0;
            for (int i = 1; i <= ix; i++) {
                if (ispt != ISP[imu][i]) {
                    ispt = ISP[imu][i];
                    if (i != 1)
                        R->iconnt[icnn][4] = icn4;
                    icnn++;
                    if (icnn > 100)
                        return fail(err, errlen, "cfpa: ICONNT overflow");
                    icn4 = 0;
                    R->iconnt[icnn][1] = LSHELL;
                    R->iconnt[icnn][2] = imu;
                    R->iconnt[icnn][3] = ispt;
                    R->iconnt[icnn][5] = ilfn + 1;
                }
                ilfn++;
                if (ilfn > 700)
                    return fail(err, errlen, "cfpa: ILFUNC overflow");
                R->ilfunc[ilfn] = IORB[imu][i];
                icn4++;
            }
            R->iconnt[icnn][4] = icn4;
            if (imu != IMUMAX) {
                /* store the CFP matrices between imu and imu+1 */
                int ktea = 0;
                int ispmax = LSHELL - imu + 1;
                for (int i = 1; i <= ispmax; i += 2) {
                    int isptA = ispmax - i + 1;
                    int insqa = lsqnfp(lad, LSHELL, isptA, imu + 1);
                    int knsqa = R->iconnt[insqa][4];
                    int ispmxb = ispmax + 1, ktebt = 0;
                    for (int ii = 1; ii <= ispmxb; ii += 2) {
                        int isptb = ispmxb - ii + 1;
                        int insqb = lsqnfp(lad, LSHELL, isptb, imu);
                        int knsqb = R->iconnt[insqb][4];
                        if (abs(isptb - isptA) == 1) {
                            icmtn++;
                            if (icmtn > 120)
                                return fail(err, errlen, "cfpa: ICMTDX overflow");
                            R->icmtdx[icmtn][1] = icdx + 1;
                            R->icmtdx[icmtn][2] = knsqb;
                            R->icmtdx[icmtn][3] = knsqa;
                            R->icmtdx[icmtn][4] = insqb;
                            R->icmtdx[icmtn][5] = insqa;
                            int kte = ktea;
                            for (int k = 1; k <= knsqa; k++) {
                                kte++;
                                int kteb = ktebt;
                                for (int j = 1; j <= knsqb; j++) {
                                    kteb++;
                                    icdx++;
                                    if (icdx > 9000)
                                        return fail(err, errlen, "cfpa: CMTTBL overflow");
                                    R->cmttbl[icdx] = CFP[kteb][kte];
                                }
                            }
                        }
                        ktebt += knsqb;
                    }
                    ktea += knsqa;
                }
            }
            /* 50 */
            wmaxd = 0.0;
            wmaxof = 0.0;
            int inmax = INUM[imu];
            if (imu == imumin) {
                next_shell = 1;
                break;
            }
            /* range of possible representations */
            int iormax = 0, ispmax = 0;
            for (int i = 1; i <= inmax; i++) {
                if (IORB[imu][i] > iormax)
                    iormax = IORB[imu][i];
                if (ISP[imu][i] > ispmax)
                    ispmax = ISP[imu][i];
            }
            iormax = iormax + LSHELL - 1;
            ispmax = ispmax + ISPSHL - 1;
            /* CFP generation */
            ix = 0;
            for (int isc = 1; isc <= ispmax; isc++) {
                int ispt2 = ispmax - isc + 1, itec = 0;
                for (int i = 1; i <= inmax; i++)
                    if (findfp(ispt2, ISPSHL, ISP[imu][i]))
                        ICANM[++itec] = i;
                if (itec == 0)
                    continue;
                for (int irc = 1; irc <= iormax; irc++) {
                    int iort = iormax - irc + 1, ite = 0;
                    for (int i = 1; i <= itec; i++) {
                        int itm = ICANM[i];
                        if (findfp(iort, LSHELL, IORB[imu][itm]))
                            ICAN[++ite] = itm;
                    }
                    if (ite == 0)
                        continue;
                    int itm = 0;
                    for (int i = 1; i <= ite; i++) {
                        double w = normfp(imu, ICAN[i], ICAN[i], ispt2, iort);
                        if (fabs(w) < sles) {
                            wmaxd = fmax(wmaxd, fabs(w));
                            continue;
                        }
                        itm++;
                        S[itm][itm] = w;
                        ICAN[itm] = ICAN[i];
                    }
                    if (itm == 0)
                        continue;
                    for (int i = 2; i <= itm; i++)
                        for (int j = 1; j <= i - 1; j++) {
                            double w = normfp(imu, ICAN[i], ICAN[j], ispt2, iort);
                            S[i][j] = w;
                            S[j][i] = w;
                        }
                    int n = 1;
                    CTE[1][1] = 1.0 / sqrt(S[1][1]);
                    ICANC[1] = 1;
                    for (int l = 2; l <= itm; l++) {
                        double w = S[l][l];
                        for (int k = 1; k <= n; k++) {
                            double dtem = 0.0;
                            for (int ip = 1; ip <= k; ip++) {
                                int ipp = ICANC[ip];
                                dtem = dtem + CTE[k][ip] * S[ipp][l];
                            }
                            D[k] = dtem;
                            w = w - dtem * dtem;
                        }
                        if (fabs(w) < sles) {
                            wmaxof = fmax(wmaxof, fabs(w));
                            continue;
                        }
                        n++;
                        ICANC[n] = l;
                        CTE[n][n] = 1.0 / sqrt(w);
                        int nm1 = n - 1;
                        for (int ip = 1; ip <= nm1; ip++) {
                            double dtem = 0.0;
                            for (int k = ip; k <= nm1; k++)
                                dtem = dtem - D[k] * CTE[k][ip];
                            CTE[n][ip] = dtem * CTE[n][n];
                        }
                    }
                    /* store the CFP */
                    double wfc = sqrt((double)(ispt2 * iort * (imu - 1)));
                    for (int i = 1; i <= n; i++) {
                        ix++;
                        if (ix > ND)
                            return fail(err, errlen, "cfpa: INUM exceeds %d", ND);
                        ISP[imu - 1][ix] = ispt2;
                        IORB[imu - 1][ix] = iort;
                        for (int j = 1; j <= itm; j++) {
                            int jp = ICAN[j];
                            double ctem = 0.0;
                            for (int ip = 1; ip <= i; ip++) {
                                int ipp = ICANC[ip];
                                ctem = ctem + CTE[i][ip] * S[ipp][j];
                            }
                            FP[ix][jp] = wfc * ctem;
                        }
                    }
                }
            }
            INUM[imu - 1] = ix;
        }
    }
    /* 220: the links of every block to the CFP matrices */
    for (int i = 1; i <= icmtn; i++) {
        int insqa = R->icmtdx[i][4], insqb = R->icmtdx[i][5];
        int isa = R->iconnt[insqa][3], isb = R->iconnt[insqb][3];
        int iqa = R->iconnt[insqa][2], ish = R->iconnt[insqa][1];
        if (isa <= isb) {
            R->iconnt[insqa][8] = i;
            R->iconnt[insqb][7] = i;
        } else {
            R->iconnt[insqa][9] = i;
            R->iconnt[insqb][6] = i;
        }
        int b = (ish + 1) / 2;
        if (iqa > R->immint[b])
            continue;
        if (iqa == 1)
            continue;
        R->iconnt[insqa][6] = -1;
        R->iconnt[insqa][7] = -1;
    }
    R->ibel = ibel;
    R->icnn = icnn;
    R->ilfn = ilfn;
    R->icmtn = icmtn;
    R->icdx = icdx;
    /* CFPDATA: CMTTBL written with F19.14 and read back by cfpb */
    for (int i = 1; i <= icdx; i++) {
        char buf[64];
        snprintf(buf, sizeof buf, "%19.14f", R->cmttbl[i]);
        R->cmttbl[i] = strtod(buf, NULL);
    }
    return 0;
}

/* ---- cfpb.f ------------------------------------------------------------ */

static rawtab *T; /* the table SYMTLZ works on */

static int lsqn(int l, int iq, int is)
{
    int m = (l + 1) / 2;
    return (l * (l + 6 - 2 * iq) + iq * (iq - 6) - 2 * is + 8) / 4 + T->ilstad[m];
}

/* CFPV: the CFP value, its CMTTBL address and whether its sign was flipped */
static double cfpv(int i, int k, int idim, int kdim, int ityp, int nstad, int l, int ilad,
                   int klad, int *ncfp, int *ipr)
{
    *ipr = 0;
    if (ityp >= 3)
        *ncfp = nstad + k + (i - 1) * kdim - 1;
    else
        *ncfp = nstad + i + (k - 1) * idim - 1;
    double v = T->cmttbl[*ncfp];
    if (ityp < 3)
        return v;
    int inad = ilad + i - 1, knad = klad + k - 1;
    int ipar = l + T->ilfunc[inad] - T->ilfunc[knad] + 2 * ityp + 400;
    if (ipar % 4 == 3)
        return v;
    *ipr = 1;
    return -v;
}

static int bad_link;

static void cfptbl(int l, int imu, int isp, int kmu, int ksp, int *ishl, int *kshl, int *idim,
                   int *kdim, int *ilad, int *klad, int *nstad, int *ityp)
{
    *ishl = lsqn(l, imu, isp);
    *kshl = lsqn(l, kmu, ksp);
    *ityp = 0;
    int icft;
    if (kmu - imu == 1) {
        if (isp - ksp == 1) {
            *ityp = 1;
            icft = T->iconnt[*kshl][6];
        } else if (isp - ksp == -1) {
            *ityp = 2;
            icft = T->iconnt[*kshl][7];
        } else
            return;
    } else if (kmu - imu == -1) {
        if (isp - ksp == 1) {
            *ityp = 3;
            icft = T->iconnt[*kshl][8];
        } else if (isp - ksp == -1) {
            *ityp = 4;
            icft = T->iconnt[*kshl][9];
        } else
            return;
    } else
        return;
    if (icft == 0)
        return;
    if (icft == -1) {
        bad_link = 1; /* the Fortran prints INVALID PARAM. and leaves the outputs */
        return;
    }
    *idim = T->iconnt[*ishl][4];
    *kdim = T->iconnt[*kshl][4];
    *ilad = T->iconnt[*ishl][5];
    *klad = T->iconnt[*kshl][5];
    *nstad = T->icmtdx[icft][1];
}

/* DGMPRD: R(N x L) = A(N x M) B(M x L), column-major, 1-based flat */
static void dgmprd(const double *a, const double *b, double *r, int n, int m, int l)
{
    int ir = 0, ik = -m;
    for (int k = 1; k <= l; k++) {
        ik = ik + m;
        for (int j = 1; j <= n; j++) {
            ir = ir + 1;
            int ji = j - n, ib = ik;
            r[ir] = 0.0;
            for (int i = 1; i <= m; i++) {
                ji = ji + n;
                ib = ib + 1;
                r[ir] = r[ir] + a[ji] * b[ib];
            }
        }
    }
}

/* DLLSQ (IBM SSP): Householder least squares, a literal port (1-based flat) */
static void dllsq(double *a, double *b, int m, int n, int l, double *x, int *ipiv, float eps,
                  int *ier, double *aux)
{
    int k, i, j, ist, iend, kpiv = 0, id, ii, jst, kst, lm, ln;
    double piv, h, sig, tol, beta;
    if (m - n < 0) {
        *ier = -2;
        return;
    }
    piv = 0.0;
    iend = 0;
    for (k = 1; k <= n; k++) {
        ipiv[k] = k;
        h = 0.0;
        ist = iend + 1;
        iend = iend + m;
        for (i = ist; i <= iend; i++)
            h = h + a[i] * a[i];
        aux[k] = h;
        if (h - piv > 0.0) {
            piv = h;
            kpiv = k;
        }
    }
    if (piv <= 0.0) {
        *ier = -1;
        return;
    }
    sig = sqrt(piv);
    tol = sig * (double)fabsf(eps);
    lm = l * m;
    ist = -m;
    for (k = 1; k <= n; k++) {
        ist = ist + m + 1;
        iend = ist + m - k;
        i = kpiv - k;
        if (i > 0) {
            h = aux[k];
            aux[k] = aux[kpiv];
            aux[kpiv] = h;
            id = i * m;
            for (i = ist; i <= iend; i++) {
                j = i + id;
                h = a[i];
                a[i] = a[j];
                a[j] = h;
            }
        }
        if (k - 1 > 0) {
            sig = 0.0;
            for (i = ist; i <= iend; i++)
                sig = sig + a[i] * a[i];
            sig = sqrt(sig);
            if (sig - tol <= 0.0) {
                *ier = k - 1;
                return;
            }
        }
        h = a[ist];
        if (h < 0.0)
            sig = -sig;
        ipiv[kpiv] = ipiv[k];
        ipiv[k] = kpiv;
        beta = h + sig;
        a[ist] = beta;
        beta = 1.0 / (sig * beta);
        j = n + k;
        aux[j] = -sig;
        if (k - n < 0) {
            piv = 0.0;
            id = 0;
            jst = k + 1;
            kpiv = jst;
            for (j = jst; j <= n; j++) {
                id = id + m;
                h = 0.0;
                for (i = ist; i <= iend; i++) {
                    ii = i + id;
                    h = h + a[i] * a[ii];
                }
                h = beta * h;
                for (i = ist; i <= iend; i++) {
                    ii = i + id;
                    a[ii] = a[ii] - a[i] * h;
                }
                ii = ist + id;
                h = aux[j] - a[ii] * a[ii];
                aux[j] = h;
                if (h - piv > 0.0) {
                    piv = h;
                    kpiv = j;
                }
            }
        }
        for (j = k; j <= lm; j += m) {
            h = 0.0;
            iend = j + m - k;
            ii = ist;
            for (i = j; i <= iend; i++) {
                h = h + a[ii] * b[i];
                ii = ii + 1;
            }
            h = beta * h;
            ii = ist;
            for (i = j; i <= iend; i++) {
                b[i] = b[i] - a[ii] * h;
                ii = ii + 1;
            }
        }
    }
    *ier = 0;
    i = n;
    ln = l * n;
    piv = 1.0 / aux[2 * n];
    for (k = n; k <= ln; k += n) {
        x[k] = piv * b[i];
        i = i + m;
    }
    if (n - 1 > 0) {
        jst = (n - 1) * m + n;
        for (j = 2; j <= n; j++) {
            jst = jst - m - 1;
            k = n + n + 1 - j;
            piv = 1.0 / aux[k];
            kst = k - n;
            id = ipiv[kst] - kst;
            ist = 2 - j;
            for (k = 1; k <= l; k++) {
                h = b[kst];
                ist = ist + n;
                iend = ist + j - 2;
                ii = jst;
                for (i = ist; i <= iend; i++) {
                    ii = ii + m;
                    h = h - a[ii] * x[i];
                }
                i = ist - 1;
                ii = i + id;
                x[i] = x[ii];
                x[ii] = piv * h;
                kst = kst + m;
            }
        }
    }
    ist = n + 1;
    iend = 0;
    for (j = 1; j <= l; j++) {
        iend = iend + m;
        h = 0.0;
        if (m - n > 0) {
            for (i = ist; i <= iend; i++)
                h = h + b[i] * b[i];
            ist = ist + m;
        }
        aux[j] = h;
    }
}

static int symtlz(int lfx, char *err, int errlen)
{
    /* the Fortran's sizes are C(490), X(49), AUX(20); these are larger, and
       the indices are checked as they grow */
    enum { NB = 4096 };
    static double C[NB], B[NB], X[NB], Dd[NB], AUX[NB], AC[NB];
    static int ITST[NB], IADRS[NB], IADRDS[NB], IPARS[NB], IPARDS[NB];
    int lfo = lfx;
    const float eps = (float)1.0e-10;
    for (int isxt = 1; isxt <= lfo; isxt += 2) {
        int isst = lfo + 2 - isxt;
        for (int imxt = 2; imxt <= isst; imxt += 2) {
            int imu = imxt / 2, isp = isst - imu + 1;
            if (isp == lfo + 1)
                continue;
            int ipna = imu != 1, ipnb = isxt != 1, idna = imu != 1, idnb = 1;
            int ipshla = 0, isshl = 0, ipdima = 0, isdim = 0, iplada = 0, islad = 0, npstda = 0,
                itpypa = 0;
            int iqshla = 0, icshl = 0, iqdima = 0, iqlada = 0, iclad = 0, nqstda = 0, itypqa = 0;
            int ipshlb = 0, ipdimb = 0, ipladb = 0, npstdb = 0, itpypb = 0;
            int iqshlb = 0, iqdimb = 0, iqladb = 0, nqstdb = 0, itypqb = 0;
            int idshla = 0, iddima = 0, idlada = 0, ndstda = 0, itypda = 0;
            int idshlb = 0, iddimb = 0, idladb = 0, ndstdb = 0, itypdb = 0;
            if (ipna)
                cfptbl(lfo, imu - 1, isp + 1, imu, isp, &ipshla, &isshl, &ipdima, &isdim, &iplada,
                       &islad, &npstda, &itpypa);
            if (ipna)
                cfptbl(lfo, isp + 1, imu - 1, isp, imu, &iqshla, &icshl, &iqdima, &isdim, &iqlada,
                       &iclad, &nqstda, &itypqa);
            if (ipnb)
                cfptbl(lfo, imu + 1, isp + 1, imu, isp, &ipshlb, &isshl, &ipdimb, &isdim, &ipladb,
                       &islad, &npstdb, &itpypb);
            if (ipnb)
                cfptbl(lfo, isp + 1, imu + 1, isp, imu, &iqshlb, &icshl, &iqdimb, &isdim, &iqladb,
                       &iclad, &nqstdb, &itypqb);
            if (idna)
                cfptbl(lfo, imu - 1, isp - 1, imu, isp, &idshla, &isshl, &iddima, &isdim, &idlada,
                       &islad, &ndstda, &itypda);
            if (idnb)
                cfptbl(lfo, imu + 1, isp - 1, imu, isp, &idshlb, &isshl, &iddimb, &isdim, &idladb,
                       &islad, &ndstdb, &itypdb);
            if (bad_link)
                return fail(err, errlen, "SYMTLZ: invalid CFP link (l = %d)", (lfo - 1) / 2);
            int isld = 0, icd = 0, idd = 0;
            const double sgn = (double)(lfo % 4 - 2);
            for (int isvc = 1; isvc <= isdim; isvc++) {
                int nadl = islad + isvc - 1;
                isld++;
                if (icd + ipdima + ipdimb >= NB || idd + iddima + iddimb >= NB ||
                    isld * isld >= NB)
                    return fail(err, errlen, "SYMTLZ: block too large");
                int idm = 0, iadr, ipar;
                if (ipna) {
                    idm = ipdima;
                    for (int i = 1; i <= ipdima; i++) {
                        icd++;
                        B[icd] = cfpv(i, isvc, iqdima, isdim, itypqa, nqstda, lfo, iqlada, iclad,
                                      &iadr, &ipar) *
                                 sgn;
                        C[icd] = cfpv(i, isvc, ipdima, isdim, itpypa, npstda, lfo, iplada, islad,
                                      &iadr, &ipar);
                        IADRS[icd] = iadr;
                        IPARS[icd] = ipar;
                    }
                }
                if (ipnb) {
                    idm = idm + ipdimb;
                    for (int i = 1; i <= ipdimb; i++) {
                        icd++;
                        B[icd] = cfpv(i, isvc, iqdimb, isdim, itypqb, nqstdb, lfo, iqladb, iclad,
                                      &iadr, &ipar) *
                                 sgn;
                        C[icd] = cfpv(i, isvc, ipdimb, isdim, itpypb, npstdb, lfo, ipladb, islad,
                                      &iadr, &ipar);
                        IADRS[icd] = iadr;
                        IPARS[icd] = ipar;
                    }
                }
                int idmd = 0;
                if (idna) {
                    idmd = iddima;
                    for (int i = 1; i <= iddima; i++) {
                        idd++;
                        Dd[idd] = cfpv(i, isvc, iddima, isdim, itypda, ndstda, lfo, idlada, islad,
                                       &iadr, &ipar);
                        IADRDS[idd] = iadr;
                        IPARDS[idd] = ipar;
                    }
                }
                if (idnb) {
                    idmd = idmd + iddimb;
                    for (int i = 1; i <= iddimb; i++) {
                        idd++;
                        Dd[idd] = cfpv(i, isvc, iddimb, isdim, itypdb, ndstdb, lfo, idladb, islad,
                                       &iadr, &ipar);
                        IADRDS[idd] = iadr;
                        IPARDS[idd] = ipar;
                    }
                }
                if (isvc != isdim && T->ilfunc[nadl] == T->ilfunc[nadl + 1])
                    continue; /* the next state has the same L: one block */
                int imv = idm * isld;
                for (int k = 1; k <= imv; k++)
                    AC[k] = C[k];
                if (isld == 1) {
                    double at = 0.0;
                    int kt = 1;
                    for (int k = 1; k <= idm; k++) {
                        if (fabs(C[k]) <= at)
                            continue;
                        kt = k;
                        at = fabs(C[k]);
                    }
                    X[1] = B[kt] / C[kt];
                    AUX[1] = 0.0;
                    for (int k = 1; k <= idm; k++)
                        AUX[1] = (B[k] - X[1] * C[k]) * (B[k] - X[1] * C[k]) + AUX[1];
                } else {
                    int ier;
                    dllsq(C, B, idm, isld, isld, X, ITST, eps, &ier, AUX);
                }
                dgmprd(AC, X, C, idm, isld, isld);
                for (int k = 1; k <= imv; k++)
                    T->cmttbl[IADRS[k]] = IPARS[k] == 1 ? -C[k] : C[k];
                dgmprd(Dd, X, C, idmd, isld, isld);
                imv = idmd * isld;
                for (int k = 1; k <= imv; k++)
                    T->cmttbl[IADRDS[k]] = IPARDS[k] == 1 ? -C[k] : C[k];
                icd = 0;
                idd = 0;
                isld = 0;
            }
        }
    }
    return 0;
}

int ac_cfp_generate(ac_cfptab *t, char *err, int errlen)
{
    memset(t, 0, sizeof *t);
    rawtab *R = calloc(1, sizeof *R);
    if (!R)
        return fail(err, errlen, "out of memory");
    int rc = -1;
    if (cfpa(R, err, errlen))
        goto out;
    T = R;
    bad_link = 0;
    for (int i = 1, k = 1; i <= 7; i += 2, k++)
        if (R->immint[k] == 1 && symtlz(i, err, errlen))
            goto out;
    /* the CFPTBL records */
    char name[81];
    snprintf(name, sizeof name, "%-80s",
             " coefficient of fractional parentage. s-f: complete, up to l=9");
    memcpy(t->name, name, sizeof t->name);
    t->ibel = R->ibel;
    t->icnn = R->icnn;
    t->ilfn = R->ilfn;
    t->icmtn = R->icmtn;
    t->icdx = R->icdx;
    for (int i = 0; i < 18; i++) {
        t->immint[i] = R->immint[i + 1];
        t->ilstad[i] = R->ilstad[i + 1];
    }
    t->iconnt = malloc((size_t)t->icnn * sizeof *t->iconnt);
    t->icmtdx = malloc((size_t)t->icmtn * sizeof *t->icmtdx);
    t->ilfunc = malloc((size_t)t->ilfn * sizeof *t->ilfunc);
    t->cmttbl = malloc((size_t)t->icdx * sizeof *t->cmttbl);
    if (!t->iconnt || !t->icmtdx || !t->ilfunc || !t->cmttbl) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    for (int j = 0; j < t->icnn; j++)
        for (int i = 0; i < 9; i++)
            t->iconnt[j][i] = R->iconnt[j + 1][i + 1];
    for (int j = 0; j < t->icmtn; j++)
        for (int i = 0; i < 5; i++)
            t->icmtdx[j][i] = R->icmtdx[j + 1][i + 1];
    for (int i = 0; i < t->ilfn; i++)
        t->ilfunc[i] = R->ilfunc[i + 1];
    for (int i = 0; i < t->icdx; i++)
        t->cmttbl[i] = R->cmttbl[i + 1];
    if (ac_cfp_build_lists(t)) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    rc = 0;
out:
    free(R);
    if (rc)
        ac_cfp_free(t);
    return rc;
}

int ac_cfp_write(const ac_cfptab *t, const char *path, char *err, int errlen)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return fail(err, errlen, "cannot create %s", path);
    int ok = 1;
#define REC(n) do { int32_t m_ = (int32_t)(n); ok &= fwrite(&m_, 4, 1, f) == 1; } while (0)
    int32_t r1[59];
    for (int i = 0; i < 18; i++)
        r1[i] = t->name[i];
    r1[18] = t->ibel;
    r1[19] = t->icnn;
    r1[20] = t->ilfn;
    r1[21] = t->icmtn;
    r1[22] = t->icdx;
    for (int i = 0; i < 18; i++) {
        r1[23 + i] = t->immint[i];
        r1[41 + i] = t->ilstad[i];
    }
    REC(sizeof r1);
    ok &= fwrite(r1, sizeof r1, 1, f) == 1;
    REC(sizeof r1);
    REC((size_t)t->icnn * 9 * 4);
    for (int j = 0; j < t->icnn; j++)
        for (int i = 0; i < 9; i++) {
            int32_t v = t->iconnt[j][i];
            ok &= fwrite(&v, 4, 1, f) == 1;
        }
    REC((size_t)t->icnn * 9 * 4);
    REC((size_t)t->icmtn * 5 * 4);
    for (int j = 0; j < t->icmtn; j++)
        for (int i = 0; i < 5; i++) {
            int32_t v = t->icmtdx[j][i];
            ok &= fwrite(&v, 4, 1, f) == 1;
        }
    REC((size_t)t->icmtn * 5 * 4);
    REC((size_t)t->ilfn * 4);
    for (int i = 0; i < t->ilfn; i++) {
        int32_t v = t->ilfunc[i];
        ok &= fwrite(&v, 4, 1, f) == 1;
    }
    REC((size_t)t->ilfn * 4);
    REC((size_t)t->icdx * 8);
    ok &= fwrite(t->cmttbl, 8, (size_t)t->icdx, f) == (size_t)t->icdx;
    REC((size_t)t->icdx * 8);
#undef REC
    ok &= fclose(f) == 0;
    return ok ? 0 : fail(err, errlen, "write error on %s", path);
}
