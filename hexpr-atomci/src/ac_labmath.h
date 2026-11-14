/*
 * ac_labmath.h -- internal helpers for (2S, 2L) labels of O(3) x SU(2).
 *
 * A label A stands for the irreducible representation (S, L); [A] is its
 * dimension and an exponent "A" in a phase means S + L (METECC section 2).
 * 6j and 9j symbols of labels factor into a spin and an orbital symbol.
 */
#ifndef AC_LABMATH_H
#define AC_LABMATH_H

#include "ac_csf.h"
#include "ac_wigner.h"

#include <math.h>

static inline int lab_eq(ac_lab a, ac_lab b)
{
    return a.ts == b.ts && a.tl == b.tl;
}

static inline double lab_dim(ac_lab a)
{
    return (double)(a.ts + 1) * (double)(a.tl + 1);
}

/* the exponent S + L of a label, in doubled units */
static inline int lab_e2(ac_lab a)
{
    return a.ts + a.tl;
}

/* (-1)^(e2/2) for an even doubled exponent; NaN flags an odd one */
static inline double phase2(int e2)
{
    if (e2 & 1)
        return NAN;
    return ((e2 / 2) & 1) ? -1.0 : 1.0;
}

static inline double lab_sixj(ac_lab a, ac_lab b, ac_lab c, ac_lab d, ac_lab e,
                              ac_lab f)
{
    double s = ac_sixj(a.ts, b.ts, c.ts, d.ts, e.ts, f.ts);
    if (s == 0.0)
        return 0.0;
    return s * ac_sixj(a.tl, b.tl, c.tl, d.tl, e.tl, f.tl);
}

static inline double lab_ninej(ac_lab a, ac_lab b, ac_lab c, ac_lab d, ac_lab e,
                               ac_lab f, ac_lab g, ac_lab h, ac_lab i)
{
    double s = ac_ninej(a.ts, b.ts, c.ts, d.ts, e.ts, f.ts, g.ts, h.ts, i.ts);
    if (s == 0.0)
        return 0.0;
    return s * ac_ninej(a.tl, b.tl, c.tl, d.tl, e.tl, f.tl, g.tl, h.tl, i.tl);
}

/* triangle of the spin and of the orbital parts */
static inline int lab_tri(ac_lab a, ac_lab b, ac_lab c)
{
    int s = a.ts + b.ts + c.ts, l = a.tl + b.tl + c.tl;
    if ((s & 1) || (l & 1))
        return 0;
    return c.ts <= a.ts + b.ts && c.ts >= (a.ts > b.ts ? a.ts - b.ts : b.ts - a.ts) &&
           c.tl <= a.tl + b.tl && c.tl >= (a.tl > b.tl ? a.tl - b.tl : b.tl - a.tl);
}

#endif
