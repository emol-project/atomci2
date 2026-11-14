/*
 * wigner_dump.c -- print ac_wigner values for argument lines read from stdin,
 * for the exact-arithmetic check in l0_exact.py.
 *
 * Input lines:  "6 tj1 .. tj6"  or  "9 tj1 .. tj9"   (all 2j)
 * Output lines: the same arguments followed by the value (%.17g).
 */
#include "ac_wigner.h"

#include <stdio.h>

int main(void)
{
    int kind;
    while (scanf("%d", &kind) == 1) {
        int t[9];
        int n = kind == 9 ? 9 : 6;
        for (int i = 0; i < n; i++)
            if (scanf("%d", &t[i]) != 1)
                return 1;
        double v = n == 9 ? ac_ninej(t[0], t[1], t[2], t[3], t[4], t[5], t[6],
                                     t[7], t[8])
                          : ac_sixj(t[0], t[1], t[2], t[3], t[4], t[5]);
        printf("%d", kind);
        for (int i = 0; i < n; i++)
            printf(" %d", t[i]);
        printf(" %.17g\n", v);
    }
    return 0;
}
