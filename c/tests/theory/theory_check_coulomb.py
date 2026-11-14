# coulomb (technical note, Sec. 4.1): (1) the Coulomb coefficient of the note,
#           (-1)^k <la||C^k||lb><lc||C^k||ld> / sqrt[k],
#       equals MOTECC's H^k = (-1)^((la-lb+lc-ld)/2) sqrt([la][lb][lc][ld]/[k]) F(k,la,lb) F(k,lc,ld)
#       with F(a,b,c) = |(a b c; 0 0 0)| = (-1)^g (a b c; 0 0 0), 2g = a+b+c;
#       (2) MOTECC's closed form of F read with (2g-2a)! (printed (2b-2a)!) is that |3j|;
#       (3) the contraction term of :((b_a b~_b)^(0k) (b_b b~_d)^(0k))^(00): from the contraction (Sec. 3.2),
#           -(-1)^(a+d) [k] {0 0 0; 1/2 1/2 1/2}{k k 0; ld la lb}, equals
#           -delta(la,ld) (-1)^(k+la+lb) sqrt[k]/sqrt(2[la]).
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "python"))
sys.path.insert(0, _here)
from pyref import threej2  # noqa: E402
from theory_wig import sixj2, sgn  # noqa: E402


def rmeC(l, k, lp):
    return (-1) ** l * math.sqrt((2 * l + 1) * (2 * lp + 1)) * threej2(2 * l, 0, 2 * k, 0, 2 * lp, 0)


def F_closed(a, b, c):
    if (a + b + c) % 2 or not (abs(b - c) <= a <= b + c):
        return 0.0
    g = (a + b + c) // 2
    f = math.factorial
    return f(g) * math.sqrt(f(2 * g - 2 * a) * f(2 * g - 2 * b) * f(2 * g - 2 * c) / f(2 * g + 1)) / (
        f(g - a) * f(g - b) * f(g - c))


bad = n = 0
for la in range(4):
    for lb in range(4):
        for lc in range(4):
            for ld in range(4):
                for k in range(7):
                    mine = (-1) ** k * rmeC(la, k, lb) * rmeC(lc, k, ld) / math.sqrt(2 * k + 1)
                    if (la + lb + k) % 2 or (lc + ld + k) % 2:
                        mot = 0.0
                    else:
                        Fa = F_closed(k, la, lb)
                        Fc = F_closed(k, lc, ld)
                        mot = ((-1) ** ((la - lb + lc - ld) // 2)
                               * math.sqrt((2 * la + 1) * (2 * lb + 1) * (2 * lc + 1) * (2 * ld + 1) / (2 * k + 1)) * Fa * Fc)
                        # (2): F is |3j|
                        g = (k + la + lb) // 2
                        n += 1
                        if abs(Fa - (-1) ** g * threej2(2 * k, 0, 2 * la, 0, 2 * lb, 0)) > 1e-12:
                            bad += 1
                            print("F", k, la, lb)
                    n += 1
                    if abs(mine - mot) > 1e-12:
                        bad += 1
                        print("Hk", la, lb, lc, ld, k, mine, mot)
# (3)
for la in range(4):
    for lb in range(4):
        for ld in range(4):
            for k in range(7):
                e2 = 1 + 1 + 2 * la + 2 * ld          # (a + d) doubled, a = (1/2, la)
                gen = -sgn(e2) * (2 * k + 1) * sixj2(0, 0, 0, 1, 1, 1) * sixj2(2 * k, 2 * k, 0, 2 * ld, 2 * la, 2 * lb)
                cf = 0.0
                if la == ld and abs(la - lb) <= k <= la + lb:
                    cf = -(-1) ** (k + la + lb) * math.sqrt(2 * k + 1) / math.sqrt(2 * (2 * la + 1))
                n += 1
                if abs(gen - cf) > 1e-12:
                    bad += 1
                    print("contr", la, lb, ld, k, gen, cf)
print("coulomb: tests", n, "bad", bad)
