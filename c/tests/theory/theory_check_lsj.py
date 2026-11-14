# lsj (technical note, Sec. 3.6):
#       (1) <S L J||S||S L J> from the reduced element of the first part (Sec. 3.1)
#           against the Lande closed form;
#       (2) the 6j {a b c; 1 c b} closed form used for the interval rule;
#       (3) the K = 0 case of the 9j LSJ reduction against the 6j J-scalar formula;
#       (4) <S L J||L||S L J> from that of the second part against its closed form.
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from theory_wig import sixj2, ninej2, sgn  # noqa: E402

bad = n = 0
for tS in range(0, 7):
    for L in range(0, 5):
        for tJ in range(abs(tS - 2 * L), tS + 2 * L + 1, 2):
            if tJ == 0:
                continue
            S, J = tS / 2, tJ / 2
            # (1) operator on part 1 (spin first): (-1)^(S+L+J'+k) sqrt([J][J']) {S J L; J' S k} <S||S||S>
            if tS > 0:
                v = sgn(tS + 2 * L + tJ + 2) * (tJ + 1) * sixj2(tS, tJ, 2 * L, tJ, tS, 2) * math.sqrt(S * (S + 1) * (tS + 1))
                c = math.sqrt(J * (J + 1) * (tJ + 1)) * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
                n += 1
                if abs(v - c) > 1e-12:
                    bad += 1
                    print("Lande S", tS, L, tJ, v, c)
            # (4) operator on part 2: (-1)^(S+L'+J+k) sqrt([J][J']) {L J S; J' L k} <L||L||L>
            if L > 0:
                v = sgn(tS + 2 * L + tJ + 2) * (tJ + 1) * sixj2(2 * L, tJ, tS, tJ, 2 * L, 2) * math.sqrt(L * (L + 1) * (2 * L + 1))
                c = math.sqrt(J * (J + 1) * (tJ + 1)) * (J * (J + 1) + L * (L + 1) - S * (S + 1)) / (2 * J * (J + 1))
                n += 1
                if abs(v - c) > 1e-12:
                    bad += 1
                    print("Lande L", tS, L, tJ, v, c)
            # (2) {J L S; 1 S L}
            if tS > 0 and L > 0:
                v = sixj2(tJ, 2 * L, tS, 2, tS, 2 * L)
                c = (sgn(tJ + 2 * L + tS + 2) * 2 * (L * (L + 1) + S * (S + 1) - J * (J + 1))
                     / math.sqrt(4 * L * (L + 1) * (2 * L + 1) * 4 * S * (S + 1) * (tS + 1)))
                n += 1
                if abs(v - c) > 1e-12:
                    bad += 1
                    print("6j", tS, L, tJ, v, c)
# (3) K = 0: sqrt([J][J]) 9j{S S' k; L L' k; J J 0} / sqrt[J] = ((-1)^k/sqrt[k]) (-1)^(S'+L+J) {J L S; k S' L'}
for tS in range(0, 5):
    for tSp in range(0, 5):
        for L in range(0, 4):
            for Lp in range(0, 4):
                for k in (1, 2):
                    for tJ in range(0, 9):
                        if (tS + tJ) % 2 or (tSp + tJ) % 2:
                            continue
                        v = (tJ + 1) * ninej2(tS, tSp, 2 * k, 2 * L, 2 * Lp, 2 * k, tJ, tJ, 0) / math.sqrt(tJ + 1)
                        c = sgn(2 * k) / math.sqrt(2 * k + 1) * sgn(tSp + 2 * L + tJ) * sixj2(tJ, 2 * L, tS, 2 * k, tSp, 2 * Lp)
                        n += 1
                        if abs(v - c) > 1e-12:
                            bad += 1
                            print("K0", tS, tSp, L, Lp, k, tJ, v, c)
print("lsj: tests", n, "bad", bad)
