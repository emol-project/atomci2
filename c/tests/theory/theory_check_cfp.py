# cfp (technical note, Sec. 3.3): the quasi-spin factor of the LS CFP.
# c_{N,ik} = <f~_N b f_{N-1}> (LS) against C = <F~ B G> (LSQ), F = N-electron (bra), G = ket.
# Code (ac_cfp_xcfp, id = +1, G_q = F_q + 1/2, i.e. ITYP 1/2, stored value = C directly):
#   v = C * den(ikm) den(iip) delta(2,kmu,imu) / (den(ikp) den(iim) den(isum)), sign rule.
# The closed form of the technical note:
#   c = (-1)^(2F_q + 1/2 - G_q - q_z) (F_q 1/2 G_q; -q_z 1/2 q_z') C,  q_z of F, q_z' = q_z - 1/2.
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "python"))
from pyref import threej2  # noqa: E402


def den(n):
    return math.sqrt(math.factorial(n - 1))


def tri_delta(ta, tb, tc):
    if (ta + tb + tc) % 2 or not (abs(ta - tb) <= tc <= ta + tb):
        return 0.0
    return math.sqrt(math.factorial((ta + tb - tc) // 2) * math.factorial((ta - tb + tc) // 2)
                     * math.factorial((-ta + tb + tc) // 2))


def code_factor(imu, kmu, iqkz):
    ikp = (kmu + iqkz + 1) // 2
    ikm = ikp - iqkz
    iim = (imu - iqkz) // 2
    iip = iim + iqkz + 1
    isum = (imu + kmu + 3) // 2
    v = den(ikm) * den(iip) * tri_delta(1, kmu - 1, imu - 1) / (den(ikp) * den(iim) * den(isum))
    if (kmu + kmu + imu + imu) % 4 == 0:
        v = -v
    return v


bad = n = 0
for l in range(0, 4):
    lfo = 2 * l + 1
    for tFq, tGq in [(a, a + s) for a in range(0, 2 * l + 2) for s in (1, -1)]:
        if tGq > 2 * l + 1 or tGq < 0:
            continue
        for N in range(1, 4 * l + 3):
            tqzF = N - lfo                    # 2 q_z of F (N electrons)
            tqzG = tqzF - 1                   # G has N-1 electrons
            if abs(tqzF) > tFq or abs(tqzG) > tGq or (tFq + tqzF) % 2 or (tGq + tqzG) % 2:
                continue
            fc = code_factor(tFq + 1, tGq + 1, tqzG)
            e2 = 2 * tFq + 1 - tGq - tqzF      # doubled exponent
            assert e2 % 2 == 0
            ph = -1.0 if (e2 // 2) % 2 else 1.0
            mine = ph * threej2(tFq, -tqzF, 1, 1, tGq, tqzG)
            n += 1
            if abs(fc - mine) > 1e-12:
                bad += 1
                print("l", l, "2Fq", tFq, "N", N, "code", fc, "formula", mine)
print("cfp quasi-spin factor: tests", n, "bad", bad)
