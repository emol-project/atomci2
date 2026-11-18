"""terms_check.py -- the checks of terms.sh (see there): argv[1] the directory, then the systems.

Everything is taken from the files acfs writes (csf.tsv, meta.json, .npy), nothing from the C
code: the 6j symbol is Racah's sum, the J blocks are put together with the formula that
meta.json states,
  <i J|H|j J> = delta(S L, S' L') h_ij + (-1/sqrt 3) (-1)^(S'+L+J) {J L S; 1 S' L'} w_ij
              + (1/sqrt 5) (-1)^(S'+L+J) {J L S; 2 S' L'} w2_ij,
and the element (j, i) of w and w2 is (-1)^((S_i - S_j) + (L_i - L_j)) times the element (i, j).
"""
import json
import re
import sys
from fractions import Fraction
from math import factorial, sqrt

import numpy as np

T, systems = sys.argv[1], sys.argv[2:]
HT = ("h_C", "h_OO", "h_MVD", "h_CONTACT")
WT = ("w_SO1", "w_SO2")
bad = []


def check(ok, what):
    if not ok:
        bad.append(what)


def tri(a, b, c):
    """a, b, c (twice the momenta) form a triangle"""
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


def delta2(a, b, c):
    return Fraction(factorial((a + b - c) // 2) * factorial((a - b + c) // 2) * factorial((-a + b + c) // 2),
                    factorial((a + b + c) // 2 + 1))


def sixj(a, b, c, d, e, f):
    """{a b c; d e f} of twice the momenta (Racah's sum)"""
    if not (tri(a, b, c) and tri(a, e, f) and tri(d, b, f) and tri(d, e, c)):
        return 0.0
    lo = max(a + b + c, a + e + f, d + b + f, d + e + c) // 2
    hi = min(a + b + d + e, b + c + e + f, a + c + d + f) // 2
    s = Fraction(0)
    for t in range(lo, hi + 1):
        s += Fraction((-1) ** t * factorial(t + 1),
                      factorial(t - (a + b + c) // 2) * factorial(t - (a + e + f) // 2) *
                      factorial(t - (d + b + f) // 2) * factorial(t - (d + e + c) // 2) *
                      factorial((a + b + d + e) // 2 - t) * factorial((b + c + e + f) // 2 - t) *
                      factorial((a + c + d + f) // 2 - t))
    return float(s) * sqrt(float(delta2(a, b, c) * delta2(a, e, f) * delta2(d, b, f) * delta2(d, e, c)))


def factor(k, tS, tL, tSp, tLp, tJ):
    """what a reduced element of rank k (1: w, 2: w2) between (S, L) and (S', L') gets in the J block"""
    ph = -1.0 if ((tSp + tL + tJ) // 2) % 2 else 1.0
    return (-1.0 if k == 1 else 1.0) * ph / sqrt(2 * k + 1) * sixj(tJ, tL, tS, 2 * k, tSp, tLp)


def text(name):
    return open(f"{T}/{name}").read()


def labels(prefix):
    """(2S, 2L) of every CSF"""
    return [(int(c[6]) - 1, 2 * int(c[7])) for c in
            (l.split("\t") for l in text(f"{prefix}.csf.tsv").splitlines()[1:])]


def matrix(prefix, term, lab):
    """the full matrix of a term from its elements i <= j; None when the term has no file"""
    try:
        i, j, v = (np.load(f"{T}/{prefix}.{term}.{k}.npy") for k in "ijv")
    except FileNotFoundError:
        return None
    check(np.all(i <= j) and np.all(v != 0.0) and np.all(np.diff(i * len(lab) + j) > 0),
          f"{prefix} {term}: elements not i <= j, nonzero, sorted")
    n = len(lab)
    M = np.zeros((n, n))
    M[i, j] = v
    if term.startswith("h_"):
        M[j, i] = v
    else:
        s = np.array([-1.0 if (((lab[a][0] - lab[b][0]) + (lab[a][1] - lab[b][1])) // 2) % 2 else 1.0
                      for a, b in zip(i, j)])
        off = i != j
        M[j[off], i[off]] = (s * v)[off]
    return M


def terms(prefix, lab):
    n = len(lab)
    m = {t: matrix(prefix, t, lab) for t in HT + WT + ("w_SS",)}
    z = np.zeros((n, n))
    return {t: (z if x is None else x) for t, x in m.items()}, {t for t, x in m.items() if x is not None}


def blocks(prefix):
    meta = json.load(open(f"{T}/{prefix}.meta.json"))
    return meta, {b["tag"]: dict(b, **{k: np.load(f"{T}/{prefix}.{b['tag']}.{k}.npy")
                                       for k in ("idx", "vec", "e")}) for b in meta["blocks"]}


def levels(name):
    return [float(m.group(1)) for m in re.finditer(r"^ *\d+/2 +\d+ +(-?\d+\.\d+) ", text(name), re.M)]


def ls_levels(name):
    return [float(m.group(1)) for m in re.finditer(r"^\d+\S+ +\d+ +(-?\d+\.\d+)$", text(name), re.M)]


def span_dev(e, v, er, vr):
    """levels within 1e-6 are compared as a span: 1 - the smallest singular value of the overlap"""
    d, a = 0.0, 0
    while a < len(e):
        b = a + 1
        while b < len(e) and e[b] - e[b - 1] < 1e-6:
            b += 1
        # the reference levels within 1e-6 of this group
        ref = [k for k in range(len(er)) if er[a] - 1e-6 < er[k] < er[b - 1] + 1e-6] if b <= len(er) else []
        if len(ref) >= b - a:
            sv = np.linalg.svd(v[a:b] @ vr[ref].T, compute_uv=False)
            d = max(d, 1.0 - sv.min())
        a = b
    return d


for name in systems:
    lab = labels(f"{name}.j")
    n = len(lab)
    M, have = terms(f"{name}.j", lab)
    check(have == set(HT + WT + ("w_SS",)), f"{name}: terms written {sorted(have)}")
    h = sum(M[t] for t in HT)
    w = sum(M[t] for t in WT)
    w2 = M["w_SS"]

    # T1: every J block from the matrices
    meta, B = blocks(f"{name}.j")
    de = dv = 0.0
    nlv = 0
    for tag, b in B.items():
        tJ, idx = b["twoJ"], b["idx"]
        H = np.empty((len(idx), len(idx)))
        for p, i in enumerate(idx):
            for q, j in enumerate(idx):
                (tS, tL), (tSp, tLp) = lab[i], lab[j]
                x = h[i, j] if (tS, tL) == (tSp, tLp) else 0.0
                if w[i, j] != 0.0:
                    x += factor(1, tS, tL, tSp, tLp, tJ) * w[i, j]
                if w2[i, j] != 0.0:
                    x += factor(2, tS, tL, tSp, tLp, tJ) * w2[i, j]
                H[p, q] = x
        check(abs(H - H.T).max() < 1e-12, f"{name} {tag}: the block put together is not symmetric")
        e, c = np.linalg.eigh(H)
        k = len(b["e"])
        de = max(de, abs(e[:k] - b["e"]).max())
        dv = max(dv, span_dev(b["e"], b["vec"], e, c.T))
        nlv += k
    check(de < 1e-10, f"{name} T1: levels differ by {de:.1e}")
    check(dv < 1e-8, f"{name} T1: vectors differ by {dv:.1e}")
    print(f"   {name}: {n} CSFs, {len(B)} J blocks, {nlv} levels")
    print(f"      T1 J blocks from the matrices of dumpmat: levels {de:.1e} hartree, vectors {dv:.1e}")

    # T3 (second half): the terms kept apart against the terms in the integrals; nog
    ej, ep, en = levels(f"{name}.j.out"), levels(f"{name}.plain.out"), levels(f"{name}.nog.out")
    d3 = max(abs(a - b) for a, b in zip(ej, ep))
    check(len(ej) == len(ep) == nlv and d3 < 1e-10,
          f"{name} T3: kept apart against plain {d3:.1e} ({len(ej)}, {len(ep)}, {nlv} levels)")
    check(en == ep, f"{name}: nog changes the levels")

    # ls: the levels of h_C, the same matrices, T2
    labl = labels(f"{name}.ls")
    check(text(f"{name}.ls.csf.tsv") == text(f"{name}.j.csf.tsv"), f"{name}: csf.tsv of ls differs")
    Ml, havel = terms(f"{name}.ls", labl)
    dm = max(abs(Ml[t] - M[t]).max() for t in M)
    check(havel == have and dm == 0.0, f"{name}: matrices of the ls run differ by {dm:.1e}")
    metal, L = blocks(f"{name}.ls")
    dl = max(abs(a - b) for a, b in zip(ls_levels(f"{name}.ls.out"), ls_levels(f"{name}.ls0.out")))
    check(dl < 1e-12, f"{name}: the LS levels change with the terms by {dl:.1e}")
    # the LS levels are those of h_C
    dc = 0.0
    for tag, b in L.items():
        hc = M["h_C"][np.ix_(b["idx"], b["idx"])]
        dc = max(dc, abs(b["vec"] @ hc @ b["vec"].T - np.diag(b["e"])).max())
    check(dc < 1e-10, f"{name}: the LS vectors do not diagonalize h_C ({dc:.1e})")
    d2, npj = 0.0, 0
    P = {}
    for it in metal["proj"]:
        a, b = L[it["bra"]], L[it["ket"]]
        got = np.load(f"{T}/{name}.ls.{it['file']}")
        want = a["vec"] @ M[it["term"]][np.ix_(a["idx"], b["idx"])] @ b["vec"].T
        check(got.shape == tuple(it["shape"]) == want.shape, f"{name} proj {it['file']}: shape")
        d2 = max(d2, abs(got - want).max() / max(1e-300, abs(M[it["term"]]).max()))
        P[(it["term"], it["bra"], it["ket"])] = got
        npj += 1
    want_n = sum(len(L) if t.startswith("h_") else
                 sum(tri(a["S2p1"] - 1, 2 * k, b["S2p1"] - 1) and tri(2 * a["L"], 2 * k, 2 * b["L"])
                     for a in L.values() for b in L.values())
                 for t, k in (("h_OO", 0), ("h_MVD", 0), ("h_CONTACT", 0), ("w_SO1", 1), ("w_SO2", 1),
                              ("w_SS", 2)))
    check(npj == want_n, f"{name} T2: {npj} files of proj, {want_n} expected")
    check(d2 < 1e-12, f"{name} T2: proj differs by {d2:.1e}")
    print(f"      T2 {npj} files of proj against c^T M c': {d2:.1e} (relative); LS levels of h_C "
          f"({dc:.1e}), unchanged by the terms ({dl:.1e})")

    # with all LS levels: the J blocks from the LS energies and the files of proj alone
    if sum(b["nroot"] for b in L.values()) == n:
        dj = 0.0
        for tag, b in B.items():
            tJ = b["twoJ"]
            bl = [x for x in L.values() if tri(x["S2p1"] - 1, 2 * x["L"], tJ)]
            off = np.cumsum([0] + [x["nroot"] for x in bl])
            H = np.zeros((off[-1], off[-1]))
            for p, x in enumerate(bl):
                for q, y in enumerate(bl):
                    blk = np.zeros((x["nroot"], y["nroot"]))
                    if p == q:
                        blk += np.diag(x["e"])
                        for t in ("h_OO", "h_MVD", "h_CONTACT"):
                            blk += P[(t, x["tag"], y["tag"])]
                    for t, k in (("w_SO1", 1), ("w_SO2", 1), ("w_SS", 2)):
                        if (t, x["tag"], y["tag"]) in P:
                            blk += factor(k, x["S2p1"] - 1, 2 * x["L"], y["S2p1"] - 1, 2 * y["L"], tJ) * \
                                P[(t, x["tag"], y["tag"])]
                    H[off[p]:off[p + 1], off[q]:off[q + 1]] = blk
            check(abs(H - H.T).max() < 1e-12, f"{name} {tag}: the block from proj is not symmetric")
            dj = max(dj, abs(np.linalg.eigvalsh(H)[:len(b["e"])] - b["e"]).max())
        check(dj < 1e-10, f"{name} T2: J levels from the files of proj differ by {dj:.1e}")
        print(f"      T2 J blocks from the LS energies and the files of proj alone: levels {dj:.1e} hartree")
    print(f"      T3 terms kept apart against the terms in the integrals: levels {d3:.1e} hartree")

# T3 (first half), small: each term alone
lab = labels("small.j")
M, _ = terms("small.j", lab)
d = 0.0
for run, ts in (("nososo2", ["w_SO2"]), ("nososs", ["w_SS"]), ("nosooo", ["h_OO"]), ("nosomvd", ["h_MVD"]),
                ("nosocontact", ["h_CONTACT"]), ("so1", ["w_SO1"])):
    M1, have = terms(f"small.one.{run}", lab)
    check(have == set(ts + ["h_C"]), f"small, run {run}: terms written {sorted(have)}")
    for t in ts + ["h_C"]:
        d = max(d, abs(M1[t] - M[t]).max())
check(d == 0.0, f"T3: a term alone differs from the run with all terms by {d:.1e}")
print(f"   T3 each term alone (6 runs) against the run with all terms: {d:.1e}")

# dense against sparse, medium
if "medium" in systems:
    lab = labels("medium.j")
    Md, _ = terms("medium.j", lab)
    Ms, _ = terms("medium.sp", lab)
    d = max(abs(Md[t] - Ms[t]).max() for t in Md)
    de = max(abs(a - b) for a, b in zip(levels("medium.j.out"), levels("medium.sp.out")))
    check(d == 0.0 and de < 1e-10, f"medium: dense against sparse {d:.1e}, levels {de:.1e}")
    print(f"   medium, dense against sparse (Davidson): matrices {d:.1e}, levels {de:.1e} hartree")
    same = text("medium.mem.out") == text("medium.disk.out")
    check(same and len(levels("medium.disk.out")) == 42, "medium: AC_LSJ_DISK changes the output")
    print(f"   medium, w and w2 in files (AC_LSJ_DISK): output {'identical' if same else 'DIFFERENT'}")

if bad:
    print("terms.sh FAILED:")
    for b in bad[:20]:
        print("   ", b)
    sys.exit(1)
print("terms.sh: passed")
