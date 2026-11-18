"""dump_check.py -- the checks of dump.sh (see there) on the files in the directory argv[1]."""
import hashlib
import json
import re
import sys

import numpy as np

T = sys.argv[1]
bad = []


def check(ok, what):
    if not ok:
        bad.append(what)


def text(name):
    return open(f"{T}/{name}").read()


def body(name):
    """the output without the lines that info adds"""
    return [l for l in text(name).splitlines()
            if not re.match(r"ATOMCI2 |\d+ configurations, \d+ CSFs:|CSF list sha256 ", l)]


def levels(name):
    """the printed J levels: (2J, number) -> energy"""
    return {(int(m.group(1)), int(m.group(2))): float(m.group(3))
            for m in re.finditer(r"^ *(\d+)/2 +(\d+) +(-?\d+\.\d+) ", text(name), re.M)}


def ls_levels(name):
    return [float(m.group(1)) for m in re.finditer(r"^\d+\S+ +\d+ +(-?\d+\.\d+)$", text(name), re.M)]


def load(prefix):
    meta = json.load(open(f"{T}/{prefix}.meta.json"))
    csf = [l.split("\t") for l in text(f"{prefix}.csf.tsv").splitlines()[1:]]
    blocks = {}
    for b in meta["blocks"]:
        blocks[b["tag"]] = dict(b, **{k: np.load(f"{T}/{prefix}.{b['tag']}.{k}.npy")
                                      for k in ("idx", "vec", "e", "res")})
    return meta, csf, blocks


def check_files(prefix, tol=1e-10):
    meta, csf, blocks = load(prefix)
    check(len(csf) == meta["n_csf"], f"{prefix}: csf.tsv has {len(csf)} lines")
    check([int(c[0]) for c in csf] == list(range(len(csf))), f"{prefix}: csf.tsv numbering")
    h = hashlib.sha256("".join(c[8] + "\n" for c in csf).encode()).hexdigest()
    check(h == meta["csf_sha256"], f"{prefix}: sha256 of the keys")
    cfg = [int(c[1]) for c in csf]
    check(cfg == sorted(cfg) and cfg[0] == 0 and cfg[-1] == meta["n_configurations"] - 1,
          f"{prefix}: configuration numbers")
    for c in csf:
        occ = dict(t.split("*") for t in c[2].split())
        sh = {x.split(",")[0]: x.split(",")[1] for x in c[4].split(";")}
        check(all(sh.get(k, "0") == v for k, v in occ.items()), f"{prefix}: CSF {c[0]} configuration")
        check(c[5].split(";")[-1] == f"{c[6]},{c[7]}", f"{prefix}: CSF {c[0]} total label")
    rmax = 0.0
    for tag, b in blocks.items():
        idx, vec, e, res = b["idx"], b["vec"], b["e"], b["res"]
        check(idx.dtype == np.int64 and vec.dtype == np.float64, f"{prefix} {tag}: types")
        check(vec.shape == (b["nroot"], b["dim"]) and idx.shape == (b["dim"],) and
              e.shape == res.shape == (b["nroot"],), f"{prefix} {tag}: shapes")
        if "twoJ" in b:
            want = [i for i, c in enumerate(csf)
                    if abs(int(c[6]) - 1 - 2 * int(c[7])) <= b["twoJ"] <= int(c[6]) - 1 + 2 * int(c[7])]
        else:
            want = [i for i, c in enumerate(csf) if (int(c[6]), int(c[7])) == (b["S2p1"], b["L"])]
        check(list(idx) == want, f"{prefix} {tag}: idx")
        check(abs(vec @ vec.T - np.eye(len(e))).max() < 1e-10, f"{prefix} {tag}: orthonormality")
        check(np.all(np.diff(e) >= -1e-12), f"{prefix} {tag}: order of the energies")
        rmax = max(rmax, res.max())
    check(rmax < tol, f"{prefix}: residual {rmax:.1e}")
    return meta, csf, blocks, rmax


def same_levels(pa, pb, etol, vtol):
    _, _, A, _ = pa
    _, _, B, _ = pb
    de = dv = 0.0
    check(A.keys() == B.keys(), "blocks differ")
    for tag in A:
        a, b = A[tag], B[tag]
        de = max(de, abs(a["e"] - b["e"]).max())
        # up to the sign; (near) degenerate levels are compared as a subspace
        ov = a["vec"] @ b["vec"].T
        dv = max(dv, abs(abs(np.linalg.svd(ov, compute_uv=False)) - 1).max())
    check(de < etol, f"energies differ by {de:.1e}")
    check(dv < vtol, f"vectors differ by {dv:.1e}")
    return de, dv


# 1. nothing else changes
check(body("s1.out") == body("s0.out"), "dump= changes the output")
check(body("s1i.out") == body("s0.out"), "info changes the output")
check(len(text("s1.out").splitlines()) == len(text("s0.out").splitlines()) + 3, "info lines")
l0, l2 = levels("s0.out"), levels("s2.out")
check(sorted(k for k in l2 if k[0] == 1) == [(1, 1), (1, 2), (1, 3)] and not any(k[0] == 5 for k in l2)
      and all(abs(l2[k] - l0[k]) < 1e-13 for k in l2 if k in l0) and len(l2) == len(l0) - 2 + 1,
      "nlevJ=")
print(f"   1. info, dump= and nlevJ= leave the rest of the output as it was ({len(l0)} levels)")

# 2, 3. the files
s = {p: check_files(p) for p in ("s1", "s2", "s3", "s4", "s5", "s6")}
m1, csf1, b1, _ = s["s1"]
for (tj, num), e in l0.items():
    check(abs(b1[f"J{tj}"]["e"][num - 1] - e) < 1e-13, f"s1 J{tj}: e.npy against the output")
check(len({s[p][0]["csf_sha256"] for p in s}) == 1, "sha256 differs between the sets of terms")
check(all(text(f"{p}.csf.tsv") == text("s1.csf.tsv") for p in s), "csf.tsv differs")
check(s["s3"][0]["terms"] == dict(coulomb=True, so1=True, so2=True, ss=True, oo=True, mvd=False,
                                  contact=False), "terms of meta.json")
check(set(s["s6"][2]) == {"S2L0", "S2L1", "S2L2", "S2L3", "S4L1", "S4L2", "S4L3"} and
      s["s6"][2]["S2L1"]["nroot"] == 3 and s["s6"][2]["S2L2"]["nroot"] == 2, "nlevSL=")
print(f"   2. files of {len(s)} runs: {m1['n_csf']} CSFs, blocks {' '.join(b1)}; residuals below "
      f"{max(x[3] for x in s.values()):.1e}")
print(f"   3. sha256 of the CSF list {m1['csf_sha256'][:16]}... for Coulomb + SO1, noso, so2 ss oo, ls")

# 4. LS levels against the J levels without spin-orbit
els = np.array(sorted(ls_levels("s5.out")))
d = max(abs(els - e).min() for e in levels("s4.out").values())
nls = sum(b["nroot"] for b in s["s5"][2].values())
check(nls == m1["n_csf"] and len(els) == nls, f"ls: {nls} levels")
check(d < 1e-10, f"ls against noso: {d:.1e}")
print(f"   4. ls: {nls} LS levels in {len(s['s5'][2])} blocks; every J level of noso is one of them "
      f"({d:.1e} hartree)")

# 5. the medium list
big = {p: check_files(p, 1e-6 if p == "b4" else 1e-10) for p in ("b1", "b2", "b3", "b4", "b5", "b6")}
mb = big["b1"][0]
check(mb["n_configurations"] == 128, f"medium list: {mb['n_configurations']} configurations")
check(big["b4"][0]["davidson_tolerance"] == 1e-6 and big["b3"][0]["davidson_tolerance"] == 1e-11,
      "tolerance in meta.json")
check(all(b["solver"] == "davidson" for b in big["b3"][2].values() if b["dim"] > 400),
      "the large blocks did not go to Davidson")
check(any(b["solver"] == "davidson" for b in big["b6"][2].values()), "ls: no block went to Davidson")
d12 = same_levels(big["b1"], big["b2"], 1e-10, 1e-8)
d13 = same_levels(big["b1"], big["b3"], 1e-10, 1e-8)
d56 = same_levels(big["b5"], big["b6"], 1e-10, 1e-8)
de4 = max(abs(big["b4"][2][t]["e"] - big["b3"][2][t]["e"]).max() for t in big["b3"][2])
check(de4 < 1e-9 and big["b4"][3] > 1e-11, f"tol=1e-6: energies differ by {de4:.1e}")
it3 = sum(b["iterations"] for b in big["b3"][2].values())
it4 = sum(b["iterations"] for b in big["b4"][2].values())
check(it4 < it3, "tol=1e-6 does not take fewer iterations")
# parts that H does not couple are solved one by one: the mixed list (two parities) and noso (the
# total (S, L)) give the levels of dsyev; b8, b11, b12 with Davidson inside the parts
l9 = levels("b9.out")
dm = 0.0
for f in ("b7.out", "b8.out"):
    lf = levels(f)
    check(lf.keys() == l9.keys(), f"mixed list, {f}: other levels than dsyev")
    dm = max(dm, max(abs(lf[k] - l9[k]) for k in l9))
l10 = levels("b10.out")
dn = 0.0
for f in ("b11.out", "b12.out"):
    lf = levels(f)
    check(lf.keys() == l10.keys(), f"noso, {f}: other levels than dsyev")
    dn = max(dn, max(abs(lf[k] - l10[k]) for k in l10))
check(dm < 1e-10 and dn < 1e-10, f"parts solved one by one differ from dsyev by {dm:.1e}, {dn:.1e}")
check("parts that H does not couple" in text("b8.err") and
      "parts that H does not couple" in text("b11.err") and
      "parts that H does not couple" in text("b12.err") and
      "parts that H does not couple" not in text("b3.err"), "the parts are not told")
print(f"   5. {mb['n_configurations']} configurations, {mb['n_csf']} CSFs; dense dsyev against dense "
      f"Davidson {d12[0]:.1e} hartree, {d12[1]:.1e} in the vectors; against sparse Davidson "
      f"{d13[0]:.1e}, {d13[1]:.1e}; ls dense against sparse {d56[0]:.1e}, {d56[1]:.1e}")
print(f"      tol=1e-6: residual {big['b4'][3]:.1e}, {it4} iterations instead of {it3}, energies "
      f"within {de4:.1e}")
print(f"      parts that H does not couple, solved one by one: a list of both parities within "
      f"{dm:.1e} hartree of dsyev, noso within {dn:.1e}")

# 6. the large list
mc, csfc, bc, rc = check_files("c1")
check(mc["n_configurations"] == 558, f"large list: {mc['n_configurations']} configurations")
check(f"558 configurations, {mc['n_csf']} CSFs:" in text("c1.out"), "info line of the large list")
ec = min(b["e"][0] for b in bc.values())
check(ec < min(b["e"][0] for b in big["b5"][2].values()), "the large list is not lower in energy")
print(f"   6. {mc['n_configurations']} configurations, {mc['n_csf']} CSFs, {len(bc)} LS blocks; "
      f"lowest level {ec:.9f} hartree")

if bad:
    print("dump.sh FAILED:")
    for b in bad[:20]:
        print("   ", b)
    sys.exit(1)
print("dump.sh: passed")
