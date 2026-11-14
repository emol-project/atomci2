"""uncontract.py -- write a fully uncontracted copy of a Dalton library basis.

    python3 uncontract.py SRC DST Z [Z ...]

Each distinct exponent of each l becomes its own primitive (Dalton library
format, "nprim nprim 0" blocks with unit coefficients).
"""
import sys

LNAME = "SPDFGHIK"


def read(path):
    """{Z: [[exponents of l=0], [l=1], ...]} from a Dalton library file."""
    out, z, blocks = {}, None, None
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].split()
        if lines[i].startswith("a ") and len(s) == 2:
            z = int(s[1])
            blocks = out.setdefault(z, [])
            i += 1
            continue
        if z is not None and s and not lines[i].startswith("$") and len(s) == 3 and all(
                t.lstrip("-").isdigit() for t in s):
            nprim, ncon = int(s[0]), int(s[1])
            exps, i = [], i + 1
            for _ in range(nprim):
                vals = lines[i].split()
                exps.append(float(vals[0]))
                need, i = ncon + 1 - len(vals), i + 1
                while need > 0:
                    need -= len(lines[i].split())
                    i += 1
            blocks.append(exps)
            continue
        i += 1
    return out


def write(dst, basis, zs):
    with open(dst, "w") as f:
        f.write("$ uncontracted\n")
        for z in zs:
            f.write(f"a {z}\n")
            for l, exps in enumerate(basis[z]):
                uniq = sorted(set(exps), reverse=True)
                n = len(uniq)
                f.write(f"$ {LNAME[l]}-TYPE FUNCTIONS\n{n:5d}{n:5d}    0\n")
                for k, e in enumerate(uniq):
                    c = ["1.0" if j == k else "0.0" for j in range(n)]
                    f.write(f"{e:18.8f} " + " ".join(c[:6]) + "\n")
                    for j in range(6, n, 7):
                        f.write("    " + " ".join(c[j:j + 7]) + "\n")


if __name__ == "__main__":
    src, dst, zs = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
    b = read(src)
    write(dst, b, zs)
    for z in zs:
        print(z, [len(set(e)) for e in b[z]])
