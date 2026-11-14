# Basis sets

## `be_sapporo_tzp2012d.bas`

- **What**: Be, Sapporo-TZP-2012-diffuse, orbital basis. 7 s, 5 p and 3 d
  contracted Gaussians (one 4-term s and one 2-term d contraction).
- **Source**: Basis Set Exchange v0.12, https://www.basissetexchange.org,
  version 1 (data from T. Noro), downloaded in GAMESS-US format.
- **Conversion**: converted by hand to ATOMCI's native GTO format, the one
  read by `GINPUT` in `src/cinewx.f` and by `scripts/atinputs/basis.py`.
  Each function is an l letter, the number of primitives, a scale factor
  (`1.0`), then all exponents, then all contraction coefficients. The list
  ends with `))`. Exponents and coefficients were copied without change.
  Basis Set Exchange gives coefficients for normalized primitives, and
  ATOMCI normalizes each primitive (`CE*alpha**((2l+3)/4)`,
  `src/aiigtf.f`) and then the contracted function, so the two
  conventions agree.
- **Verification**: every exponent and coefficient matches AIMAIN's echo of
  the basis (`INPUT BASORB` table in the run output) to the printed 6
  decimals. The atinputs reader gives NSYM=3, NBF=7,5,3.

## `b_sapporo_tzp2012d.bas`

- **What**: B, Sapporo-TZP-2012-diffuse, orbital basis. 7 s, 5 p, 3 d and
  2 f contracted Gaussians, 17 functions in all. The contractions are one
  4-term s, one 2-term s (with a negative coefficient), one 3-term p, one
  2-term d and one 2-term f.
- **Source**: Basis Set Exchange v0.12, version 1 (data from T. Noro),
  GAMESS-US format as supplied by the user.
- **Conversion**: the same hand conversion to ATOMCI's native GTO format as
  for Be. Values copied unchanged.
- **Verification**:
  - The atinputs reader gives NSYM=4, NBF=7,5,3,2.
  - An AI-only ATOMCI run (`runs/p6/aicheck`, the installed `atomci`) read
    NWT=17, LMAXX=3, Z=5 with return code 0.
  - All 25 exponent/coefficient pairs in AIMAIN's `INPUT BASORB` echo match
    the input to the printed 6 decimals. The one apparent difference,
    0.9269875 echoed as 0.926987, is a binary rounding tie, not a
    transcription error.

## `n_sapporo_tzp2012d.bas`

- **What**: N, Sapporo-TZP-2012-diffuse, orbital basis. 7 s, 5 p, 3 d and
  2 f contracted Gaussians, 17 functions in all. Contractions: one 4-term
  s, one 2-term s (negative coefficient), one 3-term p, one 2-term d, one
  2-term f, as for B.
- **Source**: Basis Set Exchange v0.12, version 1 (data from T. Noro),
  GAMESS-US format as supplied by the user.
- **Conversion**: same native GTO format as Be and B, values copied
  unchanged. This time the conversion was done by a small parser of the
  GAMESS-US block rather than by hand.
- **Verification**:
  - The atinputs reader gives NSYM=4, NBF=7,5,3,2.
  - An AI-only ATOMCI run (`runs/p7/n_aicheck`, Z=7, the installed `atomci`)
    read NWT=17, LMAXX=3, Z=7 with return code 0.
  - All 25 exponent/coefficient pairs in AIMAIN's `INPUT BASORB` echo
    match the input rounded to 6 decimals (compared as multisets, no
    differences).

## `v.bas`

- **What**: V, Sapporo-TZP-2012-diffuse, orbital basis. 9 s, 7 p, 5 d,
  3 f and 2 g contracted Gaussians, 26 functions in all (Phase 9).
  Contractions: s 7, 4 and 2 terms; p 7 and 2; d 6 and 2; f 2; g 2. Two
  coefficients are negative (s 4-term, s 2-term, p 2-term).
- **Source**: Basis Set Exchange v0.12, version 1 (data from T. Noro),
  GAMESS-US format as supplied by the user; kept verbatim in
  `src/v_sapporo_tzp2012d.gamess`.
- **Conversion**: the same native GTO format as the other files, values
  copied as strings without change. Exponent and coefficient lists are
  wrapped at 4 values per line, so every data line is under 72 columns.
- **Verification**:
  - `atomci_heff.gto.read_bas` gives the 26 functions with every exponent
    and coefficient equal to the source.
  - The atinputs reader gives NSYM=5, NBF=9,7,5,3,2.
  - An AI-only ATOMCI run (the installed `atomci`, scratch) read NWT=26,
    LMAXX=4, Z=23 with return code 0; all 51 exponent/coefficient pairs in
    AIMAIN's `INPUT BASORB` echo match the source to the printed 6 decimals.

## `v_qzp.bas`

- **What**: V, Sapporo-QZP-2012-diffuse, orbital basis. 10 s, 8 p, 6 d,
  4 f, 3 g and 2 h contracted Gaussians, 33 functions in all (Phase 9b).
- **Source**: Basis Set Exchange v0.12, version 1 (data from T. Noro),
  GAMESS-US format as supplied by the user; kept verbatim in
  `src/v_sapporo_qzp2012d.gamess`.
- **Conversion**: `scripts/convert_bas.py` (the same conversion as `v.bas`,
  which it reproduces byte for byte), values copied as strings, lists
  wrapped at 4 values per line.
- **Verification**:
  - `atomci_heff.gto.read_bas` gives the 33 functions with every exponent
    and coefficient equal to the source.
  - The atinputs reader gives NSYM=6, NBF=10,8,6,4,3,2.
  - An AI-only ATOMCI run (scratch) read NWT=33, LMAXX=5, Z=23 with return
    code 0; all 54 exponent/coefficient pairs in AIMAIN's echo match the
    source to the printed 6 decimals.

## `ti.bas`, `cr.bas`, `mn.bas`, `fe.bas`

- **What**: Ti, Cr, Mn, Fe, Sapporo-QZP-2012-diffuse, orbital basis. Same
  layout as `v_qzp.bas`: 10 s, 8 p, 6 d, 4 f, 3 g and 2 h contracted
  Gaussians (19/14/12/4/3/2 primitives), 33 functions, identical contraction
  pattern.
- **Source**: Basis Set Exchange v0.12, version 1 (data from T. Noro),
  GAMESS-US format as supplied by the user (2026-10-15); kept verbatim in
  `src/<el>_sapporo_qzp2012d.gamess`.
- **Conversion**: `scripts/convert_bas.py` (it reproduces `v_qzp.bas` and
  `v.bas` byte for byte from their sources).
- **Verification**: `results/bases/check_report.md`:
  - `read_bas` equals the source for every value;
  - exponents positive and ordered; contractions normalized to 3e-7;
  - overlap conditioning within 2x of V;
  - single-CSF HF energies of the neutral ground terms with unchanged
    ATOMCI are 21.6-31.1 mEh above numerical HF, 0.9-1.3x V's 23.9 mEh;
  - V's error is the basis family's (about 20 mEh primitives + 4 mEh
    contraction; an even-tempered set reaches the HF limit to 1 mEh).

## `cr.without-diffuse.bas`

- **What**: Cr, Sapporo-QZP-2012 without the diffuse functions, 9s7p5d3f2g1h (27 contracted functions). Phase 15s
  addendum.
- **Source**: DERIVED, not downloaded. It is `cr.bas` (Sapporo-QZP-2012-diffuse) minus the most diffuse
  single-primitive function of each l:

  | l | s | p | d | f | g | h |
  |---|---|---|---|---|---|---|
  | removed exponent | 0.0117337 | 0.027493 | 0.0420204 | 0.1343523 | 0.2000113 | 0.737899 |

  - All other exponents and coefficients are unchanged.
  - The rule reproduces the downloaded V plain set exactly: applied to `src/v_sapporo_qzp2012d.gamess`, it gives
    `src/v_sapporo_qzp2012.gamess` (27 of 27 functions identical).
  - The source is `src/cr_sapporo_qzp2012_derived.gamess`, converted with `scripts/convert_bas.py`. The converter
    reproduces `v.without-diffuse.bas` byte for byte.
  - To confirm the set, replace the source with the Basis Set Exchange Sapporo-QZP-2012 file for Cr. The download was
    not done from this session.
- **Verification**: ATOMCI single-CSF HF of Cr³⁺ 3d³ ⁴F (`runs/hfb_cr3p`), rc 0:

  | | E (Eh) |
  |---|---|
  | this basis | −1041.450546 |
  | `cr.bas` | −1041.450665 |
  | difference | +0.12 mEh |

  The be.NO header written by ATOMCI has the per-l pattern 9,7,5,3,2,1.

## `co.without-diffuse.bas` (Phase 18)

- **What**: Co, Sapporo-QZP-2012 without diffuse functions, orbital basis, 27 functions (9 s, 7 p, 5 d, 3 f, 2 g, 1 h).
- **Source**: Basis Set Exchange v0.8.13 copy shipped with GaussianBasis.jl,
  `lib/sapporo-qzp-2012.gbs` of the Julia package GaussianBasis.jl, version 1 (data from T. Noro). The Co block and the
  file header are kept in `src/co_sapporo_qzp2012.gbs`, so runs do not depend on the Julia package.
- **Conversion**: `scripts/phase18_basis.py write Co co.without-diffuse.bas`; the round trip reads back identical values.
- **Verification**: the same reader finds 27 of 27 functions identical (exponents and coefficients) between that `.gbs`
  and our `fe.without-diffuse.bas` and `cr.without-diffuse.bas` (`scripts/phase18_basis.py check`), so the copy agrees
  with the sets already in use. The function order within an l differs for Fe and Cr.
- **Derived sets**: `co_qzp2012_add_i2.2137.bas`, `co_qzp2012_add_i2.2137_i1.bas` (i functions, phase17q9i recipe),
  `co_qzp2012_open_spd.bas` (opened s, p, d for DK3, phase17q4r recipe), `co_qzp2012_add_d59.7_f77.bas` (crt2 tight pair
  scaled to the Co3+ 2p radius, phase17q10 recipe).

## `co.bas` (Phase 18, 10 Nov)

- **What**: Co, Sapporo-QZP-2012-diffuse, orbital basis, 33 functions (10s8p6d4f3g2h), the Ti..Fe layout.
- **Source**: Basis Set Exchange v0.12, version 1 (data from T. Noro), GAMESS-US format as supplied by the user
  (2026-11-10); kept verbatim in `src/co_sapporo_qzp2012d.gamess`. (The GaussianBasis.jl copy
  `sapporo-qzp-2012-diffuse.gbs` has no diffuse functions: its 27 functions equal the plain set for Ti..Co.)
- **Conversion**: `scripts/convert_bas.py` (it reproduces `fe.bas` byte for byte from its source).
- **Verification** (`results/bases/check23.txt`, `check4.txt`):
  - `read_bas` equals the source for every value; contraction pattern identical to V;
  - the 27 functions of `co.without-diffuse.bas` are a subset (exponents and coefficients identical); the 6 extra are
    the most diffuse single function per l (s 0.0136187, p 0.0392767, d 0.0517743, f 0.2262473, g 0.325065,
    h 1.01566), continuing the Ti..Fe exponents;
  - exponents positive and ordered, contractions normalized to 9e-8, overlap conditioning 0.6-1.5x of V, no flag;
  - single-CSF HF of Co 3d7 4s2 4F 33.94 mEh above HF86 (-1381.414553, Co added to `scripts/numhf.py`), 1.42x V,
    continuing Ti..Fe (21.6..31.1 mEh); Co3+ 3d6 5D 0.98 mEh below `co.without-diffuse.bas` (Fe2+ own orbitals in
    `fe.bas` vs plain: 0.85 mEh).

## `fe_qzp2012_open_d.bas`, `co_qzp2012_open_d.bas` (Phase 18, 10 Nov B)

- **What**: the plain Sapporo-QZP-2012 sets (`fe.without-diffuse.bas`, `co.without-diffuse.bas`) with only the d
  contractions opened into their 11 distinct primitives (9s7p11d3f2g1h). s, p, f, g, h unchanged; f is all single
  primitives in Sapporo-QZP-2012, so opening f changes nothing. No new exponents.
- **Conversion**: `scripts/phase18.py dbases` (the phase17q4r open_spd recipe restricted to d); non-d functions
  checked identical to the plain set.
