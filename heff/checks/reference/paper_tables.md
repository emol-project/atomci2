# Paper tables, six ions on one footing (Phase 18; scripts/phase18_tables.py)

T in mEh (Uylings O2', O2'', Tr O² = 1 in d³), f in cm⁻¹ (Tr f² = 1 in d⁴). Generated from committed result files only.

## 1. Final T1, T2 (mEh)

proj = projection of the corrected H_eff; (1b) repr. = the corrected H_eff fitted with the (1b) model on the ion's NIST J levels; measured = our (1b) fit of the NIST levels; Hansen = Hansen, Raassen, Uylings & Lister 1988 (× 1.83813; their errors are batch-dependent). d³ corrected values: Step 10 chain; Fe²⁺, Co³⁺: the same d⁶ chain (phase18_co.py; Fe²⁺ reproduces phase17_final 9.482 / 3.626).

| ion | T1 proj | T1 (1b) repr. | T1 measured (1b) | T1 Hansen | proj / (1b) | proj / Hansen | T2 proj | T2 (1b) repr. | T2 measured (1b) | T2 Hansen |
|---|---|---|---|---|---|---|---|---|---|---|
| V²⁺ | 8.658 ± 0.047 | 8.599 ± 0.065 | 8.217 ± 0.160 | 8.253 | 1.054 | 1.049 | 1.528 ± 0.056 | 1.465 ± 0.076 | 1.802 ± 0.198 | 1.930 |
| Cr³⁺ | 12.082 ± 0.050 | 11.967 ± 0.070 | 11.492 ± 0.221 | 11.562 | 1.051 | 1.045 | 1.484 ± 0.056 | 1.301 ± 0.081 | 1.341 ± 0.277 | 1.489 |
| Mn⁴⁺ | 14.324 ± 0.059 | 14.203 ± 0.078 | 13.652 ± 0.318 | 13.676 | 1.049 | 1.047 | 1.472 ± 0.055 | 1.283 ± 0.083 | 1.123 ± 0.402 | 1.305 |
| Fe⁵⁺ | 16.021 ± 0.057 | 15.895 ± 0.077 | 15.341 ± 0.429 | 15.348 | 1.044 | 1.044 | 1.464 ± 0.054 | 1.266 ± 0.083 | 0.973 ± 0.544 | 1.195 |
| Fe²⁺ | 9.482 ± 0.065 | 9.586 ± 0.056 | 9.305 ± 0.204 | 9.136 | 1.019 | 1.038 | 3.627 ± 0.054 | 4.307 ± 0.127 | 4.506 ± 0.316 | 4.154 |
| Co³⁺ | 11.894 ± 0.062 | 11.985 ± 0.060 | 11.412 ± 0.160 | 11.341 | 1.042 | 1.049 | 2.988 ± 0.053 | 3.331 ± 0.063 | 3.348 ± 0.248 | 3.198 |

## 2. Second order and Q1_rule against exact (ΔT1 %, ΔT2 mEh)

d³: Step 8 space (`besth_sall`); d⁶: `fam_d5f2g2h1`. Closure = C2; S2-diag = second order with diagonal Q denominators; Q1_rule = n = 3 + 3p→f singles + 3d→v singles + same-l 3d² doubles in Q1, balanced; + pf2 = Q1_rule with the pf2 class (3p hole + f + a second excitation) also in Q1.

| ion (space) | exact T1 / T2 | closure | S2-full | S2-diag | Q1_rule (Q1 fraction) | Q1_rule + pf2 (Q1 fraction) |
|---|---|---|---|---|---|---|
| V²⁺ (S8) | 8.695 / 1.786 | +14.66 % / +0.567 | -41.48 % / +2.771 | +6.45 % / +1.059 | -0.70 % / +0.120 (0.59 %) | +0.27 % / -0.059 (42.23 %) |
| Cr³⁺ (S8) | 12.103 / 1.714 | +13.96 % / -0.152 | +15.73 % / +1.230 | +9.24 % / +0.069 | -0.46 % / +0.178 (0.59 %) | +0.41 % / -0.010 (42.23 %) |
| Mn⁴⁺ (S8) | 14.317 / 1.656 | +11.68 % / -0.619 | +14.12 % / +0.932 | +8.59 % / -0.503 | -0.59 % / +0.234 (0.59 %) | +0.00 % / +0.041 (42.23 %) |
| Fe⁵⁺ (S8) | 15.989 / 1.592 | +9.89 % / -0.921 | +13.12 % / +0.790 | +7.69 % / -0.829 | -0.86 % / +0.293 (0.59 %) | -0.50 % / +0.090 (42.23 %) |
| Fe²⁺ (fam) | 9.853 / 3.786 | +23.79 % / -1.431 | +15.19 % / +2.721 | +11.62 % / -1.605 | -0.50 % / -0.140 (2.05 %) | – |
| Co³⁺ (fam) | 12.248 / 3.139 | +18.57 % / -1.601 | +14.46 % / +1.269 | +8.00 % / -1.804 | -0.87 % / -0.057 (2.05 %) | – |

## 3. Four-body fits on the NIST levels, Fe²⁺ (Fe III, 33 levels) and Co³⁺ (Co IV)

Fixed extras = our T3, T4, f1..f12 (proj); λ = one common scale of the fixed extras fitted with the (1b) parameters; p(λ = 0), p(λ = 1) from F tests. fam = largest-space projection; corrected = the d⁶ chain; without 11d–13d = the fam sub-space without the three diffuse virtual d (exact H_eff of that sub-space).

| ion | extras | mean error | factor | λ | p(λ=0) | p(λ=1) | T1 fit | T2 fit | mean error, T1 T2 also fixed (proj) |
|---|---|---|---|---|---|---|---|---|---|
| Fe²⁺ | none (free (1b)) | 42.33 | 1 | | | | 9.305 ± 0.204 | 4.506 ± 0.316 | |
| Fe²⁺ | corrected T3+T4+f | 14.71 | 2.88 | 1.179 ± 0.080 | 1.7e-13 | 0.035 | 9.239 ± 0.071 | 4.165 ± 0.110 | 22.49 (p 9e-06) |
| Fe²⁺ | corrected f only | 15.23 | 2.78 | 1.122 ± 0.084 | 1.4e-12 | 0.16 | 9.248 ± 0.073 | 4.227 ± 0.114 | 23.72 (p 6e-06) |
| Fe²⁺ | corrected T3+T4 only | 43.56 | 0.97 | -1.526 ± 1.669 | 0.37 | 0.14 | 9.296 ± 0.210 | 4.443 ± 0.325 | 47.72 (p 0.04) |
| Fe²⁺ | fam T3+T4+f | 13.85 | 3.06 | 1.165 ± 0.074 | 4.1e-14 | 0.036 | 9.211 ± 0.067 | 4.090 ± 0.104 | 21.34 (p 8e-06) |
| Fe²⁺ | fam f only | 14.94 | 2.83 | 1.115 ± 0.082 | 9.2e-13 | 0.17 | 9.243 ± 0.072 | 4.163 ± 0.112 | 22.54 (p 1e-05) |
| Fe²⁺ | fam T3+T4 only | 43.29 | 0.98 | -0.474 ± 1.326 | 0.72 | 0.28 | 9.273 ± 0.209 | 4.432 ± 0.323 | 47.55 (p 0.04) |
| Fe²⁺ | fam without 11d–13d | 44.48 | 0.95 | -0.415 ± 0.852 | 0.63 | 0.11 | 9.223 ± 0.214 | 4.346 ± 0.332 | 48.06 (p 0.06) |
| Co³⁺ | none (free (1b)) | 33.12 | 1 | | | | 11.412 ± 0.160 | 3.348 ± 0.248 | |
| Co³⁺ | corrected T3+T4+f | 18.60 | 1.78 | 1.106 ± 0.151 | 1.4e-07 | 0.49 | 11.334 ± 0.090 | 3.054 ± 0.139 | 28.78 (p 7e-06) |
| Co³⁺ | corrected f only | 19.12 | 1.73 | 1.023 ± 0.148 | 3.6e-07 | 0.88 | 11.326 ± 0.092 | 3.131 ± 0.143 | 29.63 (p 7e-06) |
| Co³⁺ | corrected T3+T4 only | 34.68 | 0.96 | -0.650 ± 0.989 | 0.52 | 0.11 | 11.420 ± 0.168 | 3.271 ± 0.259 | 39.28 (p 0.02) |
| Co³⁺ | fam T3+T4+f | 18.90 | 1.75 | 1.058 ± 0.149 | 2.5e-07 | 0.7 | 11.300 ± 0.091 | 2.969 ± 0.141 | 29.85 (p 4e-06) |
| Co³⁺ | fam f only | 19.46 | 1.70 | 0.990 ± 0.147 | 5.6e-07 | 0.95 | 11.313 ± 0.094 | 3.064 ± 0.146 | 29.98 (p 8e-06) |
| Co³⁺ | fam T3+T4 only | 34.66 | 0.96 | -0.430 ± 0.899 | 0.64 | 0.12 | 11.400 ± 0.168 | 3.254 ± 0.259 | 39.58 (p 0.01) |
| Co³⁺ | fam without 11d–13d | 35.34 | 0.94 | 0.003 ± 0.547 | 1 | 0.081 | 11.310 ± 0.171 | 3.159 ± 0.264 | 41.62 (p 0.006) |

## 4. f1..f12 (cm⁻¹, Tr f² = 1 in d⁴), corrected proj, with Hansen et al. 1988

Corrected = the common d⁶ chain (the same correction list for Fe²⁺ and Co³⁺, 2p item included; phase18_co.py). Hansen errors in parentheses (batch-dependent).

| f | Fe²⁺ fam | Fe²⁺ corrected | Hansen Fe III | Co³⁺ fam | Co³⁺ corrected | Hansen Co IV |
|---|---|---|---|---|---|---|
| f1 | +235 | +273 | +248 (33) | +156 | +193 | +103 (30) |
| f2 | -166 | -181 | +9 (10) | -84 | -106 | +6 (10) |
| f3 | +352 | +324 | +346 (10) | +278 | +248 | +175 (9) |
| f4 | -70 | -88 | -208 (10) | -31 | -50 | -150 (8) |
| f5 | +98 | +101 | +107 (7) | +29 | +39 | +47 (5) |
| f6 | +5 | +34 | +37 (7) | -14 | +14 | +33 (5) |
| f7 | -103 | -128 | -11 (8) | -82 | -100 | -13 (8) |
| f8 | +278 | +272 | +332 (8) | +211 | +205 | +261 (7) |
| f9 | -75 | -70 | -25 (7) | -50 | -48 | -36 (5) |
| f10 | +11 | +11 | -32 (7) | +8 | +8 | -30 (5) |
| f11 | +84 | +79 | +0 (8) | +74 | +68 | -5 (8) |
| f12 | +84 | +93 | -60 (9) | +54 | +66 | -16 (8) |

Against Hansen (whole vector): Fe²⁺ corrected cos 0.853, slope 0.838; Co³⁺ corrected cos 0.814, slope 0.924.

T3, T4 corrected (proj, mEh): Fe²⁺ -0.141, +0.129; Co³⁺ -0.135, +0.220.

Note: the paper's earlier Fe²⁺ corrected values (Phase 17 `fe_own`: fam + the Fe²⁺ RAS increments, no 2p item) differ from the common-list values by <= 3 cm⁻¹ in every f (cos vs Hansen 0.851; T3, T4 -0.147, +0.135).

## 5. Fe⁵⁺: Uylings (2021) Table 14.5, his normalization (cm⁻¹), Fe VI

| class | this work (closure, Step 8) | Uylings | difference |
|---|---|---|---|
| 3s->3d only | -11.812 | | |
| 2s->3d only | -0.261 | | |
| s->3d (3s + 2s) | -12.073 | -12.067 | -0.006 |
| 3d->s | +0.263 | +0.209 | +0.054 |
| 3d->d' 22 | -0.255 | -0.198 | -0.057 |
| 3d->d' 24 | +0.512 | +0.405 | +0.107 |
| 3d->d' 44 | -0.160 | -0.129 | -0.031 |
| 3d->g 22 | +2.624 | +2.391 | +0.233 |
| 3d->g 24 | +0.797 | +0.710 | +0.087 |
| 3d->g 44 | -1.268 | -1.107 | -0.161 |
| 3d->i 44 | +0.024 | +0.037 | -0.013 |
| total second order | -9.559 | -9.727 | +0.168 |

fitted (Uylings) -8.452; closure / fit: 1.131 (Uylings), 1.145 (Hansen); Uylings' own total / fitted 1.151

