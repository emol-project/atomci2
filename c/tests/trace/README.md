# M0 oracle recording (C port, stage 1)

Records the Fortran angular engine of ATOMCI while it runs the `boron` and `f`
samples, so the C rewrite can be checked call by call (layers L1-L3; technical note,
Appendix C). The Fortran sources are not modified: a scratch
copy of `src/` is built normally, and `atomci` is then relinked with GNU ld
`--wrap` around six routines.

| Wrapped symbol | Why it is wrapped |
|---|---|
| `xwig6j_`, `xwig9x_` | 6j and 9j values. `WIG6J`/`WIG9X` in `wt9x.o` are thin wrappers; `--wrap` only sees calls between objects, so the bodies are wrapped. |
| `xcfp_` | CFP values. The body behind `CFP` in `wt9x.o`. |
| `shms_` | One shell-factored matrix element per call (the L2 oracle). |
| `smat_`, `isqt_` | One block per SMAT call: its inputs, every ISQT term emitted inside it, and its outputs (the L3 oracle). |

The record format is documented at the top of `trace_wrap.c`.

## Running

On Debian 13 (needs `gfortran`, `liblapack-dev`, `libblas-dev`):

```bash
bash c/tests/trace/build_trace.sh
bash c/tests/trace/run_trace.sh
```

Everything goes under `$AC_WORK/m0` (`AC_WORK` defaults to `$HOME/atomci_work`), outside the
repository:

- `build/`: the scratch build, with `atomci.plain` (normal) and `atomci` (traced).
- `run/{plain,trace}-{boron,f}/`: one clean work directory per run.
  - The `fort.*` files are kept. They include the NFTE tapes, which L4/L5 need.
  - `trace-*/trace/*.txt` is the recorded oracle.

## Result of the first recording (2026-10-26)

| Check | boron | f |
|---|---|---|
| Energy, plain and traced | `ET = -24.6461953967` (golden) | first-pass `SD-CI ENERGY = -99.641223` (golden) |
| Plain vs traced output, time lines dropped | 0 differing lines | 0 differing lines |
| stderr | empty | empty |
| SMAT calls (S/E blocks) | 7507 | 7735 |
| ISQT terms (T lines); outside SMAT (X) | 10022; 0 | 10129; 0 |
| distinct SHMS records | 15347 | 16819 |
| distinct 6j / 9j / XCFP records | 2490 / 773 / 107 | 1249 / 972 / 84 |

The total size of `$AC_WORK` is about 72 MB.

**Operator ranks seen in SHMS.** The total rank is always scalar: (LST, LRT) =
(1, 1) in every record. Spin-triplet pairs (LSI = LSF = 3) appear only from
CONVT. This matches the technical note, Sec. 3.4.

**f energy lines.** `f.stdin` runs three passes. The golden value is the
first pass's SD-CI. The last `SD-CI ENERGY` line in `f.output` (-99.641037)
comes from the natural-orbital pass and is not the golden value, so
`run_trace.sh` reports the first one.
