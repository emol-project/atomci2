# The C tools of atomci2 and their checks. See README.md and tests/README.md.
#
#   make            build/libac.a and the tools (acatsp, acfs, ...); builds
#                   hexpr-atomci first if needed
#   make check      L0 to L4 against the Fortran (see tests/README.md for
#                   every check and what it needs)
#
# The checks link objects from a built Fortran tree (the M0 scratch build by
# default, c/tests/trace/build_trace.sh) and read its CFPTBL and the M0
# XCFP and SHMS recordings:
#   AC_WORK     work root                 (default $(HOME)/atomci_work)
#   FTN_BUILD   built src/ tree           (default $(AC_WORK)/m0/build)
#   M0_RUN      M0 run directories        (default $(AC_WORK)/m0/run)

CC      ?= cc
FC      := gfortran
PYTHON  ?= python3
WARN    := -Wall -Wextra -pedantic
# OpenMP parallelizes the Breit-Pauli matrix build (ac_lsj_prepare); OMPFLAGS=
# builds serially. The engine's caches are thread-local (_Thread_local).
OMPFLAGS ?= -fopenmp
CFLAGS  ?= -O2 -std=c11 $(WARN) -Werror $(OMPFLAGS)
# The angular engine (Wigner symbols, CFP, CSFs, operator trees, factorization,
# Hamiltonian expressions) is the library hexpr-atomci, in its own top-level folder.
HEXPR_ATOMCI ?= ../hexpr-atomci
CPPFLAGS += -Iinclude -I$(HEXPR_ATOMCI)/include
HXA_LIB := $(HEXPR_ATOMCI)/build/libhexpr_atomci.a
# static by default, as fast as before; shared: about 35% slower matrix elements
ifeq ($(HEXPR_ATOMCI_LINK),shared)
LDLIBS  := -L$(HEXPR_ATOMCI)/build -lhexpr_atomci -Wl,-rpath,$(abspath $(HEXPR_ATOMCI)/build) -lm
else
LDLIBS  := $(HXA_LIB) -lm
endif

AC_WORK   ?= $(HOME)/atomci_work
FTN_BUILD ?= $(AC_WORK)/m0/build
M0_RUN    ?= $(AC_WORK)/m0/run

BUILD := build
SRCS  := src/ac_smat.c src/ac_addr.c src/ac_tape.c src/ac_ints.c src/ac_ci.c \
         src/ac_basis.c src/ac_radial.c src/ac_lsj.c src/ac_tabrad.c \
         src/ac_biorth.c src/ac_hfsrel.c
TOOLS := $(BUILD)/acgen $(BUILD)/nfte_dump $(BUILD)/acci $(BUILD)/acterms $(BUILD)/acrad $(BUILD)/acfs $(BUILD)/acatsp \
         $(BUILD)/cfpgen
# system BLAS/LAPACK for the CI diagonalization (as src/Makefile)
LAPACK_LINK := $(shell pkg-config --libs lapack blas 2>/dev/null || pkg-config --libs openblas 2>/dev/null)
OBJS  := $(SRCS:src/%.c=$(BUILD)/%.o)
LIB   := $(BUILD)/libac.a

FTN_L0 := xwig6j.o delta.o f3l.o setwig.o
FTN_L1 := $(FTN_L0) xcfp.o setcfp.o cfpset.o isp.o ior.o

.PHONY: all check check-l0 check-l0-exact check-l1 check-l2 check-l3 \
        check-l3-addr check-l4 check-pyref check-rank check-cfpgen check-s2m1 check-s2m2 check-s2m3 check-s2m4 check-s2m6 \
        check-s2m6b check-s2m7 check-s2m9 check-s2m11 check-v5 check-atsp check-library clean

all: $(LIB) $(TOOLS)

$(BUILD)/%: tools/%.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LAPACK_LINK) $(LDLIBS)

$(BUILD):
	mkdir -p $@

$(HXA_LIB): $(wildcard $(HEXPR_ATOMCI)/src/* $(HEXPR_ATOMCI)/include/*)
	$(MAKE) -C $(HEXPR_ATOMCI)

$(BUILD)/%.o: src/%.c include/*.h $(wildcard $(HEXPR_ATOMCI)/include/*.h) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# the tools relink when the library changes, since they depend on $(LIB)
$(LIB): $(OBJS) $(HXA_LIB)
	rm -f $@
	ar rcs $@ $(OBJS)

# ---- tests ----------------------------------------------------------------

$(BUILD)/ftn_%.o: tests/ftn_%.f | $(BUILD)
	$(FC) -O2 -c -fno-automatic -ffixed-form $< -o $@

$(BUILD)/test_%.o: tests/test_%.c tests/ftn_decl.h include/*.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Itests $(CFLAGS) -c $< -o $@

$(BUILD)/test_l0_wigner: $(BUILD)/test_l0_wigner.o $(BUILD)/ftn_wig6j.o $(LIB)
	$(FC) -o $@ $^ $(addprefix $(FTN_BUILD)/,$(FTN_L0)) $(LDLIBS)

$(BUILD)/test_l1_cfp: $(BUILD)/test_l1_cfp.o $(BUILD)/ftn_helper.o $(BUILD)/ftn_wig6j.o $(LIB)
	$(FC) -o $@ $^ $(addprefix $(FTN_BUILD)/,$(FTN_L1)) $(LDLIBS)

$(BUILD)/wigner_dump: tests/wigner_dump.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LDLIBS)

$(BUILD)/test_l2_shms: tests/test_l2_shms.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LDLIBS)

$(BUILD)/test_l3_smat: tests/test_l3_smat.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LDLIBS)

FTN_ADDR := $(FTN_L1) setiad.o

$(BUILD)/test_l3_addr: $(BUILD)/test_l3_addr.o $(BUILD)/ftn_helper.o $(BUILD)/ftn_wig6j.o $(LIB)
	$(FC) -o $@ $^ $(addprefix $(FTN_BUILD)/,$(FTN_ADDR)) $(LDLIBS)

$(BUILD)/nfte_diff: tests/nfte_diff.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LDLIBS)

# L4: regenerate each M0 tape (fort.15 of the traced runs) and compare
L4_RUNS := $(wildcard $(M0_RUN)/trace-*)
check-l4: $(BUILD)/acgen $(BUILD)/nfte_diff
	@for d in $(L4_RUNS); do \
	    echo "== $$d"; \
	    $(BUILD)/acgen $(FTN_BUILD)/CFPTBL $$d/fort.15 $(BUILD)/$$(basename $$d).nfte || exit 1; \
	    $(BUILD)/nfte_diff $$d/fort.15 $(BUILD)/$$(basename $$d).nfte || exit 1; \
	done

check: check-l0 check-l0-exact check-l1 check-l2 check-l3 check-l3-addr check-l4

# the library by itself, away from this tree, and its PySCF example
check-library:
	$(MAKE) -C $(HEXPR_ATOMCI) check-standalone check-pyscf PYTHON="$(PYTHON)"

# ---- stage 2 --------------------------------------------------------------
# PythonRef (c/python) needs numpy; set PYREF_PYTHON to a Python that has it.
PYREF_PYTHON ?= python3

$(BUILD)/probe_rank: tests/probe_rank.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LDLIBS)

$(BUILD)/acmat: tests/acmat.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LDLIBS)

check-pyref:
	cd python && $(PYREF_PYTHON) test_pyref.py

# S2-M1: the general-form engine against PythonRef (all tensor types,
# explicit closed shells)
check-s2m1: $(BUILD)/acmat
	cd python && $(PYREF_PYTHON) test_s2m1.py ../$(BUILD)/acmat $(FTN_BUILD)/CFPTBL

# S2-M2: the stage-2 CI against the Fortran (V2a needs tests/m5/run_m5.sh
# to have been run; V2b runs the Fortran itself)
check-s2m2: $(TOOLS)
	bash tests/s2m2/run_v2a.sh
	bash tests/s2m2/run_v2b.sh

check-s2m3: $(TOOLS)
	PYTHON=$(PYREF_PYTHON) bash tests/s2m3/run_v3.sh

# V5 of S2-M4: against a determinant CI on PySCF integrals (PYREF_PYTHON
# must have PySCF)
check-v5: $(TOOLS)
	PYTHON=$(PYREF_PYTHON) bash tests/v5/run_v5.sh

# against ATSP in its own HF orbitals (needs ATSP built; ATSP_BIN)
check-atsp: $(TOOLS)
	bash tests/atsp/run_atsp.sh
	bash tests/atsp/run_atsp_ci.sh
	bash tests/atsp/run_atsp_bp.sh

$(BUILD)/hydro: tests/hydro.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LDLIBS)

# S2-M6a: mass velocity and Darwin against the Dirac energy of
# hydrogen-like ions (the check-s2m3 acrad runs compare MV with the
# basis-pair formula on the real bases)
check-s2m6: $(BUILD)/hydro
	$<

# S2-M6b: orbit-orbit against closed forms for Gaussian orbitals (the
# contact terms are in check-v5, the comparison with ATSP in check-atsp)
check-s2m6b: $(TOOLS)
	bash tests/s2m6b/run_oo.sh

# S2-M7: the orbit-orbit Zeeman term of g against a momentum-space reference
# for Gaussian orbitals (the two-electron gauge correction is in check-v5)
check-s2m7: $(TOOLS)
	PYTHON=$(PYREF_PYTHON) bash tests/s2m7/run_oz.sh

$(BUILD)/acso: tests/acso.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LAPACK_LINK) $(LDLIBS)

# S2-M4: Coulomb + H_SO1 in the LSJ basis against PythonRef (V1) and the
# analytic jj limit and Lande factors (V4)
check-s2m4: $(BUILD)/acso $(TOOLS)
	cd python && $(PYREF_PYTHON) test_s2m4.py ../$(BUILD)/acso $(FTN_BUILD)/CFPTBL
	bash tests/s2m4/run_real.sh

$(BUILD)/test_biorth: tests/test_biorth.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LAPACK_LINK) $(LDLIBS)

$(BUILD)/test_davidson: tests/test_davidson.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIB) $(LAPACK_LINK) $(LDLIBS)

# S2-M9: the algebra of the biorthonormal transformation (no radial
# functions): triangular factors, E_ab, the CI counter-transformation, closure;
# and the Davidson solver for the large J blocks against dsyev
# S2-M11: the regularized alpha^2 hfs correction (acatsp hfsrel) against the
# Python prototype in the same orbitals
check-s2m11: $(TOOLS)
	FTN_BUILD=$(FTN_BUILD) bash tests/s2m11/hfsrel_proto.sh

check-s2m9: $(BUILD)/test_biorth $(BUILD)/test_davidson
	$< $(FTN_BUILD)/CFPTBL
	$(BUILD)/test_davidson

check-rank: $(BUILD)/probe_rank
	$< $(FTN_BUILD)/CFPTBL

check-l3-addr: $(BUILD)/test_l3_addr
	$< $(FTN_BUILD)/CFPTBL $(wildcard $(M0_RUN)/trace-*/trace/smat.txt)

check-l2: $(BUILD)/test_l2_shms
	$< $(FTN_BUILD)/CFPTBL $(wildcard $(M0_RUN)/trace-*/trace/shms.txt)

check-l3: $(BUILD)/test_l3_smat
	$< $(FTN_BUILD)/CFPTBL $(wildcard $(M0_RUN)/trace-*/trace/smat.txt)

check-l0: $(BUILD)/test_l0_wigner
	$<

check-l0-exact: $(BUILD)/wigner_dump
	$(PYTHON) tests/l0_exact.py $<

# M1b: the CFP table generated in C against the Fortran CFPTBL, value by
# value and as a file, byte for byte
check-cfpgen: $(BUILD)/cfpgen
	$< -c $(FTN_BUILD)/CFPTBL
	$< $(BUILD)/CFPTBL.gen
	cmp $(BUILD)/CFPTBL.gen $(FTN_BUILD)/CFPTBL && echo "CFPTBL.gen: byte-identical"

check-l1: $(BUILD)/test_l1_cfp
	$< $(FTN_BUILD)/CFPTBL $(wildcard $(M0_RUN)/trace-*/trace/xcfp.txt)

clean:
	rm -rf $(BUILD)
