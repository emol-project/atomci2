# hexpr-atomci: the expressions of atomic CSF matrix elements.
#
#   make             build/libhexpr_atomci.a and build/libhexpr_atomci.so
#   make check       all checks; needs a Python with numpy (PYTHON)
#   make install     PREFIX/lib, PREFIX/include/hexpr_atomci
#   make check-standalone  make check on a copy of this folder alone
#   make check-pyscf       the PySCF example; needs PySCF
#   make dist VERSION=x.y  this folder as a tarball, from the last commit

CC      ?= cc
VERSION ?= dev
PYTHON  ?= python3
PREFIX  ?= /usr/local
WARN    := -Wall -Wextra -pedantic
CFLAGS  ?= -O2 -std=c11 $(WARN) -Werror
CPPFLAGS += -Iinclude -Isrc
LDLIBS  := -lm

# The 6j, 9j and tree caches are thread-local. Through the shared library that made
# ac_ci_hij about 35% slower than the static one. -ftls-model=initial-exec removes
# this, but then dlopen fails (too much TLS), so use the static library for speed.
BUILD := build
SRCS  := src/ac_wigner.c src/ac_ftnio.c src/ac_cfp.c src/ac_cfpgen.c src/ac_csf.c \
         src/ac_csfgen.c src/ac_optree.c src/ac_factor.c src/ac_terms.c
OBJS  := $(SRCS:src/%.c=$(BUILD)/%.o)
ALIB  := $(BUILD)/libhexpr_atomci.a
SOLIB := $(BUILD)/libhexpr_atomci.so
HDRS  := $(wildcard include/*.h)

.PHONY: all check check-example check-d2 check-terms check-terms-so check-dlopen check-cfp check-wigner check-pyref check-engine check-standalone check-pyscf dist install clean

all: $(ALIB) $(SOLIB)

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: src/%.c $(HDRS) src/ac_labmath.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

$(ALIB): $(OBJS)
	rm -f $@
	ar rcs $@ $^

$(SOLIB): $(OBJS)
	$(CC) -shared -Wl,-soname,libhexpr_atomci.so $^ -o $@ $(LDLIBS)

$(BUILD)/test_terms: tests/test_terms.c tests/terms_spaces.h $(ALIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(ALIB) -o $@ $(LDLIBS)

$(BUILD)/test_terms_so: tests/test_terms.c tests/terms_spaces.h $(SOLIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -L$(BUILD) -lhexpr_atomci -Wl,-rpath,'$$ORIGIN' -o $@ $(LDLIBS)

$(BUILD)/wigner_dump: tests/wigner_dump.c $(ALIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(ALIB) -o $@ $(LDLIBS)

$(BUILD)/acmat: tests/acmat.c $(ALIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(ALIB) -o $@ $(LDLIBS)

$(BUILD)/cfpgen: tests/cfpgen.c $(ALIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(ALIB) -o $@ $(LDLIBS)

check: check-example check-d2 check-terms check-terms-so check-dlopen check-cfp check-wigner check-pyref check-engine

# Nothing outside this folder may be needed: this folder alone, copied elsewhere.
check-standalone:
	T=$$(mktemp -d) && tar --exclude=./$(BUILD) -cf - . | tar -xf - -C "$$T" && \
	$(MAKE) -C "$$T" check PYTHON="$(PYTHON)"; rc=$$?; rm -rf "$$T"; exit $$rc

# Works alone or inside atomci2: git archive takes the current folder only.
dist: | $(BUILD)
	git archive --format=tar.gz --prefix=hexpr-atomci-$(VERSION)/ \
	    -o $(BUILD)/hexpr-atomci-$(VERSION).tar.gz HEAD .
	@echo $(BUILD)/hexpr-atomci-$(VERSION).tar.gz

# the terms of d2 and p2 as Slater integrals, with no radial integrals at all
$(BUILD)/d2_terms: examples/d2_terms.c $(ALIB)
	$(CC) -Iinclude $(CFLAGS) $< $(ALIB) -o $@ $(LDLIBS)

check-d2: $(BUILD)/d2_terms
	$<

# integrals from PySCF; every level against the FCI of PySCF
$(BUILD)/libhxa_block.so: examples/pyscf_atom/hxa_block.c $(ALIB)
	$(CC) -Iinclude $(CFLAGS) -fPIC -shared $< $(ALIB) -o $@ $(LDLIBS)

check-pyscf: $(BUILD)/libhxa_block.so
	@$(PYTHON) -c 'import pyscf' 2> /dev/null || { echo "SKIPPED check-pyscf: $(PYTHON) cannot import pyscf (set PYTHON to a Python with PySCF)"; exit 0; }; \
	$(PYTHON) examples/pyscf_atom/pyscf_atom.py $<

# the shared library loaded at run time, in several threads
check-dlopen: $(SOLIB)
	$(PYTHON) tests/python/test_dlopen.py $(SOLIB)

# the public header alone, with coefficients known by elementary rules
$(BUILD)/example: tests/example.c $(ALIB)
	$(CC) -Iinclude $(CFLAGS) $< $(ALIB) -o $@ $(LDLIBS)

check-example: $(BUILD)/example
	$<

# the expressions against values from atomci2 (tests/ref)
check-terms: $(BUILD)/test_terms
	$< tests/ref_hij.txt

check-terms-so: $(BUILD)/test_terms_so
	$< tests/ref_hij.txt

# the generated CFP table against the Fortran ATOMCI's, byte for byte
check-cfp: $(BUILD)/cfpgen
	$< $(BUILD)/CFPTBL.gen
	test "$$(sha256sum < $(BUILD)/CFPTBL.gen | cut -d' ' -f1)" = "$$(cut -d' ' -f1 tests/ref/CFPTBL.sha256)" && echo "CFPTBL: byte-identical"

# 6j and 9j against exact rational arithmetic
check-wigner: $(BUILD)/wigner_dump
	$(PYTHON) tests/l0_exact.py $<

# the independent determinant reference, and the engine against it
check-pyref:
	cd tests/python && $(PYTHON) test_pyref.py

check-engine: $(BUILD)/acmat
	cd tests/python && $(PYTHON) test_s2m1.py ../../$(BUILD)/acmat -

install: all
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include/hexpr_atomci
	install -m 644 $(ALIB) $(SOLIB) $(DESTDIR)$(PREFIX)/lib
	install -m 644 $(HDRS) $(DESTDIR)$(PREFIX)/include/hexpr_atomci

clean:
	rm -rf $(BUILD)
