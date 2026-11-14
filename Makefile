# Top-level delegator for ATOMCI.
#
# The build itself lives in src/ (all *.f/*.fpl/iqtime.c, include/, fplincl/,
# gDKsapporo/, and the real Makefile).  The atinputs front end
# (Python package + its bin/atinpgen wrapper) is in the top-level scripts/
# directory.
#
#   make            build everything under src/ (atomci, atgen, fpl, CFPTBL,
#                   ...); artifacts stay in src/ where the build rules put them.
#   make install    assemble a clean, runnable layout under $(prefix)
#                   (default ./local):
#                       $(prefix)/bin/   atomci, atgen, fpl, atinpgen
#                                        (+ the atomci.sh helper)
#                       $(prefix)/lib/   atinputs/ (the Python package
#                                        bin/atinpgen runs)
#                       $(prefix)/CFPTBL
#                   CFPTBL sits at the $(prefix) ROOT (not under bin/ or a
#                   data/ dir) so it is easy to symlink into a work directory.
#                   Set ATOMCI_HOME=$(prefix), then for each calculation work in
#                   your own directory, symlink CFPTBL in, and run by path:
#                       export ATOMCI_HOME=$(prefix)
#                       mkdir run && cd run
#                       ln -sf "$$ATOMCI_HOME/CFPTBL" ./CFPTBL
#                       "$$ATOMCI_HOME/bin/atomci" < /path/to/samples/native/boron.stdin
#                   `make install` prints the exact export line to use.
#   make clean      clean src/ build artifacts, and remove the default ./local
#
# Requires system BLAS/LAPACK; src/
# Makefile's lapack-guard target fails with install instructions if it's
# missing, before any link is attempted.

prefix = $(CURDIR)/local

# heff/ (the effective-Hamiltonian code, Python; see heff/README.md):
#   make heff-test     its pytest suite without the slow example runs (the
#                      Level 1 regression runs only when HEFF_DATA points to
#                      the research-data record)
#   make heff-examples the three examples of heff/examples/ from scratch with
#                      the ATOMCI of this repository (`make install` first;
#                      a few minutes each)
#   make heff-tables   rebuild the paper tables from the H_eff dumps in
#                      $(HEFF_DATA) into $(HEFF_WORK) (default heff/work) and
#                      compare them with heff/checks/reference/
PYTHON ?= python3

.PHONY: all clean install heff-test heff-examples heff-tables

all:
	$(MAKE) -C src all

clean:
	$(MAKE) -C src clean
	-rm -rf $(CURDIR)/local

install: all
	# $(prefix)/bin and $(prefix)/lib are fully owned by this install step --
	# wipe and recreate them fresh every time, rather than only ever adding
	# to them, so a tool that gets renamed or removed from the source tree
	# doesn't leave an old copy sitting on PATH forever. $(prefix)/CFPTBL is
	# a single file, always overwritten unconditionally below, so it
	# doesn't need this treatment.
	rm -rf $(prefix)/bin $(prefix)/lib
	mkdir -p $(prefix)/bin $(prefix)/lib
	cp src/atomci src/atgen src/fpl $(prefix)/bin
	cp scripts/atomci.sh $(prefix)/bin
	cp scripts/atinpgen.sh $(prefix)/bin/atinpgen
	chmod +x $(prefix)/bin/atinpgen
	rm -rf $(prefix)/lib/atinputs
	cp -r scripts/atinputs $(prefix)/lib/atinputs
	rm -rf $(prefix)/lib/atinputs/tests $(prefix)/lib/atinputs/__pycache__ \
	       $(prefix)/lib/atinputs/.pytest_cache $(prefix)/lib/atinputs/.hypothesis \
	       $(prefix)/lib/atinputs/csfgen/__pycache__
	rm -f $(prefix)/lib/atinputs/requirements-test.txt
	cp src/CFPTBL $(prefix)/CFPTBL
	@echo ""
	@echo "Installed to $(prefix)"
	@echo "Add this to your shell (or your shell profile):"
	@echo "  export ATOMCI_HOME=$(prefix)"

heff-test:
	cd heff && $(PYTHON) -m pytest -q tests

heff-examples:
	cd heff && $(PYTHON) -m pytest -q -m slow tests/test_examples.py

heff-tables:
	$(PYTHON) heff/scripts/tables/rebuild_level1.py --check
