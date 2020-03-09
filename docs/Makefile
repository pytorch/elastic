  
# Minimal makefile for Sphinx documentation
# Usage:
#    make html
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = torchelastic
SOURCEDIR     = source
BUILDDIR      = build
VERSION       := $(shell python -c "import torchelastic; print(torchelastic.__version__)")

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

redirects:
	python source/scripts/create_redirect_md.py --source_dir "$(SOURCEDIR)" --build_dir "$(BUILDDIR)/redirects"

clean:
	@echo "Deleting build directory"
	rm -rf "$(BUILDDIR)"

.PHONY: help Makefile redirects clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile redirects
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)/$(VERSION)" $(SPHINXOPTS) $(O)
