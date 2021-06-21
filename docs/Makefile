
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
VERSION       := "0.2.3.dev0"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Deleting build directory"
	rm -rf "$(BUILDDIR)"

.PHONY: help Makefile clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)/$(VERSION)" $(SPHINXOPTS) $(O)
