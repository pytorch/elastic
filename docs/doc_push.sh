#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Builds docs from the checkedout HEAD
# and pushes the artifacts to gh-pages branch in github.com/pytorch/elastic
#
# 1. sphinx generated docs are copied to <repo-root>/<version>
# 2. if a release tag is found on HEAD then redirects are copied to <repo-root>/latest
# 3. if no release tag is found on HEAD then redirects are copied to <repo-root>/master
#
# gh-pages branch should look as follows:
# <repo-root>
#           |- 0.1.0rc2
#           |- 0.1.0rc3
#           |- <versions...>
#           |- master (redirects to the most recent ver in trunk)
#           |- latest (redirects to the most recent release)
# If the most recent  release is 0.1.0 and master is at 0.1.1rc1 then,
# https://pytorch.org/elastic/master -> https://pytorch.org/elastic/0.1.1rc1
# https://pytorch.org/elastic/latest -> https://pytorch.org/elastic/0.1.0
#
# Redirects are done via Jekyll redirect-from  plugin. See:
#   sources/scripts/create_redirect_md.py
#   Makefile (redirect target)
#  (on gh-pages branch) _layouts/docs_redirect.html

repo_root=$(git rev-parse --show-toplevel)
branch=$(git rev-parse --abbrev-ref HEAD)
commit_id=$(git rev-parse --short HEAD)

if ! release_tag=$(git describe --tags --exact-match HEAD 2>/dev/null); then
    echo "No release tag found, building docs for master..."
    redirect=master
    release_tag="master"
else
    echo "Release tag $release_tag found, building docs for release..."
    redirect=latest
fi

echo "Installing torchelastic from $repo_root..."
cd "$repo_root" || exit
pip uninstall -y torchelastic
python setup.py install

torchelastic_ver=$(python -c "import torchelastic; print(torchelastic.__version__)")

echo "Building PyTorch Elastic v$torchelastic_ver docs..."
docs_dir=$repo_root/docs
build_dir=$docs_dir/build
cd "$docs_dir" || exit
pip install -r requirements.txt
make clean html

tmp_dir=/tmp/torchelastic_docs_tmp
rm -rf "${tmp_dir:?}"

echo "Checking out gh-pages branch..."
gh_pages_dir="$tmp_dir/elastic_gh_pages"
git clone -b gh-pages --single-branch git@github.com:pytorch/elastic.git  $gh_pages_dir

echo "Copying doc pages for $torchelastic_ver into $gh_pages_dir..."
rm -rf "${gh_pages_dir:?}/${torchelastic_ver:?}"
cp -R "$build_dir/$torchelastic_ver/html" "$gh_pages_dir/$torchelastic_ver"

echo "Copying redirects for $redirect -> $torchelastic_ver..."
rm -rf "${gh_pages_dir:?}/${redirect:?}"
cp -R "$build_dir/redirects" "$gh_pages_dir/$redirect"

cd $gh_pages_dir || exit
git add .
git commit --quiet -m "[doc_push][$release_tag] built from $commit_id ($branch). Redirect: $redirect -> $torchelastic_ver."
git push
