#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [ ! "$(black --version)" ]
then
    echo "Please install black."
    exit 1
fi
if [ ! "$(isort --version)" ]
then
    echo "Please install isort."
    exit 1
fi

# cd to the project directory
cd "$(dirname "$0")/.." || exit 1

GIT_URL_1="https://github.com/pytorch/elastic.git"
GIT_URL_2="git@github.com:pytorch/elastic.git"

UPSTREAM_URL="$(git config remote.upstream.url)"

if [ -z "$UPSTREAM_URL" ]
then
    echo "Setting upstream remote to $GIT_URL_1"
    git remote add upstream "$GIT_URL_1"
elif [ "$UPSTREAM_URL" != "$GIT_URL_1" ] && \
     [ "$UPSTREAM_URL" != "$GIT_URL_2" ]
then
    echo "upstream remote set to $UPSTREAM_URL."
    echo "Please delete the upstream remote or set it to $GIT_URL_1 to use this script."
    exit 1
fi

# fetch upstream
git fetch upstream


CHANGED_FILES="$(git diff --diff-filter=ACMRT --name-only upstream/master | grep '\.py$' | tr '\n' ' ')"

if [ "$CHANGED_FILES" != "" ]
then
    echo "Running isort..."
    isort "$CHANGED_FILES" --recursive --multi-line 3 --trailing-comma --force-grid-wrap 0 \
    --line-width 88 --lines-after-imports 2 --combine-as --section-default THIRDPARTY

    echo "Running black..."
    black "$CHANGED_FILES"
else
    echo "No changes made to any Python files. Nothing to do."
    exit 0
fi

# Check if any files were modified by running isort + black
# If so, then the files were formatted incorrectly (e.g. did not pass lint)
CHANGED_FILES="$(git diff --name-only | grep '\.py$' | tr '\n' ' ')"
if [ "$CHANGED_FILES" != "" ]
then
    # need this so that CircleCI fails
    exit 1
fi
