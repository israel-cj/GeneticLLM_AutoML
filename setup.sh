#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"latest"}
REPO=${2:-"https://github.com/israel-cj/GeneticLLM_AutoML.git"}
PKG=${3:-"GeneticLLM_AutoML"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

#create local venv
. ${HERE}/../shared/setup.sh ${HERE} true

PIP install -r ${HERE}/requirements.txt

#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
TARGET_DIR="${HERE}/lib/${PKG}"
rm -Rf ${TARGET_DIR}
git clone ${REPO} ${TARGET_DIR}
PIP install -U -e ${TARGET_DIR}