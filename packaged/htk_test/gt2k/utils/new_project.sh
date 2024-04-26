#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


#######################################################
# new_project.sh
# 
# creates a new project based on command line inputs
# usage:
# new_project.sh <project dir> <vector length> <gt2k base>
#
# Where:
# project dir	is the main directory of the project (does not have to exist)
# vector length	is the size of your data vector
# gt2k base 	is the base install directory of gt2k (eg - /usr/local/gt2k)
#
#######################################################

if [ $# -lt 3 ]; then
	echo "Usage: $0 <project dir> <vector length> <gt2k base>"
	exit 1;
fi

PROJ_DIR=$1
VEC_LEN=$2
GT2K_BASE=$3

if [ ! -d ${GT2K_BASE} ]; then
	echo "GT2k not found at ${GT2K_BASE}!"
	exit 1;
fi

UTILS=${GT2K_BASE}/utils
TEMPLATE=${GT2K_BASE}/template

echo "Creating new GT2k Project in ${PROJ_DIR}"

install -d ${PROJ_DIR}
cp -r ${TEMPLATE}/* ${PROJ_DIR}

cat ${TEMPLATE}/scripts/options.sh | sed -e "s|^PRJ=.*#|PRJ=${PROJ_DIR}#|" \
	| sed -e "s/^VECTOR_LENGTH=.*#/VECTOR_LENGTH=${VEC_LEN}#/" \
	| sed -e "s|^UTIL_DIR=.*#|UTIL_DIR=${UTILS}#|" \
	> ${PROJ_DIR}/scripts/options.sh

echo "A clean project has been created in ${PROJ_DIR}"
echo "Be sure to make sure all settings in ${PROJ_DIR}/scripts/options.sh"
echo "are correct, and to create your project-specific files such as your"
echo "initial hmm topology and master label file."
echo "See the examples and tutorials in ${GT2K_BASE} for details"
