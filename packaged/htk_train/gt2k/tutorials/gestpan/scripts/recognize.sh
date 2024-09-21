#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


##################################################
#
# This program is designed to take a data set and run the recognizer on it.
# Written by amin@cc.gatech.edu
# Based on the train.sh script by bashear@cc
#
# What you need:
#
# The hmm/newMacros created from the training script
# A data file with VECTOR_LENGTH features.
# The word.lattice, dict, and command files used for training.
# The options.sh file describing the project
#
# usage:
# recognize.sh <data to recognize> <place to store results> <options file> \
#		<trained model>
# 
##################################################

echo Processing $1

OPTIONS_FILE=$3;

if [ ! -x "${OPTIONS_FILE}" ]; then
   echo "Can't read options file '${OPTIONS_FILE}', make sure the file exists and is readable and executable"
   exit;
fi

. ${OPTIONS_FILE}
HMM_LOAD_OPT="-H"
RESULTS_FILE=$2
MODEL=$4;


$UTIL_DIR/prepare $1 $VECTOR_LENGTH $1.ext 	# First, prepare the file.

					# Now run HVite to recognize
					# the gesture

${HTKBIN}HVite -A -T $TRACE_LEVEL -H ${MODEL} -i $RESULTS_FILE -w ${PRJ}/word.lattice -n 10 20 ${DICTFILE} ${TOKENS} $1.ext 
