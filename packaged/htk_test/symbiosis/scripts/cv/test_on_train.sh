#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


###############################################################################
# Generate Training and Testing sets for Cross Validation:
#
# generates two files.  one, a list of filenames to be used for training and
# one file will be a list of filenames to be used for testing.
#
# argument 1: a file listing all datafiles to be considered
# argument 2: name of where to save the training file
# argument 3: name of where to save the testing file
# argument 4: script to generate the name of the training/test files
# argument 5: options file for the project (so we can locate the utils dir)
#
###############################################################################
NAME_SCRIPT=$4
ALL_FILES=$1
TRAINING=`$NAME_SCRIPT $2 0`	# generate name for the training file
TESTING=`$NAME_SCRIPT $3 0`	# generate name for the testing file

. $5				# include the project options

cp $ALL_FILES $TRAINING
cp $ALL_FILES $TESTING