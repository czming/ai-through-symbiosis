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
TRAINING=$2	
TESTING=$3

. $5				# include the project options

RANDOMIZE=${UTIL_DIR}/random	# where is the random number generator located
PERCENT_TESTING=.2		# how many files do we want to use for testing
				# right now it is 1/5 testing, 4/5 training
ITERATIONS=10 # number of times to repeat cross validation

# get the number of data samples that we have
num_samples=`cat $ALL_FILES | wc -l`;

for ((i=0; i<ITERATIONS;i++)); do
	TRAIN_NAME=`$NAME_SCRIPT $TRAINING $i`
	TEST_NAME=`$NAME_SCRIPT $TESTING $i`
	# divide the filenames up into two seperate files.
	$RANDOMIZE $ALL_FILES $num_samples $TRAIN_NAME $TEST_NAME $PERCENT_TESTING

done