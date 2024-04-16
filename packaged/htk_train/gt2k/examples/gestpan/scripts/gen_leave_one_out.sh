#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


###############################################################################
# Genereate Training and Testing for LEAVE-ONE-OUT validation
#
# generates N pairs of files.  one, a list of filenames to be used for training
# and one file will be a list of filenames to be used for testing.
#
# argument 1: a file listing all datafiles to be considered
# argument 2: base name of where to save the training file
# argument 3: base name of where to save the testing file
# argument 4: script to generate the name of the training/test files
# argument 5: options file for this project (not used)
#
# example:	gen_leave_one_out.sh datafile training testing
#
#	==> training_0 testing_0 .... training_n testing_n	
###############################################################################
NAME_SCRIPT=$4
ALL_FILES=$1
TRAINING=$2
TESTING=$3

num_samples=`cat $ALL_FILES | wc -l`;
current=0			# which element in the file list is currently 
				# being left out of the training set.

# select one element form the file and make it the single element in the
# testing file, all other elements will be in the training file.  this
# should be done for every cobination possible (ie: C(num_samples, 1) )
for i in `cat $ALL_FILES`; do
    
    # $i will always be the single element to place into the testing file.
    # since we are generating several testing files, index it by the number
    # of the current element.
    
#    echo $i > ${TESTING}$current
     echo $i > `$NAME_SCRIPT $TESTING $current`
    # the training file will consist of all elements except the one in the
    # testing file.  List all files except for the current file.
    head_list=`head -n $current $ALL_FILES`  	   	      # [0..current)
    tail_list=`tail -n $((num_samples-current-1)) $ALL_FILES` # (current..end]
    echo $head_list $tail_list > `$NAME_SCRIPT $TRAINING $current`

    # update the count for the next iteration
    current=$((current+1))

done
