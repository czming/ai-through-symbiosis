#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


###############################################################################
# Genereate Training and Testing for k-fold cross validation
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
# example:  gen_leave_one_out.sh datafile training testing
#
#   ==> training_0 testing_0 .... training_n testing_n  
###############################################################################
NAME_SCRIPT=$4
ALL_FILES=$1
TRAINING=$2
TESTING=$3
. $5

num_samples=`cat $ALL_FILES | wc -l`

fold=$VALIDATION_ITERATIONS         # how many folds we want to have

foldsize=$((num_samples/fold))
current=0

for i in `cat $ALL_FILES`; do
    
    iter=$((current/foldsize))

    if [[ $iter -ge $fold ]]; then
        iter=$((fold-1))
    elif [[ $((current%foldsize)) = 0 ]]; then
        # testing file.  List all files except for the current files.
        head_list=`head -n $current $ALL_FILES`               # [0..current)
        tail_list=`tail -n $((num_samples-current-foldsize)) $ALL_FILES` # (current..end]
        echo "$head_list" >> `$NAME_SCRIPT $TRAINING $iter`
        echo "$tail_list" >> `$NAME_SCRIPT $TRAINING $iter`
    fi

    echo $i >> `$NAME_SCRIPT $TESTING $iter`
    # update the count for the next iteration
    current=$((current+1))

done
