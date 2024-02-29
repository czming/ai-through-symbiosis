#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

###############################################################################
#
# this boot-straps htk experiments
#
# maintained by brashear@cc.gatech.edu, turtle@cc.gatech.edu
#		haileris@cc.gatech.edu
#
###############################################################################

# NOTES: scripts/train.sh scripts/options.sh &> output
#

# Load in project options (specified on the command line!)
if [ -z "$1" ]; then

	echo "usage: $0 <options file>"
	exit
fi

OPTIONS_FILE=$1
. ${OPTIONS_FILE}


###############################################################################
##########################################################
##########################################################
##							##
##  DO NOT EDIT BELOW THIS LINE 			##
##  unless you know HTK and know what you are doing.	##
##							##
##########################################################
##########################################################

# check options for proper values, verify existence and executability of utils
. ${UTIL_DIR}/check_opts.sh

##############################################################################
##############################################################################
# User Options are ok, This is the important part of the script that does
# the actual work
##############################################################################
##############################################################################

# reads in ${TOKENS} from command file and creates a grammar and dictionary
# assumes that the grammar is a simple, single gesture grammar
typeset -l GEN_GRAMMAR		# make sure it is all lowercase
if [[ "${GEN_GRAMMAR}" == "yes" ]] ||
   [[ "${GEN_GRAMMAR}" == "1" ]]; then

	rm ${GRAMMARFILE}
	rm ${DICTFILE}
	${GRAMMAR_PROG}
fi

# translates ${DATAFILES_LIST} to EXT format (USER) for HTK
# you can read the output of this step by running HList on *.ext

# check to see if we really want to do this
typeset -l GEN_EXT_FILES	# make sure it is all lowercase

if [[ "${GEN_EXT_FILES}" == "yes" ]] ||
   [[ "${GEN_EXT_FILES}" == "1" ]] &&
   [[ ! -f "${EXT_DIR}/done" ]]; then

   echo
   echo "*****************************************************"
   echo converting data files to .ext files
   echo "*****************************************************"
   rm -rf $EXT_DIR/*
   for n in $(cat ${DATAFILES_LIST});
   do
	 if [[ ! -d `dirname ${EXT_DIR}/$n` ]]; then
		echo "Making Directory: `dirname ${EXT_DIR}/$n`"
		mkdir -p `dirname ${EXT_DIR}/$n`
	 fi
         ${PREPARE_DATA} $n ${VECTOR_LENGTH} ${EXT_DIR}/$n.ext
   #      echo converted $n to `ls ${EXT_DIR} | tail -n 1`  

   done
   echo "1" > ${EXT_DIR}/done
fi


########################################################################
# Prepare data for training/testing
########################################################################

DATA_SAMPLES=all-extfiles

TT_NAME_SCRIPT=$SCRIPTS_DIR/gen_train_test_name.sh      # make consistent names

HMM_BASE_DIR=$HMM_TRAINING
BASE_OUTPUT_MLF=$OUTPUT_MLF
BASE_OUTPUT_MLF_WORD=$OUTPUT_MLF_WORD
BASE_MLF_LOCATION_GEN=$MLF_LOCATION_GEN

#clean up old training data
rm -f $LOG_RESULTS
rm -f $LOG_RESULTS_WORD
rm -f $DATA_SAMPLES
rm -f $OUTPUT_MLF*
rm -f $OUTPUT_MLF_WORD*
rm -f $WORD_LATTICE*
rm -rf $MLF_LOCATION_GEN/*
for i in ${HMM_TRAINING}*\.*
do
	rm -f $i/*
	rmdir $i
done

# generate a list of all data samples HTK has avaliable to it.
#ls ${EXT_DIR}/*.ext > $DATA_SAMPLES
find ${EXT_DIR}/ | grep "\.ext$" | sort $SORT_OPTION > $DATA_SAMPLES


${HTKBIN}HParse ${GRAMMARFILE} ${WORD_LATTICE}
if [[ $WORD_LEVEL = "yes" ]] || [[ $WORD_LEVEL = "1" ]]; then
	${HTKBIN}HParse ${GRAMMARFILE_WORD} ${WORD_LATTICE}_word
fi

MIN_CYCLES=1

# based on the TRAIN_TEST_VALIDATION variable select which script is to be
# used for generating the testing and training set of data.  This variable will
# also determine the number or times training and testing are exectuted
if [[ $TRAIN_TEST_VALIDATION = "CROSS" ]]; then

	TRAIN_TEST_SCRIPT=$SCRIPTS_DIR/cv/gen_cross_val.sh	
	TEST_TRAIN_CYCLES=$MIN_CYCLES

elif [[ $TRAIN_TEST_VALIDATION = "LEAVE_ONE_OUT" ]]; then

	TRAIN_TEST_SCRIPT=$SCRIPTS_DIR/cv/gen_leave_one_out.sh	
	TEST_TRAIN_CYCLES=`cat $DATA_SAMPLES | wc -l`
elif [[ $TRAIN_TEST_VALIDATION = "REPEAT_CROSS" ]]; then

	TRAIN_TEST_SCRIPT=$SCRIPTS_DIR/cv/repeat_cross_val.sh
	TEST_TRAIN_CYCLES=$VALIDATION_ITERATIONS
elif [[ $TRAIN_TEST_VALIDATION = "K_FOLD" ]]; then

	TRAIN_TEST_SCRIPT=$SCRIPTS_DIR/cv/k_fold.sh
	TEST_TRAIN_CYCLES=$VALIDATION_ITERATIONS
elif [[ $TRAIN_TEST_VALIDATION = "TEST_ON_TRAIN" ]]; then

	TRAIN_TEST_SCRIPT=$SCRIPTS_DIR/cv/test_on_train.sh
	TEST_TRAIN_CYCLES=$MIN_CYCLES
else
    echo "invalid testing/training option"
    echo "edit train.sh or options.sh to fix. Exiting ... "
    exit;
fi

# generate the training and testing files, if we're going to
typeset -l GEN_TRAIN_TEST 	# make sure it is all lowercase

if [[ "${GEN_TRAIN_TEST}" == "yes" ]] ||
   [[ "${GEN_TRAIN_TEST}" == "1" ]] ; then
	echo Generating training/test sets, could take a while
	rm -f $TRAINING_BASENAME*
	rm -f $TESTING_BASENAME*
	$TRAIN_TEST_SCRIPT $DATA_SAMPLES $TRAINING_BASENAME $TESTING_BASENAME \
			   $TT_NAME_SCRIPT $OPTIONS_FILE 
fi


# based on the type of training/testing validation, iterate through the
# training process
cycle=0;
while [[ $cycle -lt $TEST_TRAIN_CYCLES ]]
do
	$SCRIPTS_DIR/train_parallel_subprocess.sh $OPTIONS_FILE $cycle &
	# update the cycle for the next iteration
	cycle=$((cycle+1))
done  # matches the "while" cycles of training

wait


# print the results of HResults of CROSS_VALIDATION
if [[ $TRAIN_TEST_VALIDATION = "CROSS" ]]; then
    startline=`grep -n "Overall Results" ${LOG_RESULTS} |cut -d":" -f 1`
    lastline=`cat ${LOG_RESULTS} |wc -l`
    numlines=`echo $lastline - $startline + 1| bc -l`
    tail -n $numlines ${LOG_RESULTS}
fi

# generate an overall HResults w/ confusion matrix for leave-one-out
if [[ $TRAIN_TEST_VALIDATION = "LEAVE_ONE_OUT" ]] || [[ $TRAIN_TEST_VALIDATION = "REPEAT_CROSS" ]] || [[ $TRAIN_TEST_VALIDATION = "K_FOLD" ]]; then
	rm -f ${BASE_OUTPUT_MLF}-all
	for i in ${BASE_OUTPUT_MLF}*; do
		echo $i
		cat $i >> ${BASE_OUTPUT_MLF}-all
	done
	echo "==========================================================" >> ${LOG_RESULTS}
	echo "OVERALL RESULTS" >> ${LOG_RESULTS}
	echo "==========================================================" >> ${LOG_RESULTS}

	${HTKBIN}HResults -A -e "???" sil -T $TRACE_LEVEL -t -I $MLF_LOCATION_ORIGINAL \
		-p $TOKENS_ORIGINAL ${BASE_OUTPUT_MLF}-all >> $LOG_RESULTS

	if [[ $WORD_LEVEL = "yes" ]] || [[ $WORD_LEVEL = "1" ]]; then
		rm -f ${BASE_OUTPUT_MLF_WORD}-all
		for i in ${BASE_OUTPUT_MLF_WORD}*; do
			echo $i
			cat $i >> ${BASE_OUTPUT_MLF_WORD}-all
		done
		echo "==========================================================" >> ${LOG_RESULTS_WORD}
		echo "OVERALL RESULTS" >> ${LOG_RESULTS_WORD}
		echo "==========================================================" >> ${LOG_RESULTS_WORD}
		${HTKBIN}HResults -A -e "???" sil -e "???" _ -T $TRACE_LEVEL -t -I $MLF_LOCATION_WORD \
			$TOKENS_WORD ${BASE_OUTPUT_MLF_WORD}-all >> ${LOG_RESULTS_WORD}
	fi
fi
	
# EOF
