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
# 		haileris@cc.gatech.edu
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
WORD_LATTICE=word.lattice

TT_NAME_SCRIPT=$SCRIPTS_DIR/gen_train_test_name.sh      # make consistent names

HMM_BASE_DIR=$HMM_TRAINING
BASE_OUTPUT_MLF=$OUTPUT_MLF

#clean up old training data
rm -f $LOG_RESULTS
rm -f $DATA_SAMPLES
rm -f $OUTPUT_MLF*
rm -f $WORD_LATTICE
for i in ${HMM_TRAINING}*\.*
do
	rm -f $i/*
	rmdir $i
done

# generate a list of all data samples HTK has avaliable to it.
#ls ${EXT_DIR}/*.ext > $DATA_SAMPLES
find ${EXT_DIR}/ | grep "\.ext$" > $DATA_SAMPLES

#create our word lattice
${HTKBIN}HParse ${GRAMMARFILE} $WORD_LATTICE

MIN_CYCLES=1

# based on the TRAIN_TEST_VALIDATION variable select which script is to be
# used for generating the testing and training set of data.  This variable will
# also determine the number or times training and testing are exectuted
if [[ $TRAIN_TEST_VALIDATION = "CROSS" ]]; then

	TRAIN_TEST_SCRIPT=$SCRIPTS_DIR/gen_cross_val.sh	
	TEST_TRAIN_CYCLES=$MIN_CYCLES

elif [[ $TRAIN_TEST_VALIDATION = "LEAVE_ONE_OUT" ]]; then

	TRAIN_TEST_SCRIPT=$SCRIPTS_DIR/gen_leave_one_out.sh	
	TEST_TRAIN_CYCLES=`cat $DATA_SAMPLES | wc -l`
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
cycle=0; correct=0;
while [[ $cycle -lt $TEST_TRAIN_CYCLES ]]
do
    # generate filenames that are dependent on the specific iteration
    HMM_TRAINING=$HMM_BASE_DIR$cycle; 
    TRAINING=`$TT_NAME_SCRIPT $TRAINING_BASENAME $cycle`
    TESTING=`$TT_NAME_SCRIPT $TESTING_BASENAME $cycle`
    OUTPUT_MLF=$BASE_OUTPUT_MLF$cycle;

    ## generate the directories to store the iterations of HMM training
    hmm_count=0
    while [[ $hmm_count -lt $NUM_HMM_DIR ]]
    do
	mkdir $HMM_TRAINING.$hmm_count 
	hmm_count=$((hmm_count+1))
    done
echo
echo "*****************************************************"
echo Building Models
echo "*****************************************************"
###############################################################################
# now lets build our models 
###############################################################################
# HCompV fills in our mean and variances on the HMM model provided
# -m causes mean evaluation
# -S is the list of EXT files it should use
# -l is the segment label - corresponds to the word we are training on 
# -I is the MLF (Master Label File) - should contain the word list for each ext
#    file 
# -o the label for our output HMM, should be the word its trained on
# -m The covariances of the output HMM are always updated however updating the
#    means must be specifically requested. When this option is set, HCOMPV 
#    updates all the HMM component means with the sample mean computed from 
#    the training files. 
# -M is the directory to store output HMM 
#    (if not given will overwrite the HMMs)
# parameters= HHM file to start with
###############################################################################
for n in $(cat ${TOKENS});
  do


    ## somtimes it works better if you use different topologies for different
    ## models.
    ##
    ## below is an example of how to integrate multiple topologies
#      if [[ $n = "token1" ]]; then
#  	HMM_LOCATION=$HMM_TOKEN_1

#      elif [[ $n = "token2" ]]; then
#  	HMM_LOCATION=$HMM_TOKEN_2
	
#      elif [[ $n = "token3" ]]; then
#  	HMM_LOCATION=$HMM_TOKEN_3
#      else
#  	HMM_LOCATION=$HMM_ALL
#      fi


	typeset -l INITIALIZE_HMM	# make sure it is all lowercase
	if [[ "${INITIALIZE_HMM}" == "yes" ]] ||
	   [[ "${INITIALIZE_HMM}" == "1" ]]; then
		${HTKBIN}HCompV -A -T $TRACE_LEVEL -S $TRAINING -l $n 	\
				-I $MLF_LOCATION -o $n -m -M $HMM_TRAINING.0  	\
				$HMM_LOCATION 

	  	${HTKBIN}HInit  -A -T $TRACE_LEVEL -M $HMM_TRAINING.1 -l $n 	\
			        -S $TRAINING -I $MLF_LOCATION -o $n 	\
				$HMM_TRAINING.0/$n 
	else
		cp $HMM_LOCATION $HMM_TRAINING.2/
	fi

	${HTKBIN}HRest  -A  -m 1 -T 1 -t -i 30 -v ${MIN_VARIANCE}  -l $n \
			-M $HMM_TRAINING.2/ -S $TRAINING 	\
			-I $MLF_LOCATION $HMM_TRAINING.1/$n 

done

echo
echo "*****************************************************"
echo Training Models
echo "*****************************************************"
###############################################################################
# now we train our models
###############################################################################
# HERest updates all of the HMM parameters, that is, means,
# variances, mixture weights and transition probabilies. 
#
# -S is the list of EXT files it should use
# -I is the MLF (Master Label File) - should contain the word list for each ext
# 		file 
# -d this tells where to look for the HMMs
# -M Store output HMM macro model files in the directory dir 
#		(if not given will overwrite the HMMs)
# parameters= hmms to train on (should be our words)
# -m minimum number of training examples for a model
# -o replace file extentions by .ext
# -v f This sets the minimum variance (i.e. diagonal element of the
# covariance matrix) to the real value f (default value 0.0). 
###############################################################################


## TLW --> if there is more then one command then this should be newMacros
## TLW --> if there is only one command, then newMacros won't be generated
##	   and we need to use macros named by the commands, in this case
##	   set HMM_MACRO to the empty string, ""
HMM_MACRO="newMacros"
HMM_LOAD_OPT="-H"

## TLW --> added to account for HMM_MACRO
## if HMM_MACRO is the default macro "newMacro" then it must be loaded with
## a -H option.  if it is based on the command names, then it needs to be 
## loaded with -d
if [[ -z ${HMM_MACRO} ]]; then
    HMM_LOAD_OPT="-d";
else
    HMM_MACRO="newMacros";
    HMM_LOAD_OPT="-H";
    
fi

## first instance of HERest should be run with the -d option
${HTKBIN}HERest -v $MIN_VARIANCE \
		-A -T $TRACE_LEVEL -S $TRAINING -d $HMM_TRAINING.2/ \
		-M $HMM_TRAINING.3 -I $MLF_LOCATION ${TOKENS} 


## run $NUM_HMM_DIR training iterations over the hmm model.  each iteration
## is stored in a directory $HMM_TRAINING.# where # corresponds to the 
## training iteration.  HERest will be called on hmm.n and stored in hmm.n+1
hmm_count=3

typeset -l AUTO_ESTIMATE	# make sure it is all lowercase
if [[ "$AUTO_ESTIMATE" == "yes" ]] || 
   [[ "$AUTO_ESTIMATE" == "1" ]]; then

    last_iteration=$((NUM_HMM_DIR-1))
    next_dir=$last_iteration
    while [[ $hmm_count -lt $last_iteration ]]
    do
    	next_dir=$((hmm_count+1))

    	echo
    	${HTKBIN}HERest -v $MIN_VARIANCE \
		    -A -T $TRACE_LEVEL -S $TRAINING		  \
		    $HMM_LOAD_OPT $HMM_TRAINING.$hmm_count/$HMM_MACRO 	  \
		    -M $HMM_TRAINING.$next_dir -I $MLF_LOCATION ${TOKENS}

    	hmm_count=$((hmm_count+1))
    done
else
    next_dir=$hmm_count
fi


echo
echo "*****************************************************"
echo Checking our Models
echo "*****************************************************"
###############################################################################
# now we check our models
###############################################################################
# -H is the HMM to load
# -S is the list of EXT files it should use
# -I is the MLF (Master Label File) - should contain the word list for each ext
# 		file 
# -i is the MLF file to store output to
# -a load a label file and create an alignment network for each test file.
# -n use 'i' tokens to perform N-best recognition.
# parameters= dictionary file
# parameters= hmms to use (should correspond to our words)
###############################################################################
${HTKBIN}HVite  -A -T $TRACE_LEVEL 					\
		$HMM_LOAD_OPT $HMM_TRAINING.$next_dir/$HMM_MACRO 	\
		-w word.lattice -S $TESTING -I $MLF_LOCATION 	\
		-i $OUTPUT_MLF -n 4 20 dict $TOKENS 

# confidence levels
#HVite -H hmm.7/newMacros -w word.lattice -S $TESTING -I labels.mlf -o output.mlf -n 4 20 dict commands


echo
echo "*****************************************************"
echo Testing Models
echo "*****************************************************"
###############################################################################
# now we run the tests
###############################################################################
# -t This option causes a time-aligned transcription of each test file to be
#    output provided that it differs from the reference transcription file
# -I is the MLF (Master Label File) - should contain the word list for each ext
#    file 
# -p This option causes a phoneme confusion matrix to be output.
# -w outputs ROC info that doesn't look quite correct
# -d N : if correct answer is within the top N-Best consider it correctly
#	 classified
# parameters= MLF file to load
############################################################################## 
${HTKBIN}HResults -A -T $TRACE_LEVEL -t -I $MLF_LOCATION 		\
		  -p $TOKENS $OUTPUT_MLF >> $LOG_RESULTS

    ## ***** HACK-O-LICIOUS ********
    ##
    ## if we are running more then once, then keep track of the results from
    ## multiple outputs
    ##		  
    if [[ $TEST_TRAIN_CYCLES -gt $MIN_CYCLES ]]; then
    ## HACK!!!!
    ## run again, but this time grep the output for 100% correct. keep
    ## a running count of how many times this occurs.
	result_str=`${HTKBIN}HResults -A -T $TRACE_LEVEL -I $MLF_LOCATION  \
		  $TOKENS $OUTPUT_MLF | grep %Corr=100.00` 

	## if the string is non-empty, then our test correctly classified
	if [[ -n $result_str  ]]; then
	    correct=$((correct+1))
	fi
    fi

    # update the cycle for the next iteration
    cycle=$((cycle+1))

done  # matches the "while" cycles of training


# print the results of HResults of CROSS_VALIDATION
if [[ $TRAIN_TEST_VALIDATION = "CROSS" ]]; then
    startline=`grep -n "Overall Results" hresults.log |cut -d":" -f 1`
    lastline=`cat hresults.log |wc -l`
    numlines=`echo $lastline - $startline + 1| bc -l`
    tail -n $numlines hresults.log
fi

# print statistics on correct classification if more then one iteration
# was run
    if [[ $TEST_TRAIN_CYCLES -gt $MIN_CYCLES ]]; then
	
	   echo classified $correct of $cycle correctly
	   echo `echo "($correct/$cycle) * 100" | bc -l` % correct
    fi

# generate an overall HResults w/ confusion matrix for leave-one-out
if [[ $TRAIN_TEST_VALIDATION = "LEAVE_ONE_OUT" ]]; then
	rm -f ${OUTPUT_MLF}-all
	for i in ${OUTPUT_MLF}*
		do
		cat $i >> ${OUTPUT_MLF}-all
	done
	echo "==========================================================" >> ${LOG_RESULTS}
	echo "OVERALL RESULTS" >> ${LOG_RESULTS}
	echo "==========================================================" >> ${LOG_RESULTS}
	${HTKBIN}HResults -A -T $TRACE_LEVEL -t -I $MLF_LOCATION \
		-p $TOKENS ${OUTPUT_MLF}-all >> $LOG_RESULTS
fi
	
# EOF
