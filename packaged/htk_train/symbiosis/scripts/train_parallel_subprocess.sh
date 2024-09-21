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
cycle=$2
. ${OPTIONS_FILE}

DATA_SAMPLES=all-extfiles

TT_NAME_SCRIPT=$SCRIPTS_DIR/gen_train_test_name.sh      # make consistent names

HMM_BASE_DIR=$HMM_TRAINING
BASE_OUTPUT_MLF=$OUTPUT_MLF
BASE_OUTPUT_MLF_WORD=$OUTPUT_MLF_WORD
BASE_MLF_LOCATION_GEN=$MLF_LOCATION_GEN

# based on the type of training/testing validation, iterate through the
# training process

# generate filenames that are dependent on the specific iteration
HMM_TRAINING=$HMM_BASE_DIR$cycle; 
TRAINING=`$TT_NAME_SCRIPT $TRAINING_BASENAME $cycle`
TESTING=`$TT_NAME_SCRIPT $TESTING_BASENAME $cycle`
OUTPUT_MLF=$BASE_OUTPUT_MLF$cycle;
OUTPUT_MLF_WORD=$BASE_OUTPUT_MLF_WORD$cycle;
MLF_LOCATION_GEN=${BASE_MLF_LOCATION_GEN}/${cycle};

mkdir ${BASE_MLF_LOCATION_GEN}/${cycle}

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
typeset -l INITIALIZE_HMM	# make sure it is all lowercase
if [[ "${INITIALIZE_HMM}" == "yes" ]] ||
   [[ "${INITIALIZE_HMM}" == "1" ]]; then
	for n in $(cat ${TOKENS_ORIGINAL}); do
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
		${HTKBIN}HCompV -A -T $TRACE_LEVEL -v ${MIN_VARIANCE} -S $TRAINING -l $n 	\
				-I $MLF_LOCATION_ORIGINAL -o $n -m -M $HMM_TRAINING.0  	\
				$HMM_LOCATION 

	  	${HTKBIN}HInit  -A -T $TRACE_LEVEL -v ${MIN_VARIANCE} -M $HMM_TRAINING.1 -l $n 	\
		        -S $TRAINING -I $MLF_LOCATION_ORIGINAL -o $n 	\
				$HMM_TRAINING.0/$n 

		${HTKBIN}HRest  -A  -m 1 -T 1 -t -i 30 -v ${MIN_VARIANCE}  -l $n \
			-M $HMM_TRAINING.2/ -S $TRAINING 	\
			-I $MLF_LOCATION_ORIGINAL $HMM_TRAINING.1/$n 
	done
else
	cp $HMM_LOCATION $HMM_TRAINING.3/newMacros
fi

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

if [[ "${INITIALIZE_HMM}" == "yes" ]] || [[ "${INITIALIZE_HMM}" == "1" ]]; then
	## first instance of HERest should be run with the -d option
	${HTKBIN}HERest -v $MIN_VARIANCE \
			-A -T $TRACE_LEVEL -S $TRAINING -d $HMM_TRAINING.2/ \
			-M $HMM_TRAINING.3 -I $MLF_LOCATION_ORIGINAL ${TOKENS_ORIGINAL}
fi



## run $NUM_HMM_DIR training iterations over the hmm model.  each iteration
## is stored in a directory $HMM_TRAINING.# where # corresponds to the 
## training iteration.  HERest will be called on hmm.n and stored in hmm.n+1
hmm_count=3

last_iteration=$((NUM_HMM_DIR-1))
if [[ $TRILETTER = "yes" ]] || [[ $TRILETTER = "1" ]]; then
	last_iteration=$((NUM_HMM_DIR-2*TRI_ITERATIONS-4))
fi

while [[ $hmm_count -lt $last_iteration ]]
do
	next_dir=$((hmm_count+1))

	${HTKBIN}HERest -v $MIN_VARIANCE \
	    -A -T $TRACE_LEVEL -S $TRAINING		  \
	    $HMM_LOAD_OPT $HMM_TRAINING.$hmm_count/$HMM_MACRO 	  \
	    -M $HMM_TRAINING.$next_dir -I $MLF_LOCATION_ORIGINAL ${TOKENS_ORIGINAL}

	hmm_count=$((hmm_count+1))
done

if [[ $TRILETTER = "yes" ]] || [[ $TRILETTER = "1" ]]; then
	last_iteration=$((NUM_HMM_DIR-TRI_ITERATIONS-3))

	next_dir=$((hmm_count+1))
	HHEd -A -T $TRACE_LEVEL $HMM_LOAD_OPT $HMM_TRAINING.$hmm_count/$HMM_MACRO -M $HMM_TRAINING.$next_dir $HEDFILE1 ${TOKENS_ORIGINAL}
	hmm_count=$((hmm_count+1))

    while [[ $hmm_count -lt $last_iteration ]]
    do
    	next_dir=$((hmm_count+1))

    	${HTKBIN}HERest -v $MIN_VARIANCE \
		    -A -T $TRACE_LEVEL -S $TRAINING		  \
		    $HMM_LOAD_OPT $HMM_TRAINING.$hmm_count/$HMM_MACRO 	  \
		    -M $HMM_TRAINING.$next_dir -I $MLF_LOCATION ${TOKENS}

    	hmm_count=$((hmm_count+1))
    done

    last_iteration=$((NUM_HMM_DIR-1))

    next_dir=$((hmm_count+1))
	${HTKBIN}HERest -v $MIN_VARIANCE \
		    -A -T $TRACE_LEVEL -S $TRAINING	-s $STATS	  \
		    $HMM_LOAD_OPT $HMM_TRAINING.$hmm_count/$HMM_MACRO 	  \
		    -M $HMM_TRAINING.$next_dir -I $MLF_LOCATION ${TOKENS}
	hmm_count=$((hmm_count+1))

	next_dir=$((hmm_count+1))
	HHEd -A -T $TRACE_LEVEL $HMM_LOAD_OPT $HMM_TRAINING.$hmm_count/$HMM_MACRO -M $HMM_TRAINING.$next_dir $HEDFILE2 ${TOKENS}
	hmm_count=$((hmm_count+1))

	# Force-align MLFs
	if [[ $FORCE_ALIGN = "yes" ]] || [[ $FORCE_ALIGN = "1" ]]; then
		${HTKBIN}HVite -a -b sil -p $INSERT_PENALTY -s $GRAMMAR_SCALE_FACTOR -m -o SW -A -T $TRACE_LEVEL \
			$HMM_LOAD_OPT $HMM_TRAINING.$next_dir/$HMM_MACRO \
			-S $DATA_SAMPLES -I $MLF_LOCATION -i ${MLF_LOCATION_GEN}/labels.mlf $DICTFILE_ALIGN $TOKENS 
		MLF_LOCATION=${MLF_LOCATION_GEN}/labels.mlf
		sed 's/.rec/.lab/g' ${MLF_LOCATION} > ${MLF_LOCATION}_temp
		mv ${MLF_LOCATION}_temp ${MLF_LOCATION}
	fi

    while [[ $hmm_count -lt $last_iteration ]]
    do
    	next_dir=$((hmm_count+1))

    	${HTKBIN}HERest -v $MIN_VARIANCE \
		    -A -T $TRACE_LEVEL -S $TRAINING		  \
		    $HMM_LOAD_OPT $HMM_TRAINING.$hmm_count/$HMM_MACRO 	  \
		    -M $HMM_TRAINING.$next_dir -I $MLF_LOCATION ${TOKENS}

    	hmm_count=$((hmm_count+1))
    done

    if [[ $EXPORT_MLF = "yes" ]] || [[ $EXPORT_MLF = "1" ]]; then
	    ${HTKBIN}HVite -p $INSERT_PENALTY -s $GRAMMAR_SCALE_FACTOR -m -o SWX -A -T $TRACE_LEVEL \
			$HMM_LOAD_OPT $HMM_TRAINING.$next_dir/$HMM_MACRO \
			-S $DATA_SAMPLES -I $MLF_LOCATION_ORIGINAL -i ${MLF_LOCATION_GEN}/labels.mlf_export $DICTFILE $TOKENS
	fi
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

${HTKBIN}HVite -p $INSERT_PENALTY -t $PRUNING_THRESHOLD -s $GRAMMAR_SCALE_FACTOR -A -T $TRACE_LEVEL 					\
	$HMM_LOAD_OPT $HMM_TRAINING.$next_dir/$HMM_MACRO 	\
	-w $WORD_LATTICE -S $TESTING -I $MLF_LOCATION 	\
	-i $OUTPUT_MLF $DICTFILE $TOKENS 

if [[ $WORD_LEVEL = "yes" ]] || [[ $WORD_LEVEL = "1" ]]; then
	${HTKBIN}HVite -p $INSERT_PENALTY -s $GRAMMAR_SCALE_FACTOR -A -T $TRACE_LEVEL 					\
		$HMM_LOAD_OPT $HMM_TRAINING.$next_dir/$HMM_MACRO 	\
		-w ${WORD_LATTICE}_word -S $TESTING -I $MLF_LOCATION 	\
		-i $OUTPUT_MLF_WORD -n 4 20 $DICTFILE_WORD $TOKENS 
fi

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

${HTKBIN}HResults -A -e "???" sil -T $TRACE_LEVEL -t -I $MLF_LOCATION_ORIGINAL \
 	-p $TOKENS_ORIGINAL $OUTPUT_MLF >> $LOG_RESULTS

if [[ $WORD_LEVEL = "yes" ]] || [[ $WORD_LEVEL = "1" ]]; then
	${HTKBIN}HResults -A -e "???" sil -e "???" _ -T $TRACE_LEVEL -t -I $MLF_LOCATION_WORD \
		$TOKENS_WORD $OUTPUT_MLF_WORD >> $LOG_RESULTS_WORD
fi
	
# EOF
