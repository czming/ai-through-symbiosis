#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

##############################################################################
#
# USER MODIFICATION SECTION -- user specific files
#
##############################################################################
						######      Comments     #####
						##############################
						#
PRJ=`pwd`	# path to the current project
SCRIPTS_DIR=$PRJ/scripts			# location of scripts directory
						# for this project.
						#
UTIL_DIR=/gt2k/utils			# location of utils directory
						#
						#
VECTOR_LENGTH=1				# number of elements in your
						# feature vector. This is the
						# number of observations per
						# state for the HMMs.
						#
MIN_VARIANCE=0.01				# don't let the
						# variance fall below
						# this value during
						# HMM training

INSERT_PENALTY=0		#Penalize model for too many word insertion/deletion
						#If too many deletions, increase
						#If too many insertions, decrease

GRAMMAR_SCALE_FACTOR=0

#PRUNING_THRESHOLD="50 50 500" #Threshold for alpha-beta pruning, of form "start step-size end"
PRUNING_THRESHOLD=0

HMM_TOPOLOGY_DIR=${PRJ}/hmmdefs

# general HMM_TOPOLOGIES
HMM_LOCATION=$HMM_TOPOLOGY_DIR/6state-1dims	#text hmm topology file
HMM_ALL=$HMM_LOCATION

# whether or not to initialize the starting model in a generic way:
INITIALIZE_HMM=yes				# if you have a good initial
						# guess at your model as your
						# starting HMM, say no here.
						# otherwise, it is better
						# to let HTK initialize for you
						#
GEN_TRAIN_TEST=yes				# whether or not to generate
						# new test/train sets. if
						# you have made your own
						# or wish to reuse old sets,
						# set this to no.  otherwise
						# yes.

WORD_LEVEL=no # whether to process data as word level or letter level
TRILETTER=no # whether to enable triletter configuration
CROSS_WORD=no # whether triletters should expand across words  TODO

FORCE_ALIGN=yes # Use to enable/disable forced alignment during training
EXPORT_MLF=no # Use to export MLF for use outside project

NUM_HMM_DIR=20 # number of hmm dirs to generate, has a direct relation to number of times HERest is called
TRI_ITERATIONS=3 # number of HERest calls to make for triletter stages

TRAIN_TEST_VALIDATION="TEST_ON_TRAIN"
#TRAIN_TEST_VALIDATION="K_FOLD"
#TRAIN_TEST_VALIDATION="REPEAT_CROSS"
#TRAIN_TEST_VALIDATION="CROSS"
#TRAIN_TEST_VALIDATION="LEAVE_ONE_OUT"

#SORT_OPTION="-V" #Use this for alphabetic data order and sampling
SORT_OPTION="-R" # Use this for randomized data order and sampling
	
VALIDATION_ITERATIONS=10 #Number of repeats or folds

DATAFILES_LIST=${PRJ}/datafiles			# list of all data files

GRAMMARFILE=${PRJ}/grammar/grammar_letter_isolated_ai_general			# the grammar definition
GRAMMARFILE_WORD=${PRJ}/grammar/grammar_word_isolated_ai_general

DICTFILE=${PRJ}/dict/dict_letter2letter_ai_general # Use this for monoletter
#DICTFILE=${PRJ}/dict/dict_tri2letter #Use this for triletters 
DICTFILE_WORD=${PRJ}/dict/dict_letter2word_ai_general
#DICTFILE_WORD=${PRJ}/dict/dict_tri2word
DICTFILE_ALIGN=${PRJ}/dict/dict_tri2tri # Dictionary used during forced alignment

TOKENS_ORIGINAL=${PRJ}/commands/commands_letter_isolated_ai_general # used for building model
TOKENS=${PRJ}/commands/commands_letter_isolated_ai_general  #tri_internal letter_isolated letter 
#TOKENS=${PRJ}/commands/commands_tri_internal
TOKENS_WORD=${PRJ}/commands/commands_word_isolated_ai_general

MLF_LOCATION_ORIGINAL=${PRJ}/mlf/labels.mlf_letter # used for building model and results
MLF_LOCATION=${PRJ}/mlf/labels.mlf_tri_internal
MLF_LOCATION_WORD=${PRJ}/mlf/labels.mlf_word

MLF_LOCATION_GEN=${PRJ}/mlf/gen # Generated MLFs

WORD_LATTICE=${PRJ}/word.lattice

HEDFILE1=${PRJ}/mktri1.hed
HEDFILE2=${PRJ}/mktri2.hed
STATS=${PRJ}/stats
						#
						#
GEN_EXT_FILES=yes				# yes or no: generate .ext data
						# files (say yes unless they
						# have already been generated!

PREPARE_DATA=${UTIL_DIR}/prepare		# program for creating HTK-
						# readable data from text

EXT_DIR=${PRJ}/ext				# This is where HTK will put
						# .ext files it generates
						#
GEN_GRAMMAR=no				# yes or no: generate grammar
						# and dict files using the
						# specified GRAMMAR_PROG program

GRAMMAR_PROG=${UTIL_DIR}/create_grammar.pl      # program to create a simple 
						# grammar and dict from a list
						# of commands

OUTPUT_MLF=${EXT_DIR}/result.mlf_letter		# where HTK stores results
						# must be in the same dir as
						# .ext files
OUTPUT_MLF_WORD=${EXT_DIR}/result.mlf_word

LOG_RESULTS=${PRJ}/hresults.log_letter
LOG_RESULTS_WORD=${PRJ}/hresults.log_word

HMM_TEMP_DIR=${PRJ}/models			# directory for storing
						# intermediate models during
						# iterations of training

HMM_TRAINING=${HMM_TEMP_DIR}/hmm		# base name for iterations of
						# HMM training.  will be a 
						# a directory with .# appended
						# to it where # is the
						# iteration of HERest 
						#
# WARNING: files in directory with this		#
# 	   basename will be erased!!		#
#
#  rm -f $TRAINING_BASENAME*
TRAINING_DIR=${PRJ}/trainsets
TRAINING_BASENAME="${TRAINING_DIR}/training-extfiles"	# all lists of training files
						# will be named this with an
					    	# index number appended to it.
# WARNING: files in directory with this		#
# 	   basename will be erased!!		#
#
#  rm -f $TEST_BASENAME*
TESTING_DIR=${PRJ}/testsets
TESTING_BASENAME="${TESTING_DIR}/testing-extfiles"	# all lists of testing files
						# will be named this with an
					    	# index number appended to it.
						#
TRACE_LEVEL=1					# level of debugging
						#
${HTKBIN=}					# check to see if the path of
	#example:  ${HTKBIN=/usr/local/bin/}	# HTK is set as an environment
    						# variable if not, then use the
						# specified location.  
					    	# now it is set to NULL which
						# means that it will look in 
				   		# your path if left this way.
						# Be sure to include the 
						# trailing slash!
						#
PROMPT_B4_RM="no"				# Prompt before removing files
						# that exist.  can be "yes",
						# "no", or "".  if the value
						# is "no" or "" then files will
						# overwritten without checking
