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
PRJ=`pwd`					# path to the current project
						#
SCRIPTS_DIR=$PRJ/scripts			# location of scripts directory
						# for this project.
						#
UTIL_DIR=/usr/local/gt2k/utils			# location of utils directory
						#
						#
VECTOR_LENGTH=17				# number of elements in your
						# feature vector. This is the
						# number of observations per
						# state for the HMMs.
						#
MIN_VARIANCE=0.001				# don't let the
						# variance fall below
						# this value during
						# HMM training
HMM_TOPOLOGY_DIR=${PRJ}/hmmdefs

# general HMM_TOPOLOGIES
HMM_LOCATION=$HMM_TOPOLOGY_DIR/4state-1skip-17vec	#text hmm topology file
HMM_ALL=$HMM_LOCATION

# specific HMM_TOPLOGIES
#
#HMM_TOKEN_1=$HMM_TOPOLOGY_DIR/5_state_2-4skip_loopback_3vec
#HMM_TOKEN_2=$HMM_TOPOLOGY_DIR/7state_logical__9state_actual__noskip_3vec
#HMM_TOKEN_3=$HMM_TOPOLOGY_DIR/5state_noskip_3vec


# whether or not to initialize the starting model in a generic way:
INITIALIZE_HMM=yes                              # if you have a good initial
                                                # guess at your model as your
                                                # starting HMM, say no here.
                                                # otherwise, it is better
                                                # to let HTK initialize for you

GEN_TRAIN_TEST=yes                              # whether or not to generate
                                                # new test/train sets. if
                                                # you have made your own
                                                # or wish to reuse old sets,
                                                # set this to no.  otherwise
                                                # yes.

TRAIN_TEST_VALIDATION="CROSS"   		# "CROSS" or "LEAVE_ONE_OUT"
#TRAIN_TEST_VALIDATION="LEAVE_ONE_OUT"		# type of training/testing to
					    	# perform: cross-validation or
						# leave_one_out validation
					    	#
DATAFILES_LIST=${PRJ}/datafiles			# list of all data files
GRAMMARFILE=${PRJ}/grammar			# the grammar definition
DICTFILE=${PRJ}/dict
TOKENS=${PRJ}/commands				# list of grammar tokens
						#
						#
MLF_LOCATION=${PRJ}/labels.mlf			# master label file
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

GEN_GRAMMAR=no					# yes or no: generate grammar
						# and dict files using the
						# specified GRAMMAR_PROG program

GRAMMAR_PROG=${UTIL_DIR}/create_grammar.pl	# program to create a simple 
						# grammar and dict from a list
						# of commands

OUTPUT_MLF=${EXT_DIR}/result.mlf		# where HTK stores results
						# must be in the same dir as
						# .ext files
LOG_RESULTS=${PRJ}/hresults.log
						#
AUTO_ESTIMATE=yes				# Allow HTK to estimate gesture
						# boundaries for data with
						# multiple gestures per file.
						# Turning this off speeds up
						# training.
NUM_HMM_DIR=8					# number of hmm dirs to
						# generate, has a direct
						# relation to number of times
						# HERest is called
						#
HMM_TEMP_DIR=${PRJ}/models                      # directory for storing
                                                # intermediate models during
                                                # iterations of training

HMM_TRAINING=${HMM_TEMP_DIR}/hmm                # base name for iterations of
                                                # HMM training.  will be a
                                                # a directory with .# appended
                                                # to it where # is the
                                                # iteration of HERest
                                                #
# WARNING: files in directory with this         #
#          basename will be erased!!            #
#
#  rm -f $TRAINING_BASENAME*
TRAINING_DIR=${PRJ}/trainsets
TRAINING_BASENAME="${TRAINING_DIR}/training-extfiles"   # all lists of training>
                                                # will be named this with an
                                                # index number appended to it.
# WARNING: files in directory with this         #
#          basename will be erased!!            #
#
#  rm -f $TEST_BASENAME*
TESTING_DIR=${PRJ}/testsets
TESTING_BASENAME="${TESTING_DIR}/testing-extfiles"      # all lists of testing >
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

