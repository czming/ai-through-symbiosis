#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

##############################################################################
# check to make sure that the user did not specify something incorrectly
# in options.sh and echo out proper values to the user.
# 
# this is called by train.sh after it loads options.sh
##############################################################################

## some variables to make things easier with the menus
OVERWRITE="Overwrite File"
RENAME="Rename existing file"
ABORT="Abort script"

# make sure that vector size is a number between the ranges of 1 and 2000
if [[ -z ${VECTOR_LENGTH}  ]]    ||  
   [[ ${VECTOR_LENGTH} -gt  2000 ]] || 
   [[ ${VECTOR_LENGTH} -lt  1 ]]; then
   echo
   echo ERROR: Trace Level ${VECTOR_LENGTH} is invalid. Range must be set;
   echo between 1 and 2000 inclusive.
   echo
   exit; 
fi

echo data vector has size ${VECTOR_LENGTH}


# make sure ${HMM_LOCATION} is an actual file that exists and make sure
# we have read permisions to that file
if [[ -a ${HMM_LOCATION} ]] && [[ -r ${HMM_LOCATION} ]]; then
    echo Reading HMM definition from ${HMM_LOCATION}; 
else
    echo
    echo ERROR: HMM definition file: ${HMM_LOCATION} is invalid.; 
    echo Check that ${HMM_LOCATION} exists and that you have permissions;
    echo to read to the file; 
    echo
    exit; 
fi

# make sure ${HMM_LOCATION} is an actual file that exists and make sure
# we have read permisions to that file
if [[ -a ${MLF_LOCATION} ]] && [[ -r ${MLF_LOCATION} ]]; then
    echo Reading labels from master label file ${MLF_LOCATION}; 
else
    echo;
    echo ERROR: Master Label file: ${MLF_LOCATION} is invalid.; 
    echo Check that ${MLF_LOCATION} exists and that you have permissions;
    echo to read to the file; 
    echo;
    exit; 
fi

# make sure ${OUTPUT_MLF} does not exist.  if it does make sure the user
# doesn't care if we overwrite it and that we have write permisions to 
# that file
 if [[ -a ${OUTPUT_MLF} ]] && [[ ${PROMPT_B4_RM} = "yes" ]]; then
    echo
    echo ${OUTPUT_MLF} exists what would you like to do?;
    echo
    PS3='choice? '
     select choice in "${OVERWRITE}" "${ABORT}"
     do
         if [[ -n $choice ]]; then
            CHOICE=$choice
            break
         else
            echo invalid choice
         fi
     done
     

         if [[ ${CHOICE} = "${ABORT}" ]]; then
	    echo Exiting ... ;
            exit;
	 fi

fi

    echo Writing results to the file ${OUTPUT_MLF}; 
     

# make sure ${UTIL_DIR} is an actual directory that exists and make sure
# we have read permissions in that directory
if [[ -d ${UTIL_DIR} ]] && [[ -r ${UTIL_DIR} ]]; then
    echo Utilities located in directory: ${UTIL_DIR}; 
else
    echo;
    echo ERROR: Utility directory: ${UTIL_DIR} is invalid.; 
    echo Check that ${UTIL_DIR} exists and that you have permissions;
    echo to read files from it; 
    echo;
    exit; 
fi

# make sure that the trace level is non-NULL and that it is a number between
# [1,7].  If it is missing just set it to be 1
if [[ -z ${TRACE_LEVEL}  ]] || 
   [[ ${TRACE_LEVEL} -gt 7 ]] || 
   [[ ${TRACE_LEVEL} -lt 1 ]]; then
   echo;
   echo ERROR: Trace Level ${TRACE_LEVEL} is invalid. Range must be set;
   echo between 1 and 7 inclusive.
   echo;
   exit; 
fi

# give the user some useful comments on the data he just specified
echo Trace Level is set at ${TRACE_LEVEL}


###############################################################################
# make sure that the HTK executables are where they need to be.
###############################################################################

# if ${HTKBIN} is NULL then the which command will find the executables
# if they are somewhere in $PATH.  if ${HTKBIN} is something not in the
# path, then which will just echo the path if it is a valid path 
HEREST=`which ${HTKBIN}HERest`
HPARSE=`which ${HTKBIN}HParse`
HCOMPV=`which ${HTKBIN}HCompV`
HERESULTS=`which ${HTKBIN}HResults`

# test to see if any of these string are null ( ie, not a valid path )
if [[ -z ${HEREST}     ]] || 
   [[ -z ${HPARSE}     ]] || 
   [[ -z ${HCOMPV}     ]] || 
   [[ -z ${HERESULTS}  ]];   then 

   echo;
   echo ERROR: Some of the required HTK executables were not in ${HTKBIN};
   echo please make sure that HTKBIN is set in either the environment; 
   echo or in the script file or that the path to the HTK executables;
   echo are in your PATH variable.
   echo; 
   exit; 
fi


# test to make sure that we have execute permisions on these files
if [[ -x ${HEREST}     ]] && 
   [[ -x ${HPARSE}     ]] && 
   [[ -x ${HCOMPV}     ]] && 
   [[ -x ${HERESULTS}  ]];   then 

   echo HTK executables located: ${HEREST} ${HPARSE} ${HCOMPV} ${HERESULTS};
else
   echo;
   echo ERROR: you do not have execute permissions on some of the required;
   echo HTK executables; please run "chmod +x" on:;
   echo HTK executables ${HEREST} ${HPARSE} ${HCOMPV} ${HERESULTS}; 
   echo; 
   exit; 
fi


## check to see that creat_grammar exist and that we have execute permissions
## on it.

## check to see that the ${DATAFILES_LIST} exists and that we have read permissions
## on it.
if [[ -f ${DATAFILES_LIST} ]] &&
   [[ -r ${DATAFILES_LIST} ]]; then
   echo Using data files listed in ${DATAFILES_LIST};
else
   echo;
   echo "ERROR: Cannot read data file list \"${DATAFILES_LIST}\"!";
   echo Please make sure this files exists and is readable.
   echo;
   exit;
fi

## check to see that the prepare script exists and that we have execute
## permissions on it

if [[ -f ${UTIL_DIR}/prepare ]] &&
   [[ -x ${UTIL_DIR}/prepare ]]; then
   echo Using ${UTIL_DIR}/prepare to prepare data for HTK;
else
   echo;
   echo "ERROR: Data prep program ${UTIL_DIR}/prepare not found or not"
   echo "executable.  Please make sure UTIL_DIR is set correctly in"
   echo "$OPTIONS_FILE and that the prepare program has been compiled by"
   echo "running 'make' in ${UTIL_DIR}"
   exit;
fi

## check and make sure that it is ok to remove  the ext files

## check and make sure that the ext directory exists and that we have
## permisions to read and write to the directory

# if GEN_GRAMMAR is set to no, make sure the grammar and dict files exist
typeset -l GEN_GRAMMAR
if [[ "${GEN_GRAMMAR}" == "no" ]] ||
   [[ "${GEN_GRAMMAR}" == "0" ]]; then
	if [[ ! -f "${GRAMMARFILE}" ]] || 
	   [[ ! -f "${DICTFILE}" ]]; then
		echo;
		echo "ERROR: Told not to generate ${GRAMMARFILE} and ${DICTFILE}"
		echo "but these files do not exist.  Please create them as"
		echo "desired, or set GEN_GRAMMAR=yes in scripts/options.sh"
		exit;
	fi
fi

# if GEN_TRAIN_TEST is set to no, make sure training or testing files exist
typeset -l GEN_TRAIN_TEST
if [[ "${GEN_TRAIN_TEST} == "no" ]] ||
   [[ "${GEN_TRAIN_TEST} == "0" ]]; then
	if [[ -z "`ls ${TRAINING_BASENAME}* 2> /dev/null`" ]] ||
	   [[ -z "`ls ${TESTING_BASENAME}* 2> /dev/null `" ]]; then
		echo;
		echo "ERROR: Told not to generate training/testing sets"
		echo "of the form ${TRAINING_BASENAME}* and ${TESTING_BASENAME}*"
		echo "but could find none.  Please generate these sets, or"
		echo "set GEN_TRAIN_TEST=yes in scripts/options.sh"
	fi
fi

# test for existence of HMM_TEMP_DIR, TRAINING_DIR, TESTING_DIR
if [[ ! -d "${HMM_TEMP_DIR}" ]]; then
	echo;
	echo "ERROR: Model directory '${HMM_TEMP_DIR}' doesn't exist"
	echo "Please create it!"
	exit
fi

if [[ ! -d "${TRAINING_DIR}" ]]; then
	echo;
	echo "ERROR: Training list directory '${TRAINING_DIR}' doesn't exist"
	echo "Please create it!"
	exit
fi

if [[ ! -d "${TESTING_DIR}" ]]; then
	echo;
	echo "ERROR: Testing list directory '${TESTING_DIR}' doesn't exist"
	echo "Please create it!"
	exit
fi




