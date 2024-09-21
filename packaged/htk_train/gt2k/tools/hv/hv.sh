#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


###################################################################
## File   : hv.sh
## Author : Tracy L. Westeyn
## Email  : turtle@cc.gatech.edu
##
## Purpose: HTK HMM definition to GraphViz DOT vizualization
##
## Input  : param1 --> name of the HMM definition file to vizualize
##
##
## NOTES  : Several executables are required:
##
##		dot 	( a component of GraphViz )
##		gv  	( a PostScript viewer     )
##		htk2dot ( included executable     )
##  
##
## BUGS	   : Currently this script can only handle one file at a time 
##   	     it cannot take in multiple files and generate the
##	     the postscript for them all. 	
####################################################################

# set the input parameter
HMM_DEF_FILE=$1

# make sure an input file was specified.  if HMM_DEF_FILE is null 
# print an error and exit
if [[ -z ${HMM_DEF_FILE} ]]; then 
    print;
    print ERROR: input parameter missing.  Must Specify an HMM definition file; 
    print;
    exit; 
fi

# make sure the parameter passed in is an actual file that exists and make sure
# we have read permisions to that file
if [[ -a ${HMM_DEF_FILE} ]] && [[ -r ${HMM_DEF_FILE} ]]; then
    print Reading data from ${HMM_DEF_FILE}; 
else
    print;
    print ERROR: input parameter is invalid. Check that ${HMM_DEF_FILE};  
    print exists and that you have write permissions to the file; 
    print;
    exit; 
fi


# make sure the user does not have a file with the same name as our
# temporary filename.  we do not want over write their data
TMP_FILE="hv_tmp_file.ps"

if [[ -a ${TMP_FILE} ]]; then
    print;
    print WARNING: the file ${TMP_FILE} will be removed if this script is run.;
    print please rename or move ${TMP_FILE} to another directory while using;
    print this script;
    print;
    exit;
fi

# Find the location of necessary software
print Checking system for necessary software;

GV=`which gv`
DOT=`which dot`
HTK2DOT="./htk2dot"

# Check to see if the necessary software is installed / in $PATH
if [[ -z ${GV} ]]; then 
    print;
    print ERROR: gv not found in ${PATH}.  
    print please add gv to your path or install it if it is not on your
    print system; 
    print;
    exit; 
fi

# Check to see if the necessary software is installed / in $PATH
if [[ -z ${DOT} ]]; then 
    print;
    print ERROR: dot not found in ${PATH}.  
    print please add dot to your path or install it if it is not on your
    print system; 
    print;
    exit; 
fi

# Check to see if the necessary software is installed / in current directory
if [[ -a ${HTK2DOT} ]]; then
    print All necessary components found. Generating the visualization.;
else
    print;
    print ERROR: ${HTK2DOT} not found in ${PWD}.  
    print please type \"make\" to build the program.
    print;
    exit; 
fi



## given an HMM definition file generate the graphviz-DOT compliant syntax file
## and use DOT to generate a postscript file.  view the file in gv
${HTK2DOT} < ${HMM_DEF_FILE} | ${DOT} -Tps > ${TMP_FILE}
${GV} ${TMP_FILE}

## remove the temporary file we just made
rm ${TMP_FILE} 