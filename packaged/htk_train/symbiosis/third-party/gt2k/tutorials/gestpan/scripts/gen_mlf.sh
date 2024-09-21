#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


## generate the Master Label File for a specified set of data files.
## the first argument is a file containing paths to the datafiles.
## the second argument is the path to where the HTK_readable versions of the
## datafiles will be placed (the EXT dirctory)
##
## labels must also be provided for the datafiles.  it is assumed that the
## label for the data is embedded in the filename.  This label is extracted
## by calling the script 'extract_label.sh' on each datafile name.
##
## example usage: scripts/gen_mlf.sh fileslist ext/ > labels.mlf
##
##
## NOTE: this script assumes that each datafile contains only one example of
##       one gesture.  

DATA_LIST_FILE=$1 
EXT_DIR=$2
EXTRACT_LABELS=scripts/extract_label.sh		# script to extract data labels


integer frame_duration=2000           # each frame is about 2000ns in HTK
integer start_time=0           		
integer end_time=0           		
integer num_lines=0

echo "#!MLF!# \n"			# write the header for the file

for m in `cat $DATA_LIST_FILE`; do	# for each data file listed
   
#    if ! [ -d "$EXT_DIR/$m" ]; then	# mirror the data directory hierarchy
#    	mkdir -p "$EXT_DIR/$m"		# in the ext output
#    fi

	# see if we need to append a path to this
	echo "${EXT_DIR}" | grep '^/' > /dev/null
	if [ "$?" = "0" ]; then
	   labname="\"${EXT_DIR}/$m.lab\""  #search for HTK readable datafiles
	else
	   labname="\"`pwd`/${EXT_DIR}/$m.lab\""
	fi
	echo "$labname" | sed -e "s|//|/|g"  # output the label filename
	num_lines=`cat $m | wc -l` #   compute the num lines per file
	end_time=num_lines*frame_duration #   total time = #_frames * duration
      					  #   write start/stop time w/ label
	echo $start_time $end_time `$EXTRACT_LABELS \`basename $m\``	
	echo "."			# data seperator

done
