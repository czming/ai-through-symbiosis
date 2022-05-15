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
## labels must also be provided for the datafiles.
##
## example usage: scripts/gen_mlf.sh fileslist ext/ > labels.mlf
##
##
## This script gets labels from .lab files

DATA_LIST_FILE=$1 
EXT_DIR=$2


integer frame_duration=2000           # each frame is about 2000ns in HTK
integer start_time=0           		
integer end_time=0
integer total_time=0       		
integer num_lines=0
integer num_labels=0
integer ind=0
str=""

echo "#!MLF!#"			# write the header for the file

for m in `cat $DATA_LIST_FILE`; do	# for each data file listed
   
#    if ! [ -d "$EXT_DIR/$m" ]; then	# mirror the data directory hierarchy
#    	mkdir -p "$EXT_DIR/$m"		# in the ext output
#    fi

	# see if we need to append a path to this
	echo "${EXT_DIR}" | grep '^/' > /dev/null
	if [ "$?" = "0" ]; then
	   labname="${EXT_DIR}/$m.lab"  #search for HTK readable datafiles
	else
	   labname="`pwd`/${EXT_DIR}/$m.lab"
	fi
	echo "\"$labname\"" | sed -e "s|//|/|g"  # output the label filename
	num_lines=`cat $m | wc -l` #   compute the num lines per file
	num_labels=`cat $labname | wc -l`
	total_time=num_lines*frame_duration #   total time = #_frames * duration
      					  #   write start/stop time w/ label
    start_time=0
    ind=1
	for k in `cat $labname`; do
		if [[ $k = "SKIP" ]]; then
			ind=ind+1
			continue
		fi
		end_time=total_time*ind/num_labels
		echo $start_time $end_time $k
		start_time=total_time*ind/num_labels
		ind=ind+1
	done
	echo "."			# data seperator
done
