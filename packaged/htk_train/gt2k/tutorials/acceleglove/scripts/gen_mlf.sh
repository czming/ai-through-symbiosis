#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

# usage:
# scripts/gen_mlf.sh datafiles ext/ sentences.txt > labels.mlf
#
# You should call this from the main project directory, where:
# datafiles is the list of all of the *.TXT data files
# ext/ is the location of the HTK-readable converted data (or its destination,
#	 if it has not been created yet)
# sentences.txt is the list of sentences represented by the data. each line
#	in this file corresponds to the .TXT file with the corresponding line
#	number.
# labels.mlf is the destination for this mlf file

# This script generates a naive master label file for all of the data files 
# listed in DATA_FILE_LIST, which for this project are of the form <number>.TXT
# where the number in the data file corresponds to a line number in 
# SENTENCE_FILE.  This line is used to label the data, with "start_sentence"
# and "end_sentence" prepended and appended, and with each of these words
# given equal time in the data.  In other words, it is assumed that all signs
# are of equal length.  This is, of course, false, but HRest will re-estimate
# it for us, later.

DATA_FILE_LIST=$1
EXT_DIR=$2
SENTENCE_FILE=$3

# Variables for managing the duration of each sign
integer frame_duration=2000
integer start_time=0
integer end_time=0
integer num_lines=0
integer num_words=0
integer incr_step=0
integer next_time=0

# start our MLF file
echo "#!MLF!# \n"

for m in `cat $DATA_FILE_LIST`; do
	# first, check to see if the EXT_DIR given is the full or 
	# relative path by checking to see if it starts with "/"
        echo "${EXT_DIR}" | grep '^/' > /dev/null

	# if it's the full path, just prepend the EXT_DIR variable to the 
	# data filename, in order to be consistent with the data conversion 
	# to HTK readable format
        if [ "$?" = "0" ]; then
           labname="\"${EXT_DIR}/$m.lab\""  #search for HTK readable datafiles
	# if it's not a full path, make it one
        else
           labname="\"`pwd`/${EXT_DIR}/$m.lab\""
        fi

	# start a label entry with the filename of the .ext file to label
	echo "$labname"	| sed -e "s|//|/|g"	# output the .lab filename

	# label the data

	# extract the line number, and pull the sentence from SENTENCE_FILE
	filename=`echo $m|sed -e "s/.*\///g"`
	sentnum=${filename%.TXT}
	sent=`head -n $sentnum $SENTENCE_FILE| tail -n 1`

	# find the number of frames in the data file, calculating
	# the end time, and coming up with a common sign length based
	# on the number of words in the sentence (+ 2 for start and end)
	num_lines=`cat $m|wc -l`
	end_time=$num_lines*$frame_duration
	num_words=`echo $sent|wc -w`+2
	incr_step=$end_time/$num_words

	# print out the time data for each word, starting at zero
	# and incrementing by incr_step
	# format is "start_time end_time word"
	start_time=0
	for i in start_sentence $sent
	do
		next_time=$start_time+$incr_step
		echo "$start_time $next_time $i"
		start_time=$next_time
	done	

	# the end_sentence word must end at the end of the data file
	echo "$start_time $end_time end_sentence"
	# end the label entry with a single period on a line
	echo "."
done
