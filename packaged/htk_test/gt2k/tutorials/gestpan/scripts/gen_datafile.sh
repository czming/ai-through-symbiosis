#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


## generates a list of the data files.
## first argument is a file containing the paths to all datafiles
##
## example:
##
##		./gen_datafile.sh path_list
##
##	       
##
PATH_LIST_FILE=$1

# for each path listed in the file
# print out the contents of each directory
#
for m in `cat $PATH_LIST_FILE`; do

    for n in `ls $m`; do 
	echo $m/$n
    done	

done
