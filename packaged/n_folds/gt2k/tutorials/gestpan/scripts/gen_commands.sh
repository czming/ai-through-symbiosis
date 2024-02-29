#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


## generates a list of data labels.
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
# print a distinct label based on files in the directory
# the label is geneated based on the first file in the directory
# ( so this script assumes homogenous data per directory )
for m in `cat $PATH_LIST_FILE`; do

   scripts/extract_label.sh `ls $m | head -n 1`

done
