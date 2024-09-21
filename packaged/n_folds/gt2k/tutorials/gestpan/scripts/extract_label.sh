#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

#
# Given a filename, extract the label from the filename.
# modify this script to extract the labels as you see fit.
#
# INPUT: filename
#
# currently it assumes that the label is everything preceeding the 
# first occurance of an underscore.
#
# example useage: 
# 
#		./extract_label marc_given_02_27_03_00000.ppm
#
# example output:
#
#		marc
#


FILENAME=$1

# delete everything from the leading '_' through the end_of_string
echo $FILENAME | sed -e 's/_.*$//'
