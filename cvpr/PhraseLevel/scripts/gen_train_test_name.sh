#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


###############################################################################
# Genereate Training and Testing names for files:
#
# Given a basename and number, this script will combine them to make
# an indexed name.  This is just to ensure that we refer to multiple test/train
# files by the same name.  This way if you want to change the format of the
# name you need only to modify this script.
#
# argument 1: the base name of the test or training filename
# argument 2: the number to index the filename
#
###############################################################################

BASENAME=$1
INDEX=$2

echo "$BASENAME$INDEX"	# filenameXX