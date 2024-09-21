#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


# j=1+(int) (10.0*rand()/(RAND_MAX+1.0));

RAND_MAX=32767				# max value of $RANDOM

echo "scale=10; $RANDOM/$RAND_MAX" | bc -l
