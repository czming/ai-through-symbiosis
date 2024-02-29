#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


## FIX_ME: use get_opt to read in parameters or check to see that they are
##         passed in properly

## given a file and a vector size, add noise to zero valued vecotrs

## alias MATH 'set \!:1 = `echo "\!:3-$" | bc -l`'

# NOTES:
#
# $(<filename) is a more efficient way of doing `cat filename`
 
ZERO_SUM=1				# should be one, it is an equality flag
RAND_MAX=32767				# max value of $RANDOM

datafile=$1
vec_size=$2
element_sum=0
counter=0


# loop for each element in the file plus one extra item.  the way this code is
# structured the last dummy item is needed to ensure that all elements in the
# datafile are  properly processed.
for n in $(<$datafile) $(echo 0.0); do

    # check to see if we have a zero_sum vector thus far
    if [[ `echo "$element_sum==0" | bc -l` -eq $ZERO_SUM ]]; then
	add_noise=1;
    else
	add_noise=0;
    fi

    # if we have seen more then one line's worth of values
    # (ie: this n value is the first item from the next line)
    # then we need to start a new sum and check to see if elements from
    # the line just read were all zero.
    if [[ counter -lt $vec_size ]]; then

	# add the element to a running sum using 'bc'
	element_sum=`echo "$n+$element_sum" | bc -l`;
	counter=$((counter+1));	
    else
	# since we have read in one vector of data reset the sum.
	# we have read in the start of the next data vector, so start
	# a new sum of elements
	element_sum=0;
	element_sum=`echo "$n+$element_sum" | bc -l`;
	
	# if the vector just completed was a zero vector, 
	# ie: the sum of the elements equals zero and the flag is set to one,
	# then we need to add gaussian noise to generate a non-zero vector.
	# if the vector is already non-zero then leave it alone
	if [[ `echo $add_noise` -eq $ZERO_SUM ]]; then
	    # adding gaussian noise to zero is the same as generating a 
	    # point chosen at random from a gaussian.  Do this for each
	    # element in the zero vecor.  at this point counter should
	    # be equal to $vec_size, so just count backwards to zero.
	    # this will also serve the purpose of setting counter back
	    # to zero to count elements on the next line of input.
	    while [[ $counter -gt 0 ]]
	    do
		# generate a random number between [0,1] and evaluate
		# it at a Normal Distribution centered at 1 with spread of 1
		random_X=$(echo "scale=10; $RANDOM/$RAND_MAX" | bc -l )
		data_value=`./gaussian.sh $random_X 1 1`

		# print the random number out to stdout followed by a space
		# and supressing the new line
		print -n $data_value" "
		# decrement the counter
		counter=$((counter-1))
	    done

	    print -n "\n"
	else
    	# we did not read a zero vector, so echo out the exact vector we
	# just read.
	print $original_vector

	fi
	
	# start recording a new vector
	original_vector=""
	
	# since $n already contains the first element of the next vector,
	# reset the counter to 1
	counter=1

    fi

    # we want to keep a copy of the original data vector that we are reading.
    # each time we see a new value, tack it onto the end of our string.
    original_vector=`print -n $original_vector $n ""`

done
