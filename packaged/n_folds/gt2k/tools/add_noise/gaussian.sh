#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


## takes in an X value followed by a mean and varaince of a Normal distribution
## displays the value of a guassian distribution with mean=$2 and variance=$3
## evaluated at X

## FIX_ME need to check for divide by zero
## FIX_ME need to check to make sure that $1 $2 $3 are defined

x=$1
mean=$2
sigma=$3

PI=$(echo "scale=10; 4*a(1)" | bc -l)  	# pi = arctan(1) * 4 (with 10 digits)

#denominator = sqrt( (2 * pi) )*sigma
denom=$(echo "scale=10; sqrt((2*$PI))*$sigma" | bc -l)


numerator=1

#exponent = exp( -.5*(x-mean)^2/sigma^2 )
exponent=$(echo "scale=10; e(-.5*($x-$mean)^2/$sigma^2)" | bc -l)

echo "scale=10; ($numerator/$denom)*$exponent" | bc -l