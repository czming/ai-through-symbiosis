#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

#################################################################
# lotsocross.sh
# 
# Simple script to perform many iterations of independent
# cross validation sets, with an insertion penalty.
# Stores each iteration in oldsets/<iteration num>
# oldsets/ must exist.
#
# This script makes it easy to do many independent cross-validation
# sets to find both good models and to find an average sense of
# accuracy.
#################################################################

NUM_ITER=20
INS_PENALTY=-80.0

for i in `seq 0 ${NUM_ITER}`
do
	scripts/train.sh scripts/options.sh &> out.log
	HVite -A -T 1 -H models/hmm0.7/newMacros -w word.lattice \
		-S testsets/testing-extfiles0 -I labels.mlf -i ins_out.mlf \
		-p ${INS_PENALTY} -n 4 20 dict commands > /dev/null
	HResults -A -T 1 -t -I labels.mlf -z null -e null start_sentence \
		-e null end_sentence -p commands ins_out.mlf > ins_out.log
	mkdir oldsets/$i
	cp *.log oldsets/$i/
	cp ins_out.mlf oldsets/$i
	cp models/hmm0.7/newMacros oldsets/$i
	cp testsets/testing-extfiles0 oldsets/$i
	cp trainsets/training-extfiles0 oldsets/$i
done
