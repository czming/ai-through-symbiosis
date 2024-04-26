#!/bin/ksh
HVite -p 0 -t 0 -s 0 -A -T 2 -H ./models/hmm0.19/newMacros -w ./word_lattice_word -S ./testsets/testing-phrases -I ./mlf/labels.mlf_phrases -i ./ext/result.mlf_phrases  ./commands/commands_tri_internal
HResults -A -e ??? -T 1 -t -I /WordLevel/mlf/labels.mlf_phrases /WordLevel/commands/commands_tri_internal /WordLevel/ext/result.mlf_phrases
