#!/bin/ksh
HVite -p 0 -t 0 -s 0 -A -T 2 -H ./models/hmm0.19/newMacros -w ./word_lattice_word -S ./testsets/testing-phrases -I ./mlf/labels.mlf_phrases -i ./ext/result.mlf_phrases ./dict/dict_letter2word ./commands/commands_letter_isolated
HResults -A -e "???" sil -T 2 -t -I ./mlf/labels.mlf_phrases ./commands/commands_word_isolated ./ext/result.mlf_phrases
