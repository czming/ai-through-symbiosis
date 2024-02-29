#!/bin/ksh
find data/* -type f | sort -V  > datafiles
rm -rf ext/data/
rm -f ext/done
mkdir ext/data/
cp -r label/* ext/data/

scripts/gen_mlf_split.sh datafiles ext > mlf/labels.mlf_letter
scripts/gen_mlf_word.sh datafiles ext > mlf/labels.mlf_word
scripts/gen_mlf_phrase.sh datafiles ext > mlf/labels.mlf_phrase
HLEd -n commands/commands_tri_internal -i mlf/labels.mlf_tri_internal mktri_internal.led mlf/labels.mlf_letter
HLEd -n commands/commands_tri_cross -i mlf/labels.mlf_tri_cross mktri_cross.led mlf/labels.mlf_letter