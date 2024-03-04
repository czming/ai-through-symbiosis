#!/usr/bin/perl
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

################################################
# list_to_grammar.pl
#
# Simple script to convert a newline separated list of words into a 
# vertical pipe ( "|" ) separated list of words.  Helps in making grammars
# from large sets of words.
################################################

@list = <STDIN>;

$words = join " \| ", @list;
$words =~ s/\n//g;
$words = lc($words);
print $words;
