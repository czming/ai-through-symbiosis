#!/usr/bin/perl
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

###############################
# create_grammar.pl
# 
# create a simple grammar and dict file from the list of known commands
###############################
open GRAMMAR, '>grammar' or die "couldn't write 'grammar'";
open DICT, '>dict' or die "couldn't write 'dict'";
open COMMANDS, '<commands' or die "couldn't read 'commands'";

while (<COMMANDS>) {
    print;
    chomp;
    push @commands, ($_) if $_;
}

foreach (@commands) {
    print DICT "$_\t$_\n";
}

print GRAMMAR '$gesture =  ' . 
    join(' | ',@commands) . 
    ";\n\n ( \$gesture ) \n";
