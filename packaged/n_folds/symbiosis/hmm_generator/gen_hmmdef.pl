#!/usr/bin/perl -w
require 5.006_001;
use strict;
use Getopt::Std;

my %opts;
getopts('n:m:v:s:dh', \%opts);
printhelp() if($opts{'h'});
printhelp() unless($opts{'n'} && $opts{'v'} && $opts{'m'} && exists $opts{'s'});

my($num_states, $num_mixes, $vec_size, $skip) = ($opts{'n'}, $opts{'m'}, $opts{'v'}, $opts{'s'});
my $outfile = $ARGV[0] || '&STDOUT';

die "Must have at least 2 states\n" unless $num_states >= 2;
die "Must have at least 1 mixture\n" unless $num_mixes >=1;

my $fh;
open($fh, ">$outfile") or die "Can't open $outfile: $!";

#Header for the HMM file
print $fh "<BeginHMM>\n<NumStates> $num_states <VecSize> $vec_size   ";
print $fh "<diagC>" if($opts{'d'});
print $fh "<nullD><USER>";
print $fh "\n\n";

my $i;
my $j;
for($i = 2; $i < $num_states; $i++)
{
	print $fh "<State> $i <NumMixes> $num_mixes\n";	
	for($j = 1; $j <= $num_mixes; $j++)
	{
		print $fh "\t<Mixture> $j .5\n";
		print $fh "\t<Mean> $vec_size\n";
		print $fh "\t", ("0.0 " x $vec_size), "\n";
		print $fh "\t<Variance> $vec_size\n";
		print $fh "\t", ("1.0 " x $vec_size), "\n";
	}
}
my $end = 0;
print $fh "<TransP> $num_states\n";
for($i = 0; $i < $num_states; $i++)
{
	print $fh "\t";

	printf($fh "%2.3e  ", 0.0) for(0..$i - 1);
	if($i == 0)
	{
		printf($fh "%2.3e  %2.3e  ", 0.0, 1.0);
		$end = 2;
	}
	elsif($i == $num_states - 1)
	{
		printf($fh "%2.3e  ", 0.0) for($i..$num_states - 1);
		$end = $num_states;
	}
	else
	{
		$end = $i + $skip + 2;
		$end = $num_states if($end > $num_states);
		printf($fh "%2.3e  ", 1.0 / ($end - $i)) for($i..$end - 1);
	}

	printf($fh "%2.3e  ", 0.0) for($end..$num_states - 1);
	print $fh "\n";
}

print $fh "<EndHMM>\n";

close($fh);


sub printhelp
{
	print<<END;
Usage: $0 -n num_states -v vector_size -s skip [-d] [output file]
  -n num_states   The number of states for the HMM to have
  -m num_mixes    The number of gaussian mixtures for each dimension of the input vector
  -v vector_size  The size of the feature vector
  -s skip         The number of skip states in the HMM
  -d              Use a diagonal covariance matrix instead of a full matrix
END

  exit 0;
}
