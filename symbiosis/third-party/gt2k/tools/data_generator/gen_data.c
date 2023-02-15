/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
/* 
 * brashear (@cc.gatech.edu)
 *
 * used to generate slightly noisy data for htk
 *
 * takes the mlf file name as parameter
 *
 * NUM is the vector size
 * NUM_OUTPUT is the number of data files to output and transcribe
 * NUM_SAMPLES is the number of utterances for each data file
 * NUM is the size of the vector for the output to the data file
 */

#include<stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/* Gnu Scientific Library */
#include<gsl/gsl_randist.h>

/* constant defines */
#define NUM 4
#define NUM_SAMPLES 5
#define NUM_OUTPUT 100 
#define OUTPUT_MLF "bob.mlf"
#define OUTPUT_DATA_NAME "out"
#define OUTPUT_DATA_EXT "dat"

int main(int argc, char **argv)  {

	//int bob=open(argv[1],O_RDONLY); /* quick tester for if file exists */
	int bob=open(OUTPUT_MLF,O_RDONLY); /* quick tester for if file exists */
	FILE *datafile;
	FILE *mlffile; /* MLF file - containts our lab transcriptions */
	
	float i;
	int j,k,m,n,p,q,x;
	char dname[100];
	const gsl_rng_type * T;
	gsl_rng * r;

	/* this is just some data to send out to HTK - its junk */
	int data[3][8]={{1,2,3,4,5,6,7,8},{4,1,2,0,5,3,2,9},{9,2,4,6,1,7,3,8}};


	/* if the file doesn't exist then we need to create it w/ the MLF header */
	if (bob == -1)	 { 
		close(bob);
		mlffile=fopen(OUTPUT_MLF,"w");  /* contains our transcriptions */
		fprintf(mlffile,"#!MLF!#\n");
		}
	/* otherwise we just need to open it and append our labels */
	else  {
		close(bob);
		mlffile=fopen(OUTPUT_MLF,"a");  /* contains our transcriptions */
		}


	/* initialize gsl randomizers */
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);

for (n=0; n<NUM_OUTPUT; n++)  {

	sprintf(dname,"%s_%04i.%s",OUTPUT_DATA_NAME, n, OUTPUT_DATA_EXT);
	datafile=fopen(dname,"w"); /* file to output data stream to */

	/* print file name out the the mlf file so it knows which ext file to look
	 * for 
	 */
	fprintf(mlffile, "\"*/%s.lab\" \n",dname);

	/* loop until quit input */
	//while (j!=3)  {
	for (m=0; m<NUM_SAMPLES; m++)  {

		j= gsl_rng_uniform_int(r, 3);

		/* print out some data for htk - depends on input */	
	  	for (p=0; p<8; p++)  {
			x=gsl_rng_uniform_int(r, 3);
			for (q=0; q<x+4; q++)  {
				for (k=0; k<NUM; k++) 	
					fprintf(datafile, "%f ", gsl_ran_gaussian(r, 0.5)+data[j][p]); 
				fprintf(datafile, "\n");
				}
			}

		/* print our transcription to the MLF file */
		switch (j)  {
			case 1: { fprintf(mlffile,"one\n");
					  break;
					  }
			case 2: { fprintf(mlffile,"two\n");
					  break;
					  }
			case 0: { fprintf(mlffile,"zero\n");
					  break;
					  }
			}

		}

	/* this is the silence at the end of the transcription - without this HTK
	 * doesn't know that we've ended one transcription and won't move on to
	 * another 
	 */
	fprintf(mlffile, ".\n");

} /* end for N */

}
