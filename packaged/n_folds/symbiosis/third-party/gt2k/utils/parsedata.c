/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
/* by brashear */
/* this file takes in a file of data from findlaser; it assumes that each  */
/* gesture is deliniated by a "MARK" string (produced by hitting the 'm'   */
/* key in captaincrunch; it splits each gesture into its own file, using   */
/* the base filename + _#, where # is the gesture number		   */

/* note - i wrote this as a quick utility and its not very robust - it 	   */
/* assumes one mark per experiment, and doesn't do much error checking     */
/* i choose to prune the first gester (file_0.txt) because it often has    */
/* noise from calibrating the system                                       */					   
#include <stdio.h>
#include <string.h>

#define TRUE 1
#define FALSE 0

int main (int argc, char * argv[]) {

	char *filename=argv[1];
	FILE *file=NULL;
	char line[200];

	char outname[40];
	FILE *outfile=NULL;

	int count=0;
	int flag;
	char temp[40];
	
	if (argc<1)  {
		fprintf(stderr, "must give filename\n");
		exit(-1);
		}
	if ((file=fopen(filename, "r"))==NULL)  {
		fprintf(stderr, "problem opening file\n");
		exit(-1);
		}

	printf("input file = %s\n",filename);
	sscanf(filename, "%[^.]s",outname);
	printf("file base = %s \n", outname);

	flag=FALSE;	
	while (fgets(line, 200, file)!=NULL)  {

		/* Marks a new gesture */
		if (strstr(line,"MARK")!=NULL)  {
			flag=FALSE;	
			if (outfile!=NULL)  {
				fclose(outfile);	
				outfile=NULL;
				}
			}
		/* line of data */
		else  {
			/* we need to open a file */
			if (flag==FALSE)  {
				sprintf(temp,"%s_%d",outname,count);
				outfile=fopen(temp,"w");
				flag=TRUE;
				count++;
				}
			fprintf(outfile, line);
			}
		} /* end while */

	fclose(file);
	file=NULL;
	} /* end main */
	
