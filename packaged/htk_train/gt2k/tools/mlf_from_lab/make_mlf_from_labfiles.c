/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
#include <stdio.h>

int main()  {

	FILE *labfiles=fopen("labfiles","r");
	FILE *mlffile=fopen("labels.mlf","w");
	FILE *infile;

	char filename[100];
	char line[100];

	fprintf(mlffile,"#!MLF!#\n");

	while (fscanf(labfiles, "%s", filename)!=EOF)  {

		fprintf(mlffile,"\"*/%s\"\n", filename);
		infile=fopen(filename,"r");

		while (fgets(line,100, infile)!=NULL)  {
			fprintf(mlffile,"%s ",line);
			}

		fprintf(mlffile,".\n");
		fclose(infile);
		}

	fclose(labfiles);
	fclose(mlffile);
}
