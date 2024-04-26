/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
/* by brashear */
/* this file reads in a list of commands from the file "commands and creates
   an htk dictionary and grammar from them.  it assumes that your grammar
   will be a single action chosen from the list in commands; the output is in
   file dict and grammar
*/

#include <stdio.h>
#include <string.h>

#define TRUE 1
#define FALSE 0

#define MAX_COMMAND_LENGTH 	20
#define MAX_NUM_COMMANDS 	20
#define OUT_LENGTH 		MAX_NUM_COMMANDS*(MAX_COMMAND_LENGTH+1)+30
#define DICT_LENGTH 		MAX_NUM_COMMANDS*MAX_COMMAND_LENGTH*3

int main(int argc, char **argv)  {

	FILE *grammar=fopen("grammar","w");
	FILE *dict=fopen("dict","w");
	FILE *commands=fopen("commands","r");

	char output[OUT_LENGTH];
	char command[MAX_NUM_COMMANDS][MAX_COMMAND_LENGTH]; 
	char dict_list[DICT_LENGTH];
	int command_count=0;
	int flag=FALSE;

	if (commands==NULL)  {
		fprintf(stderr, "problem opening 'commands' file\n");
		exit(-1);
		}
	if (dict==NULL || grammar==NULL)  {
		fprintf(stderr, "problem opening new file\n");
		exit(-1);
		}

	sprintf(output,"$gesture = ");
	sprintf(dict_list,"");
	while ( fscanf(commands, "%s", command[command_count])!=EOF) {
		printf("%s \n", command[command_count]);
		if (flag==TRUE)
			sprintf(output,"%s |",output);
		sprintf(output, "%s %s",output, command[command_count]);
		sprintf(dict_list,"%s%s\t%s\n",dict_list,
			command[command_count], command[command_count]);
		command_count++;
		flag=TRUE;
		}
	sprintf(output,"%s; \n\n ( $gesture ) \n", output);
	fprintf(grammar, "%s", output);
	fprintf(dict, "%s",dict_list);

	fclose(commands);
	fclose(grammar);
	fclose(dict);
}
			
