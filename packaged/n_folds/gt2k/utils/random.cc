/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
////////////////////////////////////////////////
//Randomly partition a dataset with input probability 
//
// This version of the simple 1 fold crossvalidation
// randomizer should be very random and should not
// have a limit on how many datapoints it can randomly
// split into two files
//
// Copyright 2001
// Bradley A. Singletary and Georgia Institute of Technology
//
////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "MersenneTwister.h"

#define LINESIZE 1000
MTRand rGen;

int main(int argc, char *argv[]){
  unsigned int i, j;
  unsigned int numexts;
  float trainRatio;
  char line[LINESIZE];
  FILE *elFile;
  FILE *trFile;
  FILE *teFile;

  if (argc != 6){
    fprintf(stderr,"%s extlist numexts trainfile testfile trainRatio\n",argv[0]);
    exit(-1);
  }
  elFile=fopen(argv[1],"r");
  if (elFile == NULL){
    printf("Error opening file list.\n");
    exit(-1);
  }
  numexts=atoi(argv[2]); 
  trFile=fopen(argv[3],"w"); 
  if (trFile == NULL){
    printf("Error opening train file.\n");
    exit(-1);
  }
  teFile=fopen(argv[4],"w"); 
  if (teFile == NULL){
    printf("Error opening test file.\n");
    exit(-1);
  }

  sscanf(argv[5],"%f",&trainRatio);
  srand(time(NULL));


  for (i = 0; i < numexts && !feof(elFile); i++)
  {
     fgets(line, LINESIZE, elFile);
     if(rGen.rand() >trainRatio)
     {
       fprintf(trFile,"%s",line);
     }
     else
     {
       fprintf(teFile,"%s",line);
     } 
  }

  fclose(teFile);
  fclose(trFile);
  fclose(elFile);
  return 0;
}
