/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
// CALL ME LIKE THIS:

// htk-prep-amin <gesture> <gesture>_example<#>
// for example: htk-prep-amin play play_example0

#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#define SIZE 40
#define MAXDATA 50000

int litohtki(int data)
{
  int sepBytes[4];
  
  sepBytes[0] = (data & 0xFF000000) >> 24;
  sepBytes[1] = (data & 0xFF0000) >> 16;
  sepBytes[2] = (data & 0xFF00) >> 8;
  sepBytes[3] = data & 0xFF;
  
  return ((sepBytes[3] * 16777216) + (sepBytes[2] * 65536) + (sepBytes[1] * 256) + sepBytes[0]);
}

short lstohtks(short data)
{
  short sepBytes[2];
  
  sepBytes[0] = (data & 0xFF00) >> 8;
  sepBytes[1] = data & 0xFF;

  return (sepBytes[1] * 256 + sepBytes[0]);
}

int switchWords(int data)
{
  int sepWords[2];

  sepWords[0] = (data & 0xFFFF0000) >> 16;
  sepWords[1] = (data & 0xFFFF) << 16;

  return sepWords[0] | sepWords[1];
}

// prepare the data to be used by HTK

int main (int argc, char **argv)
{
  char *gesture;
  char *example, *tmp, *tmp2 = NULL, dir[30];

  char datafilename[SIZE];  // the name of the input file
  char gesturefilename[SIZE];  //name of file with gesture
  char labelfilename[SIZE]; // name of the label file
  char binoutfile[SIZE];  // name of the binary file to be written

  char timefilename[SIZE]; // file where time to execute gesture is stored

  FILE *datain;  // file from which data is coming
  FILE *gesturein; // file with the gesture
  FILE *labelfile; // label file
  int binout = -1;  // binary file to be written
  int tempBin;

  FILE *timefile;

  char line[MAXDATA];

  // arrays for data 

  int data[MAXDATA][18];

  // first 4 information to be written to the binary file

  int number_of_samples = 0, nos; // number of samples in this data (ie. number of data)
  long sample_period, sp;  // period of the sample. should be first line from file
  short size, s; 
  short type, t;

  // random variables (not in the statistical sense.  Just stuff used in
  // different places

  int i, j;

  long time;
  float temp;
  short temp2[2];

  gesture = argv[1];  // the gesture is the first argument sent in
  example = argv[2];  // the example number is the second argument

  printf("gesture = %s\n", gesture);
  printf("example = %s\n", example);

  // build the file name
  // should be like <gesture>_<example#>   ex: play_example0
  if ((tmp = strchr(example, '/')) == NULL)
  {
     tmp = example;
  }
  else
  {
     tmp++;
     tmp2 = example;
     i = 0;
     while (tmp2[i] != '/')
     {
	dir[i] = tmp2[i];
        i++;
     }
     dir[i] = '\0';
  }
  sprintf(datafilename, "%s", tmp); 

  printf("datafilename = %s\n", datafilename);

  sprintf(labelfilename, "%s.lab", datafilename);  // build the label file name
  sprintf(binoutfile, "%s.ext", datafilename);  // build the ext file name
  if (tmp2 == NULL)
  {
     sprintf(timefilename, "time_%s", datafilename); // build timefile name
  }
  else
  {
     sprintf(timefilename, "%s/time_%s", dir, datafilename);
  }

  printf("Starting data conversion for file %s\n", example);
  printf("LAB file is %s\n", labelfilename);
  printf("EXT file is %s\n", binoutfile);
  printf("Time file is %s\n", timefilename);

  // open the data file containing the information about the gesture

  datain = fopen(example, "r");
  if (datain == NULL)
  {
    printf("Error opening gesture data file.\n");
    return(-1);
  }

  // open the label file

  labelfile = fopen(labelfilename, "w");
  if (labelfile == NULL){
    printf("Error opening label file.\n");
    return(-1);
  }

  // open the binary out file

  binout = open(binoutfile, O_WRONLY | O_CREAT, 0);
  if (binout <= 0)
  {
    printf("Error opening binary file.\n");
    return(-1);
  }

  // open the time file

  timefile = fopen(timefilename, "r");
  if (timefile == NULL)
  {
    printf("Error opening time file.\n");
    return(-1);
  }

  // read the first line in from the data file.  It should be the time

  fscanf(timefile, "%ld\n", &time);
  sample_period = time;

  printf("time = %ld\n", sample_period);

  // read in the data into the arrays

  while(!feof(datain))
  {
    fscanf(datain, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", 
	   &data[number_of_samples][0],&data[number_of_samples][1],
	   &data[number_of_samples][2], &data[number_of_samples][3], 
	   &data[number_of_samples][4], &data[number_of_samples][5], 
	   &data[number_of_samples][6], &data[number_of_samples][7], 
	   &data[number_of_samples][8], &data[number_of_samples][9], 
	   &data[number_of_samples][10], &data[number_of_samples][11], 
	   &data[number_of_samples][12], &data[number_of_samples][13], 
	   &data[number_of_samples][14], &data[number_of_samples][15], 
	   &data[number_of_samples][16], &data[number_of_samples][17]);                                      
   
    number_of_samples++;
  }

  if (data[number_of_samples][0] == 0 && data[number_of_samples][1] == 0 && 
      data[number_of_samples][2] == 0 && data[number_of_samples][3] == 0 && 
      data[number_of_samples][4] == 0 && data[number_of_samples][5] == 0 && 
      data[number_of_samples][6] == 0 && data[number_of_samples][7] == 0 && 
      data[number_of_samples][8] == 0 && data[number_of_samples][9] == 0 && 
      data[number_of_samples][10] == 0 && data[number_of_samples][11] == 0 && 
      data[number_of_samples][12] == 0 && data[number_of_samples][13] == 0 && 
      data[number_of_samples][14] == 0 && data[number_of_samples][15] == 0 && 
      data[number_of_samples][16] == 0 && data[number_of_samples][17] == 0)
    number_of_samples--;

  // write label file

  // label has "gesture-name start-time end-time"
  fprintf(labelfile, "%d %ld %s\n", 0, sample_period, gesture);
  
  // write binary file

  size = (short) (16 * sizeof(float));  // size of vector, 18 ints
  type = (short) 9;  // user defined data type

  sample_period = sample_period / number_of_samples;

  printf("Writing this data to .ext header: %d, %ld, %d, %d\n",
	 number_of_samples, sample_period, size, type);

  nos = switchWords(number_of_samples);
  sp = switchWords(sample_period);
  s = size;
  t = type;

  nos = litohtki(number_of_samples);
  sp = litohtki(sample_period);
  s = lstohtks(size);
  t = lstohtks(type);

  write(binout, &nos, sizeof(int));// write the number of examples to the bin file
  write(binout, &sp, sizeof(int));  // write the period of the sample to the bin file
  write(binout, &s, sizeof(short)); // write the vector length to file
  write(binout, &t, sizeof(short)); // write the type of file. In this case, user type == 9

  // write the data to the binary file

  tempBin = open("tempbin", O_RDWR);

  for (i=0; i<number_of_samples; i++)
  {
    for(j = 0; j < 16; j++)
    {
      temp = (float)data[i][j];
      //temp = ((float)data[i][j] - 22.0) / 220.0;
      write(tempBin, &temp, sizeof(float));
      lseek(tempBin, 0, SEEK_SET);
      read(tempBin, &temp2[0], sizeof(short));
      read(tempBin, &temp2[1], sizeof(short));
      lseek(tempBin, 0, SEEK_SET);
      temp2[0] = lstohtks(temp2[0] & 0xFFFF);
      temp2[1] = lstohtks(temp2[1] & 0xFFFF);
 
      //temp = lstohtks((short)data[i][j]);
      //write(binout, &temp, sizeof(float));
      write(binout, &temp2[1], sizeof(short));
      write(binout, &temp2[0], sizeof(short));
    }
  }

  close(tempBin);
  // close everything up

  fclose(datain);
  fclose(labelfile);
  fclose(timefile);
  close(binout);

  return(1);
}
