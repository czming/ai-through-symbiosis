/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
/* gcc htk_prepare.c -g -o htk_prepare -lm */
/* rerun splitter */

/* brashear 26 nov 02 - 
 * i've taken out the reference to fpsent because we aren't using a sentence
 * file anymore - we've moved on to a mlf file
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

#include "htkhelper.h" 

void swap12(unsigned char &data1,unsigned char &data2)
{
    data1=data1^data2;
    data2=data1^data2;
    data1=data1^data2;
}


void swap1(unsigned char *data, unsigned int bytes)
{
 
  if(bytes==2)
  {
    swap12(data[0],data[1]);
  }
  else if(bytes==4)
  {
    swap12(data[0], data[3]);
    swap12(data[1], data[2]);
  }

}



int main (int argc, char **argv)
{
  int scanf_value=0;
  unsigned int counter, sampPeriod = 2000;

  unsigned int i,j;
  float points[1000];
  float fp[1000][1000];

  char *filename, *dirname, sentence_filename[1000], data_out[1000], label_out[1000];
  char *locname;
  short type, size;
  FILE *fpdata, *fpsent, *fplout;
  FILE *dout;
  unsigned int num_points;
  char word[1000];

  float temp; 
  unsigned char *bob;
  

  counter = 0;
  filename = argv[1];
  locname = filename;				/* mirror data path in ext/ dir
  						 * structure */
  sscanf(argv[2], "%d",&num_points);
  if (argc == 4){
    sprintf(data_out,"%s",argv[3]);
  } else {
    sprintf(data_out, "%s.ext", filename);
  }
  
  fpdata = fopen(filename, "r");
  if (fpdata==NULL)
	fprintf(stderr, "null file pointer - fpdata\n");
   dout=fopen(data_out, "w");
   if (dout==NULL)
		fprintf(stderr, "null file pointer - dout \n");
  fprintf(stderr, "opening for data %s\n", data_out);

  /* FEATURES: num_points is the dimension of the feature vector */
  for (i=0; i< num_points; i++) {
      scanf_value = fscanf(fpdata, "%f", &(points[i]));
  }


  
  for (i=0; i< num_points; i++) {
    fp[i][counter]=(float) points[i];
  }
  counter++;
  
  while (scanf_value > 0)
     {
         for (i=0; i< num_points; i++) {
             scanf_value = fscanf(fpdata, "%f", &(points[i]));
         }
         for (i=0; i< num_points; i++) {
           fp[i][counter]=(float) points[i];
         }
           counter++;
     }

  

  /* HEADER */
  /* float data */  
  size = (short) (num_points * sizeof(float));
  /* user type */
  type = (short) 9;
  /* write header */
  if (counter > 0 ) {
      /* Write label */

	HTKDataVector *htkdata=new HTKDataVector(counter, sampPeriod, size,
		type);

	unsigned int m_samples=htkdata->getSamples();
	unsigned int m_period=htkdata->getSampPeriod();
	unsigned short int m_size=htkdata->getSampSize();
	unsigned short int m_kind=htkdata->getKind();

      fwrite(&m_samples, sizeof(unsigned int),1,dout);
      fwrite(&m_period, sizeof(unsigned int),1,dout);
      fwrite(&m_size, sizeof(unsigned short int),1,dout);
      fwrite(&m_kind, sizeof(unsigned short int),1,dout);

      fprintf(stderr, "number of frames %d\n", counter);
      /* write data */
	
      for (i=0; i < counter-1; i++)
         {
             for (j=0;j<num_points;j++) {
		         temp=fp[j][i];
		         swap1(reinterpret_cast<unsigned char *>(&temp),sizeof(float));
                 fwrite(&temp, sizeof(float),1,dout); 
             }
         }
      
      fclose(fpdata);
      fclose(dout);
  }      
  
}
