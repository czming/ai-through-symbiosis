/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <queue>
#include "htkhelper.h"

void swap2(unsigned char &data1,unsigned char &data2)
{
    data1=data1^data2;
    data2=data1^data2;
    data1=data1^data2;
}


void swap(unsigned char *data, unsigned int bytes)
{
  if(bytes==2)
  {
    swap2(data[0],data[1]);
  }
  else if(bytes==4)
  {
    swap2(data[0], data[3]);
    swap2(data[1], data[2]);
  }
}

HTKDataVector::HTKDataVector()
{
  nSamples=0;
  sampPeriod=0;
  sampSize=0;
  parmKind=0;
  nSamples_sw=0;
  sampPeriod_sw=0;
  sampSize_sw=0;
  parmKind_sw=0;
}
HTKDataVector::HTKDataVector(unsigned int nSamp,unsigned int period,
			     unsigned short int size, unsigned short kind)
{
  nSamples=nSamp;
  sampPeriod=period;
  sampSize=size;
  parmKind=kind;
  nSamples_sw=nSamples;
  sampPeriod_sw=sampPeriod;
  sampSize_sw=sampSize;
  parmKind_sw=parmKind;
  swap(reinterpret_cast<unsigned char *>(&nSamples_sw),
       sizeof(unsigned int));     
  swap(reinterpret_cast<unsigned char *>(&sampPeriod_sw),
       sizeof(unsigned int));
  swap(reinterpret_cast<unsigned char *>(&sampSize_sw),
       sizeof(unsigned short int));
  swap(reinterpret_cast<unsigned char *>(&parmKind_sw),
       sizeof(unsigned short int));
}


///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
FreqHistData::FreqHistData()
{
  fprintf(stderr,"default constructor called, for no good reason\n");
  exit(0);
}
FreqHistData::~FreqHistData()
{
  float *tmp;
  while(readbuf.empty()==false){
    tmp = readbuf.front();
    readbuf.pop_front();
    delete [] tmp;
  }
  while(writebuf.empty()==false){
    tmp = writebuf.front();
    writebuf.pop_front();
    delete [] tmp;
  }
}

FreqHistData::FreqHistData(unsigned int nElements,unsigned int buffSz):
  HTKDataVector::HTKDataVector(0,FRAME_RATE_100ns, sizeof(float)*nElements,KIND_USER_DEFINED)
{
  unsigned int i;
  float *tmp=NULL;
  buffSize=buffSz;
  numElements=nElements;

  for(i = 0; i < buffSize; i++)
  {
    tmp=new float[numElements];
    readbuf.push_back(tmp);
  }
}
  
void FreqHistData::QueueHistData(float *inSamp)
{
  float *tmp;
  unsigned int i;
  if(readbuf.empty()==true)
  {
    tmp=writebuf.front();
    writebuf.pop_front();
  }
  else
  {
    tmp=readbuf.front();
    readbuf.pop_front();
  }
  memcpy(tmp,inSamp,sampSize);
  for(i = 0; i < numElements; i++)
  {
    swap(reinterpret_cast<unsigned char *>(&tmp[i]),sizeof(float));
  }
  writebuf.push_back(tmp);

}
int FreqHistData::WriteHistData(unsigned int gesture, unsigned int runNum, unsigned int exampleNum)
{

  FILE *of=NULL;
  float *tmp;
  char outStr[255];
  unsigned long cnt=writebuf.size();
  unsigned long rate=FRAME_RATE_100ns;

  if(writebuf.size() > MIN_FRAMES_FOR_GESTURE)
  {
    if(gesture == 0)
    {
      sprintf(outStr,"%1.1u_%5.5u_%6.6u.ext",gesture,runNum,exampleNum);
      of=fopen(outStr,"wb");
      if(of==NULL)
      {
         fprintf(stderr,"could not open binary file\n");
	 exit(0);
      }
      nSamples=writebuf.size();
      nSamples_sw=nSamples;
      swap(reinterpret_cast<unsigned char *>(&nSamples_sw),sizeof(unsigned int));
      fprintf(stderr,"%u %u:%u %u %u %u\n",gesture,exampleNum,nSamples,sampPeriod,sampSize,parmKind);
      fwrite(&nSamples_sw,sizeof(unsigned int),1,of);
      fwrite(&sampPeriod_sw,sizeof(unsigned int),1,of);
      fwrite(&sampSize_sw,sizeof(unsigned short int),1,of);
      fwrite(&parmKind_sw,sizeof(unsigned short int),1,of);

      while(writebuf.empty()==false)
      {
	tmp=writebuf.front();
	writebuf.pop_front();
	fwrite(tmp,sampSize,1,of);
	readbuf.push_back(tmp);
      }
      fclose(of);

      sprintf(outStr,"%1.1u_%5.5u_%6.6u.lab",gesture,runNum,exampleNum);
      of=fopen(outStr,"w");
      if(of==NULL)
      {
         fprintf(stderr,"could not open label file\n");
	 exit(0);
      }
      fprintf(of,"0   %lu   lineup\n",cnt*rate);
      fclose(of);

    }
    else if(gesture == 1)
    {
      sprintf(outStr,"%1.1u_%5.5u_%6.6u.ext",gesture,runNum,exampleNum);
      of=fopen(outStr,"wb");
      if(of==NULL)
      {
         fprintf(stderr,"could not open binary file\n");
	 exit(0);
      }
      nSamples=writebuf.size();
      nSamples_sw=nSamples;
      swap(reinterpret_cast<unsigned char *>(&nSamples_sw),sizeof(unsigned int));
      fwrite(&nSamples_sw,sizeof(unsigned int),1,of);
      fwrite(&sampPeriod_sw,sizeof(unsigned int),1,of);
      fwrite(&sampSize_sw,sizeof(unsigned short int),1,of);
      fwrite(&parmKind_sw,sizeof(unsigned short int),1,of);

      deque<float *>::iterator diter; 

      for (  diter=writebuf.end(); diter != writebuf.begin(); ) 
      {
	diter--;
	tmp=*diter;
	fwrite(tmp,sampSize,1,of);
      }
      fclose(of);
      //kill last element
      tmp=writebuf.front();
      writebuf.pop_front();
      readbuf.push_back(tmp);
      sprintf(outStr,"%1.1u_%5.5u_%6.6u.lab",gesture,runNum,exampleNum);
      of=fopen(outStr,"w");
      if(of==NULL)
      {
         fprintf(stderr,"could not open label file\n");
	 exit(0);
      }
      fprintf(of,"0   %lu   other\n",cnt*rate);
      fclose(of);

    }
    else
    {
      return 1;
    }
  }
  else
  {
    return 1;
  }


  return 0;
}

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
RawImgData::RawImgData()
{
}
RawImgData::~RawImgData()
{

  float *tmp;
  while(readbuf.empty()==false){
    tmp = readbuf.front();
    readbuf.pop_front();
    delete [] tmp;
  }
  while(writebuf.empty()==false){
    tmp = writebuf.front();
    writebuf.pop_front();
    delete [] tmp;
  }

}

RawImgData::RawImgData(unsigned int nElements,unsigned int buffSz):
  HTKDataVector::HTKDataVector(0,FRAME_RATE_100ns, sizeof(float)*nElements,KIND_USER_DEFINED)
{
  unsigned int i;
  float *tmp;
  buffSize=buffSz;
  numElements=nElements;
  for(i = 0; i < buffSize; i++)
  {
    tmp=new float[numElements];
    readbuf.push_back(tmp);
  }
}
  
void RawImgData::QueueImgData(float *inSamp)
{
  float *tmp;
  unsigned int i;

  if(readbuf.empty()==true)
  {
    tmp=writebuf.front();
    writebuf.pop_front();
  }
  else
  {
    tmp=readbuf.front();
    readbuf.pop_front();
  }
  memcpy(tmp,inSamp,sampSize);
  for(i = 0; i < numElements; i++)
  {
    swap(reinterpret_cast<unsigned char *>(&tmp[i]),sizeof(float));
  }
  writebuf.push_back(tmp);

}

int RawImgData::WriteImgData(unsigned int gesture, unsigned int runNum, unsigned int exampleNum)
{

  FILE *of=NULL;
  float *tmp;
  char outStr[255];
  unsigned long cnt=writebuf.size();
  unsigned long rate=FRAME_RATE_100ns;

  if((writebuf.size() > MIN_FRAMES_FOR_GESTURE) && (writebuf.size() <= MAX_FRAMES_FOR_GESTURE))
  {
    if(gesture == 0)
    {
      sprintf(outStr,"%1.1u_%5.5u_%6.6u.ext",gesture,runNum,exampleNum);
      of=fopen(outStr,"wb");
      if(of==NULL)
      {
         fprintf(stderr,"could not open binary file\n");
	 exit(0);
      }
      nSamples=writebuf.size();
      nSamples_sw=nSamples;
      swap(reinterpret_cast<unsigned char *>(&nSamples_sw),sizeof(unsigned int));
      fprintf(stderr,"%u %u:%u %u %u %u\n",gesture,exampleNum,nSamples,sampPeriod,sampSize,parmKind);
      fwrite(&nSamples_sw,sizeof(unsigned int),1,of);
      fwrite(&sampPeriod_sw,sizeof(unsigned int),1,of);
      fwrite(&sampSize_sw,sizeof(unsigned short int),1,of);
      fwrite(&parmKind_sw,sizeof(unsigned short int),1,of);

      while(writebuf.empty()==false)
      {
	tmp=writebuf.front();
	writebuf.pop_front();
	fwrite(tmp,sampSize,1,of);
	readbuf.push_back(tmp);
      }
      fclose(of);

      sprintf(outStr,"%1.1u_%5.5u_%6.6u.lab",gesture,runNum,exampleNum);
      of=fopen(outStr,"w");
      if(of==NULL)
      {
         fprintf(stderr,"could not open label file\n");
	 exit(0);
      }
      fprintf(of,"0   %lu   lineup\n",cnt*rate);
      fclose(of);

    }
    else if(gesture == 1)
    {
      sprintf(outStr,"%1.1u_%5.5u_%6.6u.ext",gesture,runNum,exampleNum);
      of=fopen(outStr,"wb");
      if(of==NULL)
      {
         fprintf(stderr,"could not open binary file\n");
	 exit(0);
      }
      nSamples=writebuf.size();
      nSamples_sw=nSamples;
      swap(reinterpret_cast<unsigned char *>(&nSamples_sw),sizeof(unsigned int));
      fwrite(&nSamples_sw,sizeof(unsigned int),1,of);
      fwrite(&sampPeriod_sw,sizeof(unsigned int),1,of);
      fwrite(&sampSize_sw,sizeof(unsigned short int),1,of);
      fwrite(&parmKind_sw,sizeof(unsigned short int),1,of);

      deque<float *>::iterator diter; 

      for (  diter=writebuf.end(); diter != writebuf.begin(); ) 
      {
	diter--;
	tmp=*diter;
	fwrite(tmp,sampSize,1,of);
      }
      fclose(of);
      //kill last element
      tmp=writebuf.front();
      writebuf.pop_front();
      readbuf.push_back(tmp);
      sprintf(outStr,"%1.1u_%5.5u_%6.6u.lab",gesture,runNum,exampleNum);
      of=fopen(outStr,"w");
      if(of==NULL)
      {
         fprintf(stderr,"could not open label file\n");
	 exit(0);
      }
      fprintf(of,"0   %lu   other\n",cnt*rate);
      fclose(of);

    }
    else
    {
      fprintf(stderr,"unrecognized gesture\n");
      return 1;
    }
  }
  else
  {
    fprintf(stderr,"wrong number of examples in gesture:%d\n",writebuf.size());
    return 1;
  }


  return 0;
}


///////////////////////////////////////////////////////
