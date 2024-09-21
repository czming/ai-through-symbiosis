/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
#ifndef HTK_HELPER_MONKEY_H_
#define HTK_HELPER_MONKEY_H_
/////////////////////////////////////////////////////
//This is a set of classes to help out with annoying old
//htk data formats. Tested on linux. The idea is to
//derive your own format from the HTKDataVector and
//fill in the data requirements later on.
/////////////////////////////////////////////////////
#include <queue>
using namespace std;

#define MIN_GESTURE_SPACING 20
#define MIN_FRAMES_FOR_GESTURE ((MIN_GESTURE_SPACING) + (0))
#define MAX_FRAMES_FOR_GESTURE 60
#define FRAME_RATE_100ns 333667
#define KIND_USER_DEFINED 9

class HTKDataVector{
 public:
  HTKDataVector();
  ~HTKDataVector(){};
  HTKDataVector(unsigned int numSamp, unsigned int period,
		unsigned short int size, unsigned short kind);

  unsigned int getSamples() { return nSamples_sw; };
  unsigned int getSampPeriod() { return sampPeriod_sw; };
  unsigned short int getSampSize() { return sampSize_sw; };
  unsigned short int getKind() { return parmKind_sw; }; 

 
 protected:
  //header data

  unsigned int nSamples;
  unsigned int nSamples_sw;

  unsigned int sampPeriod;
  unsigned int sampPeriod_sw;

  unsigned short int sampSize;
  unsigned short int sampSize_sw;
  
  unsigned short int parmKind;
  unsigned short int parmKind_sw;

};

class FreqHistData:public HTKDataVector{

 public:
  FreqHistData();
  ~FreqHistData();
  FreqHistData(unsigned int nElements,unsigned int buffSize);
  void QueueHistData(float *inSamp);
  int WriteHistData(unsigned int gesture,unsigned int runNum, unsigned int exampleNum);


 protected:
  deque<float*> writebuf;
  deque<float*> readbuf;
  unsigned int buffSize;
  unsigned int numElements;
};

class RawImgData:public HTKDataVector{

 public:
  RawImgData();
  ~RawImgData();
  RawImgData(unsigned int nElements,unsigned int buffSize);
  void QueueImgData(float *inSamp);
  int WriteImgData(unsigned int gesture,unsigned int runNum, unsigned int exampleNum);


 protected:
  deque<float*> writebuf;
  deque<float*> readbuf;
  unsigned int buffSize;
  unsigned int numElements;
};

#endif
