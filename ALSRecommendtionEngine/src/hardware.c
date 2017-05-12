#include <stdio.h>
#include <stdlib.h>
#include "headers.h"

void topLevelHW(float Sbuffer[NMAXRAT*NFEATS] ,float Rbuffer[NMAXRAT],int nratings ,float Aout[NFEATS*NFEATS],float Vout[NFEATS]){
	int b,i,j,k,r,c;
	float _buffer1[BUFFER_SIZE][NFEATS];
	float _buffer2[NFEATS][BUFFER_SIZE];
	float _rbuff[BUFFER_SIZE];
	float _A[NFEATS][NFEATS];
	float _V[NFEATS];
	float temp;
	int streampos;
	
initAout:for(i=0;i<NFEATS;i++){
initAin	:for(j=0;j<NFEATS;j++){
			_A[i][j] = 0;
		}
	}
init_V:for(i=0;i<NFEATS;i++){
		_V[i] = 0;
	}
	

	
	int iterations = nratings/BUFFER_SIZE+1;
	streampos=0;
	for(i=0;i<iterations;i++){
		readSData(Sbuffer,_buffer1,_buffer2,streampos,nratings);
		readRData(Rbuffer,_rbuff,streampos,nratings);
		mul_hw1(_buffer1,_buffer2,_A);
		mul_hw2(_buffer1,_rbuff,_V);
		streampos += BUFFER_SIZE;
	}
	

	
cpoutAout:for(i=0;i<NFEATS;i++){
cpoutAin:		for(j=0;j<NFEATS;j++){
			Aout[i*NFEATS+j] = _A[i][j];
		}
	}
cpoutV:	for(i=0;i<NFEATS;i++){
		Vout[i] = _V[i];
	}

}

void mul_hw1(float _buffer1[BUFFER_SIZE][NFEATS],float _buffer2[NFEATS][BUFFER_SIZE],float _A[NFEATS][NFEATS]){
	int i,j,r;
_A1:for(i=0;i<NFEATS;i++){
_A2:for(j=0;j<NFEATS;j++){
if(i<=j){
	    float result2 = _A[i][j];
	   	    _A3:for(r=0;r<BUFFER_SIZE;r++){
	    	result2 += _buffer2[i][r]*_buffer1[r][j];
	    		}
	    _A[i][j] = result2;
    }
    }
    }
}

void mul_hw2(float _buffer1[BUFFER_SIZE][NFEATS],float _rbuff[BUFFER_SIZE],float _V[NFEATS]){
	int i,j;
_V1:for(i=0;i<NFEATS;i++){
	   float result1= _V[i];
_V2:for(j=0;j<BUFFER_SIZE;j++){
	   result1+=_buffer1[j][i]*_rbuff[j];
   }
   _V[i] = result1;
   }
}

void readSData(float Sbuffer[NMAXRAT*NFEATS],float _buffer1[BUFFER_SIZE][NFEATS],float _buffer2[NFEATS][BUFFER_SIZE],int streampos,int nratings){
	float temp;
	int i,j,row;
COPY_IN1:for(i=0;i<BUFFER_SIZE;i++){
	 row = streampos*NFEATS;
COPY_IN2:for(j=0;j<NFEATS;j++){
		 temp = streampos<nratings?Sbuffer[row+j]:0;
		 _buffer1[i][j] = temp;
		 _buffer2[j][i] = temp;
	 }
	 streampos++;
	 }
}

void readRData(float Rbuffer[NMAXRAT],float _rbuff[BUFFER_SIZE],int streampos,int nratings){
	int i;
COPY_IN3:for(i=0;i<BUFFER_SIZE;i++){
#pragma HLS PIPELINE REWIND
			 
		 _rbuff[i] = (streampos+i)<nratings?Rbuffer[streampos+i]:0;
	 }
}
