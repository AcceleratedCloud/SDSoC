#ifndef HEADERS_H
#define HEADERS_H
#include<stdio.h>
#include<stdlib.h>
#include "structs.h"

#define __SDSOC__

#ifdef __SDSOC__
#include"sds_lib.h"
#endif

static const int NFEATS      = 80;
static const int BUFFER_SIZE = 20;
static const int NMAXRAT     = 7000;



void CholeskySolver(float A[NFEATS*NFEATS],float B[NFEATS],float X[NFEATS]);
int compareByUser(const void*,const void*);
int compareByMovie(const void*,const void*);
int findNumOfEntries(FILE*);
int initU(int,float**,int);
int initM(int,float**,int);
int fileExists(char* fileName);

void updateU(float** U,float** M ,info userInf[], int numUsers, float Sbuffer[BUFFER_SIZE*NFEATS],float* Rbuffer,float A[NFEATS*NFEATS],float V[NFEATS]);
void updateM(float** U,float** M ,info movieInf[],int numMovies,float Sbuffer[BUFFER_SIZE*NFEATS],float* Rbuffer,float A[NFEATS*NFEATS],float V[NFEATS]);
void bufferizeAndSend(float**,int*,float*,float*,float*,int,float*,float*);

void getDataFromFile(FILE*,char*,sparseEntry*);
void calculateRMSE(int nData ,info* Ru , int numUsers ,float** M, float** U);
int  exportRes(int nData,info* Ru,int numUsers,float** M,float** U);
//---HARDWARE-----
//specifies the system port
#pragma SDS data sys_port(Sbuffer:ACP,Rbuffer:ACP)

//specify data Mover
#pragma SDS data mem_attribute (Sbuffer:PHYSICAL_CONTIGUOUS|CACHEABLE,Rbuffer:PHYSICAL_CONTIGUOUS|CACHEABLE,Aout:PHYSICAL_CONTIGUOUS|CACHEABLE,Vout:PHYSICAL_CONTIGUOUS|CACHEABLE)
#pragma SDS data copy(Sbuffer[0:nratings*NFEATS],Rbuffer[0:nratings])
#pragma SDS data data_mover(Sbuffer:AXIDMA_SIMPLE , Rbuffer:AXIDMA_SIMPLE, Aout:AXIDMA_SIMPLE , Vout:AXIDMA_SIMPLE)

//accelerator interface
#pragma SDS data access_pattern(Sbuffer:SEQUENTIAL,Rbuffer:SEQUENTIAL,Aout:SEQUENTIAL,Vout:SEQUENTIAL)

void topLevelHW(float Sbuffer[NMAXRAT*NFEATS],float Rbuffer[NMAXRAT],int nratings,float Aout[NFEATS*NFEATS],float Vout[NFEATS]);
void mul_hw1(float _buffer1[BUFFER_SIZE][NFEATS],float _buffer2[NFEATS][BUFFER_SIZE],float _A[NFEATS][NFEATS]);
void mul_hw2(float _buffer1[BUFFER_SIZE][NFEATS],float _rbuff[BUFFER_SIZE],float _V[NFEATS]); 
void readSData(float* SBUFFER,float _buffer1[BUFFER_SIZE][NFEATS],float _buffer2[NFEATS][BUFFER_SIZE],int,int);
void readRData(float* RBUFFER,float _rbuff[BUFFER_SIZE],int,int);
//---DEBUG ----
void printResults(float** U,float** M,info* userInf,int numUsers);
void printArrA(float*);
void printArrV(float*);
 void printTheSystemToFile(float A[NFEATS*NFEATS],float X[NFEATS],float B[NFEATS]);
void printTheSystem(float A[NFEATS*NFEATS],float X[NFEATS],float B[NFEATS]);
int checkDataset(sparseEntry*,sparseEntry*,int);
void debug1(sparseEntry*,sparseEntry*,int);
void debug2(float** U , float** M,int,int);
void debug3(info*,int);
 void printBuffer(float* SBUFFER,int b);
#endif
