#include <stdio.h>
#include <math.h>
#include "headers.h"
#include "stdio.h"


void calculateRMSE(int nData ,info* Ru ,int numUsers , float** M , float** U){
	int i,k,userId,movieId,nRatings,j;
	int* movies;
	double RMSE = 0.0,temp = 0.0;
	double sum = 0.0;
	float* ratings;
	for(i=0;i<numUsers;i++){
		userId     = Ru[i].id;
		nRatings   = Ru[i].numRatings;
		movies     = Ru[i].rId;
		ratings    = Ru[i].rating;
		for(j=0;j<nRatings;j++){
			movieId = movies[j];
			temp = 0.0;
			for(k = 0;k<NFEATS;k++  ){
				temp += U[userId][k]*M[movieId][k];
			}
			sum = sum + (ratings[j] - temp)*(ratings[j] - temp);
		}
	}
	RMSE = sqrtf(sum/(float)nData);
	printf("RMSE = %g \n",RMSE);
}

int initU(int maxRating,float** U,int numUsers ){
	int i,j;
	//initialize with random values 
	for(i=0;i<numUsers;i++){
		for(j=0;j<NFEATS;j++){
			U[i][j] = ((float)rand()/(float)(RAND_MAX))*maxRating;
		}
	}
	return 1;
}

int initM(int maxRating , float** M , int numMovies){
	int i,j;
	//initialize with rabdom values
	for(i = 0;i<numMovies;i++){
		for(j=0;j<NFEATS;j++){
			M[i][j] = ((float)rand()/(float)(RAND_MAX))*maxRating;
		}
	}

	return 1;
}

int compareByUser(const void* a, const void* b){
	//used for quicksort
	sparseEntry *sparseEntryA = (sparseEntry*)a;
	sparseEntry *sparseEntryB = (sparseEntry*)b;
	return(sparseEntryA->rowUser - sparseEntryB->rowUser);
}

int compareByMovie(const void* a, const void* b){
	//used for quicksort
	sparseEntry *sparseEntryA = (sparseEntry*)a;
	sparseEntry *sparseEntryB = (sparseEntry*)b;
	return(sparseEntryA->colMovie - sparseEntryB->colMovie);
}

int exportRes(int nData,info* Ru,int numUsers,float** M,float** U){
	int i,k,userId,movieId,nRatings,j;
	int* movies;
	float* ratings;
	float temp;
	FILE* fp = fopen("res.dat","w");
	for(i=0;i<numUsers;i++){
		userId     = Ru[i].id;
		nRatings   = Ru[i].numRatings;
		movies     = Ru[i].rId;
		ratings    = Ru[i].rating;
		for(j=0;j<nRatings;j++){
			movieId = movies[j];
			temp = 0.0;
			for(k = 0;k<NFEATS;k++  ){
				temp += U[userId][k]*M[movieId][k];
			}
			fprintf(fp,"[%d][%d] - prediction: %g - actual %g - error %g \n",userId,movieId,temp,ratings[j],temp-ratings[j]);
		}
	}
	fclose(fp);
	return 1;
}
