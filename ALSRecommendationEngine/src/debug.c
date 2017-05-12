#include <stdio.h>
#include "headers.h"
#include "structs.h"

//---------------------------------------------------------------
//---------------------------------------------------------------
//DEBUGING - DEBUGING - DEBUGING - DEBUGING - DEBUGING - DEBUGING
//---------------------------------------------------------------
//---------------------------------------------------------------
void printArrA(float* A){
	int i,j;
	printf("----A----\n");
	for(i=0;i<NFEATS;i++){
		for(j=0;j<NFEATS;j++){
			printf("%g ",A[i*NFEATS+j]);	
		}
		printf("\n");
	}
}

void printArrV(float* V){
	int i,j;
	printf("----V-----");
	for(i=0;i<NFEATS;i++){
		printf(" %g",V[i]);
	}
	printf("\n");
}

void printTheSystemToFile(float A[NFEATS*NFEATS],float X[NFEATS],float B[NFEATS]){
	int i,j;
	FILE* fd = fopen("results.sys","a");
	for(i=0;i<NFEATS;i++){
		for(j=0;j<NFEATS;j++){
			fprintf(fd,"%g ",A[i*NFEATS+j]);
		}
		fprintf(fd,"\n");
	}
	fprintf(fd,"\n");

	for(i=0;i<NFEATS;i++){
		fprintf(fd,"%g\n",X[i]);
	}
	fprintf(fd,"\n");

	for(i=0;i<NFEATS;i++){
		fprintf(fd,"%g\n",B[i]);
	}

	fprintf(fd,"\n");
	fclose(fd);
}

void printTheSystem(float A[NFEATS*NFEATS],float X[NFEATS],float B[NFEATS]){
	int i,j;
	printf("SYSTEM\n");
	printf("----A----\n");
	for(i=0;i<NFEATS;i++){
		for(j=0;j<NFEATS;j++){
			printf("%g ",A[i*NFEATS+j]);
		}
		printf("\n");
	}
	printf("\n----X----\n");
	for(i=0;i<NFEATS;i++){
		printf("%g\n",X[i]);
	}
	printf("----B----\n");
	for(i=0;i<NFEATS;i++){
		printf("%g\n",B[i]);
	}
}

int checkDataset(sparseEntry* Ru , sparseEntry* Rm , int nData){
	int i,k;
	printf("\n Checking Users \n");
	k=0;
	for(i=0;i<nData;i++){
		if(Ru[i].rowUser == k){
			continue;
		}
		else{
			k++;
			if(Ru[i].rowUser!=k){
				printf("Dataset Error 'Incobatible User IDs'\n");
				return -1;
			}
		}
	}

	printf("Checking Movies \n");
	k=0;
	for(i=0;i<nData;i++){
		if(Rm[i].colMovie == k){
			continue;
		}
		else{
			k++;
			if(Rm[i].colMovie!=k){
				printf("Dataset Error 'Incobatible Movie IDs'\n");
				printf("k = %d Rm[%d].colMovie = %d ", k ,i-1,Rm[i-1].colMovie);
				printf("k = %d Rm[%d].colMovie = %d ", k ,i, Rm[i].colMovie);
				return -1;
			}
		}
	}

	printf("Dataset OK\n");
	return 1;
}

void debug1(sparseEntry* Ru , sparseEntry* Rm , int nData){
	int i;
	printf("\nUSER|MOVIE|RATING\n");
	for(i = 0 ; i<nData ;i++  ){
		printf("%d|%d|%g \n",Ru[i].rowUser,Ru[i].colMovie,Ru[i].rating);
	}
	printf("\n");
	for(i = 0;i<nData;i++){
		printf("%d|%d|%g \n",Rm[i].rowUser,Rm[i].colMovie,Rm[i].rating);
	}
	printf("\n Total Data = %d \n",nData);

}

void debug2(float** U , float** M,int numUsers,int numMovies){
	int i,j;
	printf("\n----U----\n");
	printf("NUSERS = %d,NMOVIES = %d,NFEATS = %d \n",numUsers,numMovies,NFEATS);
	for(i=0;i<numUsers;i++){
		for(j=0;j<NFEATS;j++){
			printf("%g ",U[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("\n----M----\n");
	for(i=0;i<numMovies;i++){
		for(j=0;j<NFEATS;j++){
			printf("%g ",M[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("numUsers = %d\nnumMovies = %d\nnumFeatures = %d\n",numUsers,numMovies,NFEATS);

}

void debug3(info* A,int n){
	int i,j;
	for(i=0;i<n;i++){
		printf("id: %d | numRatings: %d\n",A[i].id,A[i].numRatings);
		for(j=0;j<A[i].numRatings;j++){
			printf("|%d-%g",A[i].rId[j],A[i].rating[j]);
		}
		printf("|\n");
	}
}

void printBuffer(float* SBUFFER,int b){
	int i,j;
	printf("---BUFFER[%d]---\n",b);
	for(i=0;i<BUFFER_SIZE;i++){
		for(j=0;j<NFEATS;j++){
			printf("%g ",SBUFFER[i*NFEATS+j]);
		}
		printf("\n");
	}
}
