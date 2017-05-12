#include"structs.h"
#include"headers.h"
#include <string.h>
#include <unistd.h>
#include <math.h>

void updateU(float** U,float** M ,info userInf[], int numUsers, float Sbuffer[BUFFER_SIZE*NFEATS],float Rbuffer[NMAXRAT] , float A[NFEATS*NFEATS],float V[NFEATS]){
	int i,j;
	for(i=0;i<numUsers;i++){
		int*   sel      = userInf[i].rId;
		float* ratings  = userInf[i].rating;
		int    nratings = userInf[i].numRatings;
		int    id       = userInf[i].id;
		bufferizeAndSend(M,sel,ratings,A,V,nratings,Sbuffer,Rbuffer);
		for(j=0;j<NFEATS;j++)A[j*NFEATS+j]+=0.01*nratings;
		CholeskySolver(A,V,U[id]);
	}
	
}

void updateM(float** U,float** M ,info movieInf[], int numMovies, float Sbuffer[BUFFER_SIZE*NFEATS],float Rbuffer[NMAXRAT] , float A[NFEATS*NFEATS],float V[NFEATS]){
	int i,j;
        for(i=0;i<numMovies;i++){
        	int*   sel      = movieInf[i].rId;
        	float* ratings  = movieInf[i].rating;
	        int    nratings = movieInf[i].numRatings;
	        int    id       = movieInf[i].id;
	        bufferizeAndSend(U,sel,ratings,A,V,nratings,Sbuffer,Rbuffer);
			for(j=0;j<NFEATS;j++)A[j*NFEATS+j]+=0.01*nratings;
	        CholeskySolver(A,V,M[id]);
	}
}


void bufferizeAndSend(float** M,int* sel,float* ratings,float* A,float* V,int nratings,float* Sbuffer,float* Rbuffer){

	int i,j,b,buffIndx=0,rbuffIndx,mov,k1=0;
	buffIndx=0;
	for(i=0;i<nratings;i++)
	{
		mov = sel[i];
		for(j=0;j<NFEATS;j++)
		{
			Sbuffer[buffIndx] = M[mov][j];
			buffIndx++;
		}
		Rbuffer[i] = ratings[i];
	}

	topLevelHW(Sbuffer,Rbuffer,nratings,A,V);
}

void CholeskySolver(float A[NFEATS*NFEATS],float B[NFEATS],float X[NFEATS]){
	int i,j,k;
	float L[NFEATS*NFEATS];
	float Y[NFEATS];

	for (i = 0; i < NFEATS; i++)
		for (j = 0; j < (i+1); j++) {
			float s = 0;
			float value,temp;
			for (k = 0; k < j; k++)
				s += L[i*NFEATS+k] * L[j*NFEATS+k];

			temp = i<=j?A[i*NFEATS+j]:A[j*NFEATS+i];
			value = (i == j)?sqrtf(A[i*NFEATS+i] - s):(1.0 / L[j*NFEATS+j] * (temp - s));
			L[i*NFEATS+j] = value;
		}

	for(i=0;i<NFEATS;i++){
		Y[i] = B[i];
		for(j=0;j<i;j++){
			Y[i] -= L[i*NFEATS+j]*Y[j];
		}
		Y[i]/=L[i*NFEATS+i];
	}

	for(i=NFEATS-1;i>=0;i--){
		X[i] = Y[i];
		for(j=i+1;j<NFEATS;j++){
			X[i]-=X[j]*L[j*NFEATS+i];
		}
		X[i]/=L[i*NFEATS+i];
	}
}




