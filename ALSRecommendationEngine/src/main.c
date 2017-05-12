#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<string.h>
#include "headers.h"
#ifdef __SDSOC__
#include"sds_lib.h"
#endif
#include "structs.h"



void usage(){
	printf("---------USAGE---------\n\n");
	printf("./ALSrcmnt  --> this screen \n");
	printf("./ALSrcmnt  <file.dat>  <delimeter> <debug 0/1>\n");
	printf("Recomended file format: userID<delimeter>itemID<delimete>Rating\n");
	printf("file must be sorted according to usersID\n");
	printf("All Users from 0 to numUsers Exist, There are no holes\n");
	printf("All Movies from 0 to numMovies Exist, There are no holes \n");
	printf("this means All Users have rated at least one Movie \n");
	printf("this means each Movies is rated at least once \n");
	printf("-------------------------------------------------------------------\n\n");
}


int main(int argc, char *argv[] ){

#ifdef __SDSOC__
	unsigned long long int total_time_stamp;
	unsigned long long int total_no_read_stamp;
	unsigned long long int total_time;
	unsigned long long int total_no_read;
	unsigned long long int finish_stamp;

	total_time_stamp = sds_clock_counter();
#endif

	int     i,j;
	int     nData,numMovies=1,numUsers=1;
	char*   fileName;
	char*   delimeter;
	float   lamda;
	float** U;
	float** M;
	int     debug;
	int 	iterations;
	FILE*   file;
	sparseEntry* Ru;
	sparseEntry* Rm;
//-----READ COMMAND LINE ARGUMENTS------------
	if(argc < 6){
		usage();
		return 0;
	}

	fileName   = argv[1];
	delimeter  = argv[2];
	lamda      = atof(argv[3]);
	iterations = atoi(argv[4]);
	debug      = atoi(argv[5]);
//------------------------------------------

//----READ DATA FROM FILE-------------------
	if(fileExists(fileName)){
		file = fopen(fileName,"r");

		if(file == NULL) 
			return -1;

		nData = findNumOfEntries(file);
		fclose(file);

		Ru = (sparseEntry*)malloc(nData*sizeof(sparseEntry));
		file = fopen(fileName,"r");

		if(file == NULL) 
			return -1;

		getDataFromFile(file,delimeter,Ru);
		fclose(file);

	}
	else
		return -2;		
#ifdef __SDSOC__
	total_no_read_stamp = sds_clock_counter();
#endif
//------------------------------------------------------------


	Rm = (sparseEntry*)malloc(nData*sizeof(sparseEntry));

	for(i=0;i<nData;i++){
		Rm[i] = Ru[i];	
	}

	qsort(Ru,nData,sizeof(sparseEntry),compareByUser);
	qsort(Rm,nData,sizeof(sparseEntry),compareByMovie);

	if(!checkDataset(Ru,Rm,nData)) 
		return -4;

	if(debug)
		debug1(Ru,Rm,nData);

	for(i=1;i<nData;i++){
		if(Ru[i].rowUser!=Ru[i-1].rowUser)
			numUsers++;
		if(Rm[i].colMovie!=Rm[i-1].colMovie)
			numMovies++;
	}

//---------ALLOCATE AND INIT THE ESTIMATORS----------------------------
	U = (float**)malloc(numUsers*sizeof(float*));
	M = (float**)malloc(numMovies*sizeof(float*));
	for(i=0;i<numUsers;i++){
		U[i] = (float*)malloc(NFEATS*sizeof(float));
	}
	for(i=0;i<numMovies;i++){
		M[i] = (float*)malloc(NFEATS*sizeof(float));
	}

	//UPDATEFLAG -> average RAting per Movie needed HERE
	initU(5,U,numUsers);
	initM(5,M,numMovies);

	if(debug)
		debug2(U,M,numUsers,numMovies);
//------------------------------------------------------------------

//------STRUCTURE DATA BY USER--------------------------------------
	info* userInf  = (info*)malloc(numUsers*sizeof(info));
	info* movieInf = (info*)malloc(numMovies*sizeof(info));
	int c = 0;
	int id = 0;
	for(i=0;i<=nData;i++){

		if(i!=nData && Ru[i].rowUser==id){
			c++;
		}
		else{
			userInf[id].id = id;
			userInf[id].numRatings = c;
			userInf[id].rating     = (float*)malloc(c*sizeof(float));
			userInf[id].rId        = (int*)malloc(c*sizeof(int));

			for(j=0;j<c;j++){
				userInf[id].rId[j]      = Ru[i-c+j].colMovie; 
				userInf[id].rating[j]   = Ru[i-c+j].rating;	
			}
			id++;
			c=1;
		}
	}
//---------------------------------------------------------------------------

	free(Ru);

//----------STRUCTURE DATA BY MOVIE----------------------------------------
	c=0;
	id = 0;

	for(i=0;i<=nData;i++){

		if(i!=nData && Rm[i].colMovie==id){
			c++;
		}
		else{
			movieInf[id].id = id;
			movieInf[id].numRatings = c;
			movieInf[id].rating     = (float*)malloc(c*sizeof(float));
			movieInf[id].rId        = (int*)malloc(c*sizeof(int));

			for(j=0;j<c;j++){
				movieInf[id].rId[j]      = Rm[i-c+j].rowUser;      
				movieInf[id].rating[j]   = Rm[i-c+j].rating;
			}
			id++;
			c=1;
		}      
	}


	if(debug){
		printf("----USER INFO----\n");
		debug3(userInf,numUsers);
		printf("----MOVIE INFO----\n");
		debug3(movieInf,numMovies);
	}
//-------------------------------------------------------------------------

	free(Rm);

//------START LEARNING--------------------------------------------------
#ifdef __SDSOC__
	printf("allocating SBUFFER \n");
	float* Sbuffer = (float*)sds_alloc(NMAXRAT*NFEATS*sizeof(float));
	printf("allocating Rbuffer\n");
	float* Rbuffer = (float*)sds_alloc(NMAXRAT*sizeof(float));
	printf("allocating A\n");
	float* A       = (float*)sds_alloc(NFEATS*NFEATS*sizeof(float));
	printf("allocating V\n");
	float* V       = (float*)sds_alloc(NFEATS*sizeof(float));
#else
	float* Sbuffer = (float*)malloc(NMAXRAT*NFEATS*sizeof(float));
	float* Rbuffer = (float*)malloc(NMAXRAT*sizeof(float));
	float* A       = (float*)malloc(NFEATS*NFEATS*sizeof(float));
	float* V       = (float*)malloc(NFEATS*sizeof(float));
#endif
	calculateRMSE(nData,userInf,numUsers,M,U);

	for(i=0;i<iterations;i++){
		printf("#-#-#-#- ITERATION %d -#-#-#-#\n",i);
		updateU(U,M,userInf,numUsers,Sbuffer,Rbuffer,A,V);
		if(debug){
			printf("updated Users\n");
			debug2(U,M,numUsers,numMovies);
		}
		updateM(U,M ,movieInf,numMovies,Sbuffer,Rbuffer,A,V);
		if(debug){
			printf("updated Movies\n");
			debug2(U,M,numUsers,numMovies);
		}

		calculateRMSE(nData,userInf,numUsers,M,U);
	}


#ifdef __SDSOC__
	finish_stamp = sds_clock_counter();
	total_time        = finish_stamp - total_time_stamp;
	total_no_read     = finish_stamp - total_no_read_stamp;

	printf("\n-----------------\n");
	printf("total time         = %lld\n",total_time);
	printf("total time no read = %lld\n",total_no_read);
	printf("-----------------\n");
#endif

//-----FREE MEMORY----- 
		for(i=0;i<numUsers;i++){
			free(userInf[i].rating);
			free(userInf[i].rId);
			free(U[i]);
		}
		free(U);
		free(userInf);
		for(i=0;i<numMovies;i++){
			free(movieInf[i].rating);
			free(movieInf[i].rId);
			free(M[i]);
		}
		free(M);
		free(movieInf);
		#ifdef __SDSOC__
		sds_free(A);
		sds_free(V);
		sds_free(Sbuffer);
		sds_free(Rbuffer);
		#else
		free(A);
		free(V);
		free(Sbuffer);
		free(Rbuffer);
		#endif
		return 0;
}


