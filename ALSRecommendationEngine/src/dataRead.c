#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "headers.h"
#include "structs.h"

int findNumOfEntries(FILE* file){
	char    *line = NULL;
	size_t  len   = 0;
	ssize_t read;
	int     k     = 0;
	printf("Reading Dataset\n");
	while(1){
		read = getline(&line,&len,file);
		if(read==-1) break;
		else if(line[0]=='#'||line[0]=='\0'||!strcmp(line,"")||!strcmp(line," ")||line[0]=='\n'){
			continue;
		}
		k++;
	}
	printf("Dataset Entries: %d\n",k);
	free(line);
	return k;
}


void getDataFromFile(FILE *file,char* delimeter, sparseEntry* R){
	char    *line = NULL;
	size_t  len   = 0;
	ssize_t read;
	int     k     = 0;
	printf("Reading Dataset\n");

	while(1){
		read = getline(&line,&len,file);
		if(read==-1) break;
		else if(line[0]=='#'||line[0]=='\0'||!strcmp(line,"")||!strcmp(line," ")||line[0]=='\n'){
			continue;
		}
		R[k].rowUser = atoi(strtok(line,delimeter));
		R[k].colMovie = atoi(strtok(NULL,delimeter));
		R[k].rating = atof(strtok(NULL,delimeter));
		k++;
	}
	free(line);
}

int fileExists(char* fileName){
	if(access(fileName,F_OK)!=-1){
		return 1;
	}
	else{
		printf("Dataset-File doesnt exist\n");
		return 0;
	}				     
}


