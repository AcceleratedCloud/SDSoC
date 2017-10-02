#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include <stdint.h>
#include <stdlib.h>
#ifndef __TEST__
#include <sds_lib.h>
#else
#include <time.h>
#endif

#include "accelerator.h"

using namespace std;

#ifndef __TEST__
class perf_counter{
public:
	uint64_t tot, cnt, calls;
	perf_counter() : tot(0), cnt(0), calls(0) {};
	inline void reset(){
		tot = cnt = calls = 0; 
	}
	inline void start(){ 
		cnt = sds_clock_counter(); calls++; 
	};
	inline void stop(){ 
		tot += (sds_clock_counter() - cnt); 
	};
	inline uint64_t avg_cpu_cycles(){ 
		return (tot / calls); 
	};
};
#else
class perf_counter{
public:
	clock_t tot, cnt, calls;
	perf_counter() : tot(0), cnt(0), calls(0) {};
	inline void reset(){
		tot = cnt = calls = 0; 
	}
	inline void start(){ 
		cnt = clock(); calls++; 
	};
	inline void stop(){ 
		tot += (clock() - cnt); 
	};
	inline clock_t avg_cpu_cycles(){ 
		return (tot / calls); 
	};
};
#endif

vector<string> split(const string &s){
	vector<string> elements;
	stringstream ss(s);
	string item;
	while (getline(ss, item)){
		size_t prev = 0;
		size_t pos;
		while ((pos = item.find_first_of("(,[])=", prev)) != std::string::npos){
			if (pos > prev) elements.push_back(item.substr(prev, pos - prev));
			prev = pos + 1;
		}
		if (prev < item.length()) elements.push_back(item.substr(prev, std::string::npos));
	}
	return elements;
}

int main(){
	
	perf_counter ctr;
	
	string data_file = "chunk.dat";
	int chunkSize = 0;
	
	string line;
	
	ifstream conf;
	conf.open("conf");
	while (getline(conf, line)){
		if (line.length()){
			if (line[0] != '#' && line[0] != ' '){
				vector<string> tokens = split(line);
				if (tokens[0] == "data_file"){
					data_file = tokens[1];
				}
				else if (tokens[0] == "chunkSize"){
					chunkSize = (int)atof(tokens[1].c_str());
				}
			}
		}
	}
	conf.close();

	cout << "* KMeans centroids_kernel Testing *" << endl;
	cout << " # data file:                " << data_file << endl;
	cout << " # chunkSize:                " << chunkSize << endl;
	cout << " # numClusters:              " << numClusters << endl;
	cout << " # numFeatures:              " << numFeatures << endl;
	
	float *data1, *data2, *centroids, *counts_sums;
#ifndef __TEST__
	data1 = (float *)sds_alloc((int)(chunkSizeMax * numFeatures / 2) * sizeof(float));
	data2 = (float *)sds_alloc((int)(chunkSizeMax * numFeatures / 2) * sizeof(float));
	centroids = (float *)sds_alloc(numClusters * numFeatures * sizeof(float));
	counts_sums = (float *)sds_alloc(numClusters * (1 + numFeatures) * sizeof(float));
	if (!data1 || !data2 || !centroids || !counts_sums){
		if (data1) sds_free(data1);
		if (data2) sds_free(data2);
		if (centroids) sds_free(centroids);
		if (counts_sums) sds_free(counts_sums);
		return(1);
	}
#else	
	data1 = (float *)malloc((int)(chunkSizeMax * numFeatures / 2) * sizeof(float));
	data2 = (float *)malloc((int)(chunkSizeMax * numFeatures / 2) * sizeof(float));
	centroids = (float *)malloc(numClusters * numFeatures * sizeof(float));
	counts_sums = (float *)malloc(numClusters * (1 + numFeatures) * sizeof(float));
	if (!data1 || !data2 || !centroids || !counts_sums){
		if (data1) free(data1);
		if (data2) free(data1);
		if (centroids) free(centroids);
		if (counts_sums) free(counts_sums);
		return(1);
	}		
#endif
	
	ifstream chunk;
	chunk.open(data_file.c_str());
	int i = 0;
	while (getline(chunk, line) && (i < chunkSize)){
		if (line.length()){
			if (line[0] != '#' && line[0] != ' '){
				vector<string> tokens = split(line);
				if (i < (int)(chunkSize / 2)){
					for (int j = 0; j < numFeatures; j++){
						data1[i * numFeatures + j] = atof(tokens[j + 1].c_str());
					}
					i++;
				}
				else{
					for (int j = 0; j < numFeatures; j++){
						data2[(i - (int)(chunkSize / 2)) * numFeatures + j] = atof(tokens[j + 1].c_str());
					}
					i++;
				}
			}
		}
	}
	chunk.close();
	
	for (int k = 0; k < numClusters; k++){
		for (int j = 0; j < numFeatures; j++){
			centroids[k * numFeatures + j] = data1[k * numFeatures + j];
		}
	}
	
	ctr.start();
#ifndef __TEST__			
	KM_centroids_kernel_accel(data1, data2, centroids, chunkSize, counts_sums);
#else	
	KM_centroids_kernel(data1, data2, centroids, chunkSize, counts_sums);
#endif
	ctr.stop();

#ifndef __TEST__
	uint64_t cycles = ctr.avg_cpu_cycles();
	cout << "! Time running KMeans centroids_kernel in hardware: " << (int)(cycles / (666.67 * 1000)) << " msec" << endl;
#else	
	clock_t cycles = ctr.avg_cpu_cycles();	
	cout << "! Time running KMeans centroids_kernel in software: " << (int)((cycles * 1000) / CLOCKS_PER_SEC) << " msec" << endl;
#endif
	
	ofstream cs;
	cs.open("counts_sums.out");
	for (int k = 0; k < numClusters; k++){
		cs << (int)counts_sums[k * (1 + numFeatures)];
		for (int j = 1; j < (1 + numFeatures); j++){
			cs << "," << counts_sums[k * (1 + numFeatures) + j];
		}
		cs << endl;
	}
	cs.close();

#ifndef __TEST__
	sds_free(data1);
	sds_free(data2);
	sds_free(centroids);
	sds_free(counts_sums);
#else	
	free(data1);
	free(data2);
	free(centroids);
	free(counts_sums);
#endif

	return(0);
	
}
