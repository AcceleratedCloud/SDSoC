// Logistic Regression implementation using Batch Gradient Descent

#include <iostream>
#include <fstream>
#include <cmath>
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

int classify(float features[(1 + numFeatures)], float weights[numClasses * (1 + numFeatures)]){
	
	float prob = -1.0;
	int prediction = -1;
	
	for (int k = 0; k < numClasses; k++){
		
		float dot = 0.0;
		for (int j = 0; j < (1 + numFeatures); j++){
			dot += features[j] * weights[k * (1 + numFeatures) + j];
		}
		
		if (1.0 / (1.0 + exp(-dot)) > prob){
			prob = 1.0 / (1.0 + exp(-dot));
			prediction = k;
		}
		
	}
	
	return prediction;
	
}

int main(){
	
	perf_counter ctr;
	
	string train_file = "train.dat";
	int numSamples = 0;
	int chunkSize = 0;
	float alpha = 0.0;
	int iterations = 0;
	string test_file = "test.dat";
	
	string line;
	
	ifstream conf;
	conf.open("conf");
	while (getline(conf, line)){
		if (line.length()){
			if (line[0] != '#' && line[0] != ' '){
				vector<string> tokens = split(line);
				if (tokens[0] == "train_file"){
					train_file = tokens[1];
				}
				else if (tokens[0] == "numSamples"){
					numSamples = (int)atof(tokens[1].c_str());
				}
				else if (tokens[0] == "chunkSize"){
					chunkSize = (int)atof(tokens[1].c_str());
				}
				else if (tokens[0] == "alpha"){
					alpha = atof(tokens[1].c_str());
				}
				else if (tokens[0] == "iterations"){
					iterations = (int)atof(tokens[1].c_str());
				}		
				else if (tokens[0] == "test_file"){
					test_file = tokens[1];
				}
			}
		}
	}
	conf.close();
	
	cout << "* LogisticRegression Application *" << endl;
	cout << " # train file:               " << train_file << endl;
	cout << " # test file:                " << test_file << endl;
	
	float **data1, **data2, *data1_buf, *data2_buf, *weights, *gradients;
	data1 = (float **)malloc((int)(numSamples / chunkSize) * sizeof(float *));
	data2 = (float **)malloc((int)(numSamples / chunkSize) * sizeof(float *));
	if (!data1 || !data2){
		if (data1) free(data1);
		if (data2) free(data2);
		return(1);
	}
	for (int c = 0; c < (int)(numSamples / chunkSize); c++){
		data1[c] = (float *)malloc((int)(chunkSize * (numClasses + (1 + numFeatures)) / 2) * sizeof(float));
		data2[c] = (float *)malloc((int)(chunkSize * (numClasses + (1 + numFeatures)) / 2) * sizeof(float));
		if (!data1[c] || !data2[c]){
			if (data1[c]) free(data1[c]);
			if (data2[c]) free(data2[c]);
			return(1);
		}
	}
#ifndef __TEST__
	data1_buf = (float *)sds_alloc((int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2) * sizeof(float));
	data2_buf = (float *)sds_alloc((int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2) * sizeof(float));
	weights = (float *)sds_alloc(numClasses * (1 + numFeatures) * sizeof(float));
	gradients = (float *)sds_alloc(numClasses * (1 + numFeatures) * sizeof(float));
	if (!data1_buf || !data2_buf || !weights || !gradients){
		if (data1_buf) sds_free(data1_buf);
		if (data2_buf) sds_free(data2_buf);
		if (weights) sds_free(weights);
		if (gradients) sds_free(gradients);
		return(1);
	}
#else
	data1_buf = (float *)malloc((int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2) * sizeof(float));
	data2_buf = (float *)malloc((int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2) * sizeof(float));
	weights = (float *)malloc(numClasses * (1 + numFeatures) * sizeof(float));
	gradients = (float *)malloc(numClasses * (1 + numFeatures) * sizeof(float));
	if (!data1_buf || !data2_buf || !weights || !gradients){
		if (data1_buf) free(data1_buf);
		if (data2_buf) free(data2_buf);
		if (weights) free(weights);
		if (gradients) free(gradients);
		return(1);
	}
#endif

	ifstream train;
	train.open(train_file.c_str());
	int n = 0;
	int i = 0;
	int c = 0;
	while (getline(train, line) && (n < numSamples)){
		if (line.length()){
			if (line[0] != '#' && line[0] != ' '){
				vector<string> tokens = split(line);
				int label = (int)atof(tokens[0].c_str());
				if (i < (int)(chunkSize / 2)){
					for (int k = 0; k < numClasses; k++){
						if (k == label){
							data1[c][i * (numClasses + (1 + numFeatures)) + k] = 1.0;
						}
						else{
							data1[c][i * (numClasses + (1 + numFeatures)) + k] = 0.0;
						}
					}
					data1[c][i * (numClasses + (1 + numFeatures)) + numClasses] = 1.0;
					for (int j = 1; j < (1 + numFeatures); j++){
						data1[c][i * (numClasses + (1 + numFeatures)) + numClasses + j] = atof(tokens[j].c_str());
					}
					i++;
				}
				else{
					for (int k = 0; k < numClasses; k++){
						if (k == label){
							data2[c][(i - (int)(chunkSize / 2)) * (numClasses + (1 + numFeatures)) + k] = 1.0;
						}
						else{
							data2[c][(i - (int)(chunkSize / 2))  * (numClasses + (1 + numFeatures)) + k] = 0.0;
						}
					}
					data2[c][(i - (int)(chunkSize / 2))  * (numClasses + (1 + numFeatures)) + numClasses] = 1.0;
					for (int j = 1; j < (1 + numFeatures); j++){
						data2[c][(i - (int)(chunkSize / 2))  * (numClasses + (1 + numFeatures)) + numClasses + j] = atof(tokens[j].c_str());
					}
					i++;
				}
				n++;
				if (i == chunkSize){
					i = 0;
					c++;
				}
			}
		}
	}
	train.close();
	
	cout << "    * LogisticRegression Training *" << endl;
	cout << "     # numSamples:               " << numSamples << endl;
	cout << "     # chunkSize:                " << chunkSize << endl;
	cout << "     # numClasses:               " << numClasses << endl;
	cout << "     # numFeatures:              " << numFeatures << endl;
	cout << "     # alpha:                    " << alpha << endl;
	cout << "     # iterations:               " << iterations << endl;

	float _gradients[numClasses * (1 + numFeatures)];

	for (int k = 0; k < numClasses; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			weights[k * (1 + numFeatures) + j] = 0.0;
		}
	}

	for (int l = 0; l < iterations; l++){	

		for (int k = 0; k < numClasses; k++){
			for (int j = 0; j < numFeatures; j++){
				_gradients[k * (1 + numFeatures) + j] = 0.0;
			}
		}
		for (int c = 0; c < (int)(numSamples / chunkSize); c++){
			for (int ikj = 0; ikj < (int)(chunkSize * (numClasses + (1 + numFeatures)) / 2); ikj++){
				data1_buf[ikj] = data1[c][ikj];
				data2_buf[ikj] = data2[c][ikj];
			}
			ctr.start();
#ifndef __TEST__			
			LR_gradients_kernel_accel(data1_buf, data2_buf, weights, chunkSize, gradients);
#else			
			LR_gradients_kernel(data1_buf, data2_buf, weights, chunkSize, gradients);
#endif			
			ctr.stop();
			
			for (int k = 0; k < numClasses; k++){
				for (int j = 0; j < numFeatures; j++){
					_gradients[k * (1 + numFeatures) + j] += gradients[k * (1 + numFeatures) + j];
				}
			}
		}

		for (int k = 0; k < numClasses; k++){
			for (int j = 0; j < (1 + numFeatures); j++){
				weights[k * (1 + numFeatures) + j] -= (alpha / numSamples) * _gradients[k * (1 + numFeatures) + j];
			}
		}

	}
	
#ifndef __TEST__
	uint64_t cycles = ctr.avg_cpu_cycles();
	cout << "! Average time running LR_gradients_kernel in hardware: " << (int)(cycles / (666.67 * 1000)) << " msec" << endl;
#else
	clock_t cycles = ctr.avg_cpu_cycles();	
	cout << "! Average time running LR_gradients_kernel in software: " << (int)((cycles * 1000) / CLOCKS_PER_SEC) << " msec" << endl;
#endif

	ofstream w;
	w.open("weights.out");
	for (int k = 0; k < numClasses; k++){
		w << weights[k * (1 + numFeatures)];
		for (int j = 1; j < (1 + numFeatures); j++){
			w << "," << weights[k * (1 + numFeatures) + j];
		}
		w << endl;
	}
	w.close();
	
	cout << "    * LogisticRegression Testing *" << endl;
	
	float tr = 0.0;
	float fls = 0.0;
	float example[(1 + numFeatures)];

	ifstream test;
	test.open(test_file.c_str());
	while (getline(test, line)){
		if (line.length()){
			if (line[0] != '#' && line[0] != ' '){
				vector<string> tokens = split(line);
				int label = (int)atof(tokens[0].c_str());
				example[0] = 1.0;
				for (int j = 1; j < (1 + numFeatures); j++){
					example[j] = atof(tokens[j].c_str());
				}
				int prediction = classify(example, weights);
				if (prediction == label){
					tr++;
				}
				else{
					fls++;
				}
			}
		}
	}
	test.close();
	
	printf ("     # accuracy:       %1.3f (%i/%i)\n", (tr / (tr + fls)), (int)tr, (int)(tr + fls));
	printf ("     # true:           %i\n", (int)tr);
	printf ("     # false:          %i\n", (int)fls);

	for (int c = 0; c < (int)(numSamples / chunkSize); c++){
		free(data1[c]);
		free(data2[c]);
	}
	free(data1);
	free(data2);
#ifndef __TEST__
	sds_free(data1_buf);
	sds_free(data2_buf);
	sds_free(weights);
	sds_free(gradients);
#else
	free(data1_buf);
	free(data2_buf);
	free(weights);
	free(gradients);
#endif

	return(0);
	
}
