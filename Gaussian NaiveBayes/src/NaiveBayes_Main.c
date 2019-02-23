#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "accelerator.h"
#include "sds_lib.h"


#define PI 3.1415926
#define N_feat 784		// number of feaures
#define N_tr_data 2000  // Number of training data
#define N_ts_data 1000	// Number of test data
#define N_class 10		// Number of classes
#define Packs 20


typedef struct perf {
	float tot, cnt, calls;
} perf;

perf sw , hw ;

void reset(perf *est) { est->tot = est->cnt = est->calls = 0; }
void start(perf *est) { est->cnt = sds_clock_counter(); est->calls++; };
void stop(perf *est) { est->tot += (sds_clock_counter() - est->cnt); };
float avg_cpu_cycles( perf *est) {return (est->tot / est->calls); };

// Print Usage
void usage(const char* prog){

   printf("Read training data then classify test data using Naive Bayes:\nUsage:\n[options] training_data test_data\n Options:\n -d <int> Decsion rule.  1 = gaussian (default)\n \t\t\t 2 = multinomial\n \t\t\t 3 = bernoulli\n -a \t Smoothing parameter alpha. default 1.0 (Laplace)\n -v \t Verbose.\n");

}

void NBtraining(
				int DataPack,
				int n_per_class[N_class],
				float data[N_tr_data*N_feat],
				float priors[N_class],
				float means[N_class*N_feat],
				float variances[N_class*N_feat]) {

	int i = 0 , j = 0, class = 0, data_pointer[N_class + 1];
	float sums[N_feat], sq_sums[N_feat], sq_feature_means[N_feat];


	data_pointer[0] = 0;
	for (i = 0; i < N_class; i++){
		data_pointer[i+1] = (n_per_class[i]*N_feat + data_pointer[i]);
		priors[i] = n_per_class[i]/(float)DataPack;
	}
	for (class = 0; class < N_class ; class++){
		for (j = 0; j < N_feat; j++){
			sums[j] = 0;
			sq_sums[j] = 0;
		}

		for (i = 0; i < n_per_class[class]; i++){
			for (j = 0; j < N_feat; j++){
				sums[j] += data[data_pointer[class] + i*N_feat + j];
				sq_sums[j] +=  data[data_pointer[class] + i*N_feat + j] * data[data_pointer[class] + i*N_feat + j];
			}
		}
		for (j = 0; j < N_feat; j++){
			means[(class * N_feat) + j]=(sums[j]/(float)n_per_class[class]);
			sq_feature_means[j]=(sq_sums[j]/(float)n_per_class[class]);
			variances[(class * N_feat) + j] = sq_feature_means[j] - ((means[(class * N_feat) + j])*(means[(class * N_feat) + j]));
		}
	}
}

int NBprediction(	float data[N_feat],
					float means[N_class*N_feat],
					float variances[N_class*N_feat],
					float priors[N_class]){

    int i, j, prediction;
    float numerator, max_likelihood, d_Pi, threshold;

    d_Pi = 2 * M_PI	;
    prediction = 0;
    threshold = 0.000005;
    max_likelihood = -INFINITY;

    for (i = 0; i < N_class; i++){
	        numerator = log(priors[i]);
	        for (j = 0; j < N_feat; j++){
	            if (variances[i*N_feat + j] > threshold){
	                numerator += log(1/sqrt(d_Pi * variances[i*N_feat + j])) + ((-1*(data[j] - means[i*N_feat + j])*(data[j] - means[i*N_feat + j])) / (2 * variances[i*N_feat + j]));
				}
	        }
	        if (numerator > max_likelihood){
	            max_likelihood = numerator;
	            prediction = i;
	        }
    }
    return prediction;
}


int main(int argc, const char* argv[]){

	FILE *fp, *flp;
	float *means, *variances, *priors,*_means, *_variances, *_priors, *values, *data_kernel, ***data;
	float hw_cycles, sw_cycles;
  	int HW; // Decision Hardware
    int verbose = 0;  // Verbose
    int label, i, j, c, k, n_total = 0, prediction ,cor = 0, total = 0, offset, pck = 0, **n, *point, *DataPack;
	char line[10000], str[10000], *token;
	const char s[2] = ",";
	double time_spent_s , total_time, time_spent;
	
	data = malloc( Packs * sizeof(int**));
	for(i = 0; i < Packs; i++){
		data[i] = malloc(N_class * sizeof(int*));
		for (j = 0; j < N_class; j++){
			data[i][j] = malloc(N_tr_data*N_feat*sizeof(float));
		}
	}
	priors = sds_alloc(N_class * sizeof(float));
	_priors = sds_alloc(N_class * sizeof(float));
	means = sds_alloc(N_class* N_feat * sizeof(float));
	_means = sds_alloc(N_class* N_feat * sizeof(float));
	variances = sds_alloc(N_class * N_feat * sizeof(float));
	_variances = sds_alloc(N_class * N_feat * sizeof(float));
  	values = sds_alloc(N_feat * sizeof(float));
    n = sds_alloc(Packs*sizeof(int));
	for (i = 0; i < Packs; i++) n[i] = sds_alloc(N_class*sizeof(int));
	DataPack = sds_alloc(Packs * sizeof(int));
	data_kernel = sds_alloc(N_tr_data*N_feat*sizeof(float));

	printf("Press 1 for Hardware or 0 for Software\n");
	scanf("%d",&HW);
	
    if (argc < 3) {
        usage(argv[0]);
        return EXIT_SUCCESS;
	}

	printf("# called with: argc = %d ->", argc);

	for(i = 0; i < argc; i++) {
		printf("%s ", argv[i]);
		if( strcmp( argv[i], "-v" ) == 0 ) verbose = 1;
		if( strcmp( argv[i], "-h" ) == 0 ) {
			usage(argv[0]);
			return EXIT_SUCCESS;
		}
	}
	printf("\n");

	for (c = 0; c < Packs; c++){
		DataPack[c] = 0;
		for (i = 0; i < N_class; i++) {
			n[c][i] = 0;
		}
	}

	if ((fp=fopen(argv[argc-2],"r"))==NULL) {
		printf("Unable to read file %s.\n", argv[argc-2]);
	return(0);
	}
	while ((fgets(line, 10000, fp))!= NULL){
		if (n_total % N_tr_data == 0 && n_total != 0){
			pck++;
		}
		flp=fopen ("line.txt","w+");
		fputs(line,flp);
		rewind(flp);
		fscanf(flp, "%s", str);
		token = strtok(str, s);
		i = 0;
		while( token != NULL ) {
			if (i==0) {
				label = atoi(token);
			}
			if (i!=0) values[i-1]= atof(token);
			token = strtok(NULL, s);
			i++;
		}
		offset = n[pck][label];
		for( i = 0; i < N_feat; i++){
			data[pck][label][offset*N_feat + i] = values[i];
		}
		n[pck][label]++;
		DataPack[pck]++;
		n_total++;

		fclose(flp);
	}
	fclose(fp);


	//Train
    printf("Training:\n");
	reset(&sw);
	reset(&hw);
	total_time = 0;

	for (c = 0; c < pck+1; c++){
		int k = 0;
		for (i = 0 ; i < N_class ; i++){
			for (j = 0; j < n[c][i]*N_feat ; j++){
				data_kernel[k] = data[c][i][j];
				k++;
			}
		}
		if (HW == 1) {
			clock_t begin = clock();
			start(&hw);
			NBtraining_accel(DataPack[c], n[c], data_kernel, priors, means, variances);
			stop(&hw);
			clock_t end = clock();
			time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			total_time += time_spent;
		}
		else{
			clock_t begin_s = clock();
			start(&sw);
			NBtraining(DataPack[c], n[c], data_kernel, priors, means, variances);
			stop(&sw);
			clock_t end_s = clock();
			time_spent_s = (double)(end_s - begin_s) / CLOCKS_PER_SEC;
			total_time += time_spent_s;
		}

		for (k = 0; k < N_class; k++){
			_priors[k] += priors[k];
			for ( j = 0; j < N_feat; j++){
				_variances[k * N_feat + j] += variances[k * N_feat + j];
				_means[k * N_feat + j] += means[k * N_feat + j];
			}
		}
	}

	for (k = 0; k < N_class; k++){
		_priors[k] = _priors[k] / (float)(pck+1);
		for (j = 0; j < N_feat; j++){
			_variances[k * N_feat + j] = _variances[k * N_feat + j] / (float)(pck+1);
			_means[k * N_feat + j] = _means[k * N_feat + j] / (float)(pck+1);
		}
	}

	hw_cycles = avg_cpu_cycles(&hw);
	sw_cycles = avg_cpu_cycles(&sw);

	//Print results
	if (HW == 1){
		printf("Training Function has been accelerated\nHardware_Time = %f\nHardware_Cycles = %f\n",total_time,hw_cycles);
	}
	else{
		printf("Training Function in ARM processor(Sw-only)\nSoftware_Time = %f\nSoftware_Cycles = %f\n",total_time,sw_cycles);
	}


    //Classify
    printf("Classifying:\n");
	reset(&hw);
	reset(&sw);
	total_time = 0;

    if (verbose) printf("class\total_posrob\tresult\n");

	if ( ( fp = fopen( argv[argc-1] , "r" ) ) == NULL ) {
		printf("Unable to read file %s.\n", argv[argc-1]);
		return EXIT_SUCCESS;
	}

	while ((fgets(line, 10000, fp))!= NULL) {
		flp = fopen ("line.txt","w+");
		fputs(line,flp);
		rewind(flp);
		fscanf(flp, "%s", str);
		token = strtok(str, s);
		i = 0;
	   	while( token != NULL ) {
	   		if (i==0) label = atoi(token);
			if (i!=0) values[i-1]= atof(token);
	     	token = strtok(NULL, s);
	     	i++;
	   	}
		if (HW == 1){
			clock_t begin = clock();
			start(&hw);
			prediction = NBprediction_accel(values, _means, _variances, _priors);
			stop( &hw);
			clock_t end = clock();
			time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			total_time += time_spent;
		}
		else{
			clock_t begin_s = clock();
			start(&sw);
			prediction = NBprediction(values, _means, _variances, _priors);
			stop( &sw);
			clock_t end_s = clock();
			time_spent_s = (double)(end_s - begin_s) / CLOCKS_PER_SEC;
			total_time += time_spent_s;
		}

        if(prediction == label){
            if(verbose) printf("correct\n");
            cor++;
        }
        else{
            if(verbose) printf("incorrect\n");
        }
        total++;

		fclose(flp);
	}
    fclose(fp);

	hw_cycles = avg_cpu_cycles(&hw);
	sw_cycles = avg_cpu_cycles(&sw);

	// Print Result
	if (HW == 1){
		printf("Prediction Function has been accelerated\nHardware_Time = %f\nHardware_Cycles = %f\n",total_time,hw_cycles);
	}
	else{
		printf("Prediction Function in ARM processor(Sw-only)\nSoftware_Time = %f\nSoftware_Cycles = %f\n",total_time,sw_cycles);
	}
    printf ("Accuracy: %3.2f %% (%i/%i)\n", (100*(float)(cor)/total),cor,total);

    sds_free(priors);
	sds_free(_priors);
    sds_free(variances);
    sds_free(_variances);
    sds_free(means);
    sds_free(_means);
    sds_free(n);
    sds_free(values);
    sds_free(data_kernel);
    sds_free(DataPack);

    return EXIT_SUCCESS;
}
