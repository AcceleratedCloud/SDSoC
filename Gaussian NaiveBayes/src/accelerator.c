#include "accelerator.h"
#include <math.h>
#include <stdio.h>


void NBtraining_accel(
				int DataPack,
				int n_per_class[N_class],
				float data[N_tr_data*N_feat],
				float priors[N_class],
				float means[N_class*N_feat],
				float variances[N_class*N_feat]) {

	int i = 0 , j = 0, class = 0, k = 0, pnt, data_pointer[N_class + 1];
	float var[N_feat], sums[N_feat], sq_sums[N_feat], feature_means[N_feat], sq_feature_means[N_feat], ldata[N_feat];

	#pragma HLS array_partition variable = data_pointer complete
	#pragma HLS array_partition variable = var block factor=28
	#pragma HLS array_partition variable = sums block factor=28
	#pragma HLS array_partition variable = sq_sums block factor=28
	#pragma HLS array_partition variable = feature_means block factor=28
	#pragma HLS array_partition variable = sq_feature_means block factor=28
	#pragma HLS array_partition variable = ldata block factor=28


	data_pointer[0] = 0;
	for (class = 0; class < N_class; i++){
	#pragma HLS pipeline II=1
		data_pointer[class+1] = (n_per_class[class]*N_feat + data_pointer[class]);
		priors[class] = n_per_class[class]/(float)DataPack;
	}

	for (class = 0; class < N_class ; class++){
		pnt = data_pointer[class];
		for ( k = 0; k < 28; k++){
		#pragma HLS pipeline II=1
			for (j = 0; j < N_feat; j+=28){
				sums[k+j] = 0;
				sq_sums[k+j] = 0;
			}
		}

		for (i = 0; i < n_per_class[class]; i++){
		#pragma HLS loop_tripcount min=1 max = 250

			for (j = 0; j < N_feat; j++){
			#pragma HLS pipeline II=1
				ldata[j] = data[data_pointer[class] + i*N_feat + j];
			}


			for ( k = 0; k < 28; k++){
			#pragma HLS pipeline II=1
				for (j = 0; j < N_feat; j+=28){
					sums[k+j] += ldata[k+j];
					sq_sums[k+j] +=  ldata[k+j] * ldata[k+j];
				}
			}
		}
		for ( k = 0; k < 28; k++){
			for (j = 0; j < N_feat; j+=28){
			#pragma HLS pipeline II=1
				feature_means[j+k]=(sums[j+k]/(float)n_per_class[class]);
				sq_feature_means[j+k]=(sq_sums[j+k]/(float)n_per_class[class]);
			}
		}		
				
		for ( k = 0; k < 28; k++){
			for (j = 0; j < N_feat; j+=28){
			#pragma HLS pipeline II=1
				var[k+j] = sq_feature_means[j+k] - ((feature_means[k+j])*(feature_means[k+j]));
			}
		}

		for (j = 0; j < N_feat; j++){
		#pragma HLS pipeline II=1
			means[(class * N_feat) + j ] = feature_means[j];
			variances[(class * N_feat) + j ] = var[j];
		}
	}
}


int NBprediction_accel( float data[N_feat],
                        float means[N_class*N_feat],
                        float variances[N_class*N_feat] ,
                        float priors[N_class] ) {

    int i, j, prediction;
    float threshold, numerator, max_likelihood, lvar[N_class][N_feat], lpr[N_class], lmean[N_class][N_feat], ldata[N_feat], d_Pi, temp[6], A[N_feat], B[N_feat];

    #pragma HLS array_partition variable=A cyclic factor = 56
    #pragma HLS array_partition variable=B cyclic factor = 56
    #pragma HLS array_partition variable=temp complete
    #pragma HLS array_partition variable=lvar cyclic factor = 56 dim=2
    #pragma HLS array_partition variable=lpr complete
    #pragma HLS array_partition variable=lmean cyclic factor = 56 dim=2
    #pragma HLS array_partition variable=ldata cyclic factor = 56

	#pragma HLS dataflow

    d_Pi = 2 * M_PI	;
    prediction = 0;
    threshold = 0.000005;
    max_likelihood = -INFINITY;

    for ( i = 0; i < N_class; i++ ){
    #pragma HLS pipeline II=1
        lpr[i] = priors[i];
    }

    for ( i = 0; i < N_class; i++ ){
	#pragma HLS pipeline II=1
        for ( j = 0; j < N_feat; j++ ){
            lvar[i][j] = variances[i*N_feat + j];
            lmean[i][j] = means[i*N_feat + j];
        }
    }

    for ( i = 0; i < N_feat; i++ ){
    #pragma HLS pipeline II=1
        ldata[i] = data[i];
    }


    for (i = 0; i < N_class; i++){
	#pragma HLS unroll factor=2
        numerator = log(lpr[i]);

        for (j = 0; j < N_feat; j++){
		#pragma HLS pipeline II=1
            if (lvar[i][j] > threshold){
                temp[0] = ldata[j] - lmean[i][j];
                temp[1] = temp[0] * temp[0];
                temp[2] = (-2) * lvar[i][j];
                temp[3] = temp[1] / temp[2];
                A[j] = temp[3];
                temp[4] = d_Pi * temp[2]/(-2);
                temp[5] = sqrt(temp[4]);
                B[j] =  log(1/temp[5]);				
            }
        }
		for (j = 0; j < N_feat; j += 8){
        #pragma HLS pipeline II=1
            float temp1 = A[j] + B[j];
            float temp2 = A[j + 1] + B[j + 1];
            float temp3 = A[j + 2] + B[j + 2];
            float temp4 = A[j + 3] + B[j + 3];
            float temp5 = A[j + 4] + B[j + 4];
            float temp6 = A[j + 5] + B[j + 5];
            float temp7 = A[j + 6] + B[j + 6];
            float temp8 = A[j + 7] + B[j + 7];

            float level_1_1 = temp1 + temp2;
            float level_1_2 = temp3 + temp4;
            float level_1_3 = temp5 + temp6;
            float level_1_4 = temp7 + temp8;

            float level_2_1 = level_1_1 + level_1_2;
			#pragma HLS resource variable=level_2_1 core=FAddSub_nodsp

            float level_2_2 = level_1_3 + level_1_4;
            #pragma HLS resource variable=level_2_2 core=FAddSub_nodsp

            float level_3 = level_2_1 + level_2_2;
            #pragma HLS resource variable=level_3 core=FAddSub_nodsp

            numerator += level_3;
            #pragma HLS resource variable=numerator core=FAddSub_nodsp
		}
        if (numerator > max_likelihood){
            max_likelihood = numerator;
            prediction = i;
        }
    }
    return prediction;
}