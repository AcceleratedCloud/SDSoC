#include <cmath>

#include "accelerator.h"

void main_kernel_accel(float data[(int)(chunkSizeMax * numFeatures / 2)], float centroids[numClusters][numFeatures], int chunkSize, float counts_sums[numClusters][1 + numFeatures]){

	float features[numFeatures];
	float dif[numClusters][numFeatures], dist[numClusters];

	for (int i = 0; i < (int)(chunkSize / 2); i++){
#pragma HLS loop_tripcount min=1 max=2500
		
		for (int j = 0; j < numFeatures; j++){
#pragma HLS pipeline II=1
			features[j] = data[i * numFeatures + j];
		}		
		
		for (int j = 0; j < numFeatures; j++){
			for (int k = 0; k < numClusters; k++){
				dif[k][j] = features[j] - centroids[k][j];
			}
		}
		
		for (int k = 0; k < numClusters; k++){		
			dist[k] = 0.0;
		}
		
		for (int j = 0; j < numFeatures; j++){
			for (int k = 0; k < numClusters; k++){
				dist[k] += pow(dif[k][j], 2);
			}	
		}
		
		float minDistance = INFINITY;
		int bestIndex = -1;

		for (int k = 0; k < numClusters; k++){
			if (dist[k] < minDistance){
				minDistance = dist[k];
				bestIndex = k;
			}
		}

		counts_sums[bestIndex][0] += 1;

		for (int j = 0; j < numFeatures; j++){
			counts_sums[bestIndex][j + 1] += features[j];
		}
		
	}

}

void KM_centroids_kernel_accel(float data1[(int)(chunkSizeMax * numFeatures / 2)], float data2[(int)(chunkSizeMax * numFeatures / 2)], float centroids[numClusters * numFeatures], int chunkSize, float counts_sums[numClusters * (1 + numFeatures)]){
	
	int chunkSize1, chunkSize2;
	chunkSize1 = chunkSize;
	chunkSize2 = chunkSize;
	
	float centroids1[numClusters][numFeatures], centroids2[numClusters][numFeatures];
	for (int k = 0; k < numClusters; k++){
		for (int j = 0; j < numFeatures; j++){
#pragma HLS pipeline II=1			
			centroids1[k][j] = centroids[k * numFeatures + j];
		}
	}
	for (int j = 0; j < numFeatures; j++){
		for (int k = 0; k < numClusters; k++){
			centroids2[k][j] = centroids1[k][j];
		}
	}	
	
	float counts_sums1[numClusters][1 + numFeatures], counts_sums2[numClusters][1 + numFeatures];
	for (int k = 0; k < numClusters; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			counts_sums1[k][j] = 0.0;
			counts_sums2[k][j] = 0.0;
		}
	}
	
	main_kernel_accel(data1, centroids1, chunkSize1, counts_sums1);
	main_kernel_accel(data2, centroids2, chunkSize2, counts_sums2);
	
	for (int k = 0; k < numClusters; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
#pragma HLS pipeline II=1			
			counts_sums[k * (1 + numFeatures) + j] = counts_sums1[k][j] + counts_sums2[k][j];
		}
	}
	
}

void KM_centroids_kernel(float data1[(int)(chunkSizeMax * numFeatures / 2)], float data2[(int)(chunkSizeMax * numFeatures / 2)], float centroids[numClusters * numFeatures], int chunkSize, float counts_sums[numClusters * (1 + numFeatures)]){
	
	int _chunkSize;
	_chunkSize = chunkSize;
	
	float _centroids[numClusters][numFeatures];
	for (int k = 0; k < numClusters; k++){
		for (int j = 0; j < numFeatures; j++){			
			_centroids[k][j] = centroids[k * numFeatures + j];
		}
	}	
	
	float _counts_sums[numClusters][1 + numFeatures];
	for (int k = 0; k < numClusters; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			_counts_sums[k][j] = 0.0;
		}
	}
	
	float features[numFeatures];
	
	for (int i = 0; i < (int)(_chunkSize / 2); i++){

		for (int j = 0; j < numFeatures; j++){
			features[j] = data1[i * numFeatures + j];
		}

		float minDistance = INFINITY;
		int bestIndex = -1;

		for (int k = 0; k < numClusters; k++){

			float dist = 0.0;
			for (int j = 0; j < numFeatures; j++){
				dist += pow(features[j] - _centroids[k][j], 2);
			}

			if (dist < minDistance){
				minDistance = dist;
				bestIndex = k;
			}

		}

		_counts_sums[bestIndex][0] += 1;

		for (int j = 0; j < numFeatures; j++){
			_counts_sums[bestIndex][j + 1] += features[j];
		}

	}
	for (int i = 0; i < (int)(_chunkSize / 2); i++){
		
		for (int j = 0; j < numFeatures; j++){
			features[j] = data2[i * numFeatures + j];
		}		
		
		float minDistance = INFINITY;
		int bestIndex = -1;	

		for (int k = 0; k < numClusters; k++){
		
			float dist = 0.0;
			for (int j = 0; j < numFeatures; j++){
				dist += pow(features[j] - _centroids[k][j], 2);
			}

			if (dist < minDistance){
				minDistance = dist;
				bestIndex = k;
			}
			
		}

		_counts_sums[bestIndex][0] += 1;

		for (int j = 0; j < numFeatures; j++){	
			_counts_sums[bestIndex][j + 1] += features[j];
		}
		
	}	
	
	for (int k = 0; k < numClusters; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			counts_sums[k * (1 + numFeatures) + j] = _counts_sums[k][j];
		}
	}
	
}
