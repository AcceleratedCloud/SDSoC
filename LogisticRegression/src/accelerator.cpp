#include <cmath>

#include "accelerator.h"

void main_kernel_accel(float data[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float weights[numClasses][(1 + numFeatures)], int chunkSize, float gradients[numClasses][(1 + numFeatures)]){

	float labels[numClasses], features[(1 + numFeatures)];
	float dot[numClasses], dif[numClasses];

	for (int i = 0; i < (int)(chunkSize / 2); i++){
		
		for (int k = 0; k < numClasses; k++){
			labels[k] = data[i * (numClasses + (1 + numFeatures)) + k];
		}

		for (int j = 0; j < (1 + numFeatures); j++){
			features[j] = data[i * (numClasses + (1 + numFeatures)) + numClasses + j];
		}

		for (int k = 0; k < numClasses; k++){
			dot[k] = 0.0;
		}

		for (int j = 0; j < (1 + numFeatures); j ++){
			for (int k = 0; k < numClasses; k++){
				dot[k] += features[j] * weights[k][j];
			}
		}

		for (int k = 0; k < numClasses; k++){
			float prediction = 1.0 / (1.0 + exp(-dot[k]));
			dif[k] = prediction - labels[k];
		}

		for (int j = 0; j < (1 + numFeatures); j++){
			for (int k = 0; k < numClasses; k++){
				gradients[k][j] += dif[k] * features[j];
			}
		}
		
	}

}

void LR_gradients_kernel_accel(float data1[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float data2[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float weights[numClasses * (1 + numFeatures)], int chunkSize, float gradients[numClasses * (1 + numFeatures)]){

	int chunkSize1, chunkSize2;
	chunkSize1 = chunkSize;
	chunkSize2 = chunkSize1;

	float weights1[numClasses][(1 + numFeatures)], weights2[numClasses][(1 + numFeatures)];
	for (int k = 0; k < numClasses; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			weights1[k][j] = weights[k * (1 + numFeatures) + j];
		}
	}
	for (int j = 0; j < (1 + numFeatures); j++){
		for (int k = 0; k < numClasses; k++){
			weights2[k][j] = weights1[k][j];
		}
	}

	float gradients1[numClasses][(1 + numFeatures)], gradients2[numClasses][(1 + numFeatures)];
	for (int k = 0; k < numClasses; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			gradients1[k][j] = 0.0;
			gradients2[k][j] = 0.0;
		}
	}

	main_kernel_accel(data1, weights1, chunkSize1, gradients1);
	main_kernel_accel(data2, weights2, chunkSize2, gradients2);

	for (int k = 0; k < numClasses; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			gradients[k * (1 + numFeatures) + j] = gradients1[k][j] + gradients2[k][j];
		}
	}

}
	
void LR_gradients_kernel(float data1[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float data2[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float weights[numClasses * (1 + numFeatures)], int chunkSize, float gradients[numClasses * (1 + numFeatures)]){

	int _chunkSize;
	_chunkSize = chunkSize;
	
	float _weights[numClasses][(1 + numFeatures)];
	for (int k = 0; k < numClasses; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			_weights[k][j] = weights[k * (1 + numFeatures) + j];
		}
	}

	float _gradients[numClasses][(1 + numFeatures)];
	for (int k = 0; k < numClasses; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			_gradients[k][j] = 0.0;
		}
	}

	float labels[numClasses], features[(1 + numFeatures)];

	for (int i = 0; i < (int)(_chunkSize / 2); i++){

		for (int k = 0; k < numClasses; k++){
			labels[k] = data1[i * (numClasses + (1 + numFeatures)) + k];
		}

		for (int j = 0; j < (1 + numFeatures); j++){
			features[j] = data1[i * (numClasses + (1 + numFeatures)) + numClasses + j];
		}

		for (int k = 0; k < numClasses; k++){
			
			float dot = 0.0;
			for (int j = 0; j < (1 + numFeatures); j++){
				dot += _weights[k][j] * features[j];
			}
			
			float dif = 1.0 / (1.0 + exp(-dot)) - labels[k];
			
			for (int j = 0; j < (1 + numFeatures); j++){
				_gradients[k][j] += dif * features[j];
			}
			
		}

	}
	for (int i = 0; i < (int)(_chunkSize / 2); i++){

		for (int k = 0; k < numClasses; k++){
			labels[k] = data2[i * (numClasses + (1 + numFeatures)) + k];
		}

		for (int j = 0; j < (1 + numFeatures); j++){
			features[j] = data2[i * (numClasses + (1 + numFeatures)) + numClasses + j];
		}

		for (int k = 0; k < numClasses; k++){
			
			float dot = 0.0;
			for (int j = 0; j < (1 + numFeatures); j++){
				dot += _weights[k][j] * features[j];
			}
			
			float dif = 1.0 / (1.0 + exp(-dot)) - labels[k];
			
			for (int j = 0; j < (1 + numFeatures); j++){
				_gradients[k][j] += dif * features[j];
			}
			
		}

	}

	for (int k = 0; k < numClasses; k++){
		for (int j = 0; j < (1 + numFeatures); j++){
			gradients[k * (1 + numFeatures) + j] = _gradients[k][j];
		}
	}
	
}
